from collections import defaultdict
from dataclasses import dataclass
from functools import reduce

import mitsuba as mi
import numpy as np
import polars as pl
import sionna.rt
import trimesh
from trimesh import Trimesh

from mlink.antenna import AntennaDatabase


SIONNA_SCHEMA = {
    "eta_r": "permittivity",
    "sigma": "conductivity",
    "d": "thickness",
    "s": "scattering_coefficient",
}


@dataclass
class Scene:
    mesh: Trimesh
    material_database: pl.DataFrame
    face2material: dict[int, int]
    antenna_database: AntennaDatabase
    sionna_scene: sionna.rt.Scene | None = None
    sionna_scene_geometry: sionna.rt.Scene | None = None

    def to_sionna(self, frequency: float):
        if self.sionna_scene is not None and self.sionna_scene.frequency == frequency:
            return self.sionna_scene

        material2face = defaultdict(list)
        for face_id, material_id in self.face2material.items():
            material2face[material_id].append(face_id)

        meshes = []
        for material_id, face_list in material2face.items():
            material_data = self.material_database.filter(
                (pl.col("id") == material_id) & (pl.col("frequency") == frequency)
            ).head(1)

            sionna_material = sionna.rt.RadioMaterial(
                name=material_data[0, "name"],
                thickness=material_data[0, "thickness"],
                relative_permittivity=material_data[0, "permittivity"],
                conductivity=material_data[0, "conductivity"],
            )

            # holder = sionna.rt.HolderMaterial(props=mi.Properties())
            # holder.radio_material = sionna_material

            single_material_meshes = self.mesh.submesh([face_list], append=False)
            assert isinstance(single_material_meshes, list)
            single_material_meshes = [
                trimesh2mitsuba(m, sionna_material) for m in single_material_meshes
            ]
            meshes.extend(single_material_meshes)

        assert len(meshes) > 0
        mi_scene = mi.load_dict(
            {
                "type": "scene",
                "integrator": {"type": "path"},
            }
            | {f"mesh_{i}": mesh for i, mesh in enumerate(meshes)}
        )

        assert isinstance(mi_scene, mi.Scene)
        si_scene = sionna.rt.Scene(mi_scene)

        si_scene.tx_array = sionna.rt.PlanarArray(
            num_rows=1, num_cols=1, pattern="iso", polarization="V"
        )
        si_scene.rx_array = si_scene.tx_array

        for i, tx_coord in enumerate(self.antenna_database.tx_coords):
            name = f"transmitter_{i:03d}"
            position = mi.Point3f(tx_coord)
            transmitter = sionna.rt.Transmitter(name, position)
            si_scene.add(transmitter)

        for i, rx_coord in enumerate(self.antenna_database.rx_coords):
            name = f"receiver_{i:03d}"
            position = mi.Point3f(rx_coord)
            receiver = sionna.rt.Receiver(name, position)
            si_scene.add(receiver)

        si_scene.frequency = frequency
        self.sionna_scene = si_scene
        return si_scene

    @classmethod
    def from_sionna(cls, scene: sionna.rt.Scene):
        # --- get frequency as a float ---
        freq = scene.frequency
        try:
            freq_val = float(freq)
        except Exception:
            freq_val = float(np.asarray(freq).ravel()[0])

        # --- access underlying Mitsuba scene (Sionna RT stores it here) ---
        mi_scene = getattr(scene, "mi_scene", None)
        if mi_scene is None:
            # fallback for some builds
            mi_scene = getattr(scene, "_scene", None)
        if mi_scene is None:
            raise AttributeError("Could not access underlying Mitsuba scene (scene.mi_scene).")

        # --- extract geometry + per-shape material params from Mitsuba shapes ---
        material_rows = []
        face_to_material = {}
        face_count = 0
        submeshes = []

        shapes = list(mi_scene.shapes())
        for mat_id, shape in enumerate(shapes):
            # Only handle mesh-like shapes
            if not hasattr(shape, "face_count") or not hasattr(shape, "vertex_count"):
                continue

            tmsh = mitsuba2trimesh(shape)
            submeshes.append(tmsh)

            # map faces to this "material id"
            for _ in range(tmsh.faces.shape[0]):
                face_to_material[face_count] = mat_id
                face_count += 1

            # Pull Sionna radio-material parameters from the shape's bsdf
            params = mi.traverse(shape)

            def get_scalar(key: str, default: float) -> float:
                if key not in params:
                    return float(default)
                v = params[key]
                return float(np.asarray(v).ravel()[0])

            # Try to get a nice name (optional)
            name = None
            if hasattr(shape, "id"):
                try:
                    name = shape.id()
                except Exception:
                    pass
            if not isinstance(name, str) or len(name) == 0:
                name = f"shape_{mat_id}"

            material_rows.append(
                {
                    "id": mat_id,
                    "frequency": freq_val,
                    "name": name,
                    # These are the important ones used by Scene.to_sionna() and cost.py:
                    "thickness": get_scalar("bsdf.d", 0.1),
                    "permittivity": get_scalar("bsdf.eta_r", 4.0),
                    "conductivity": get_scalar("bsdf.sigma", 0.01),
                }
            )

        # Concatenate all extracted meshes
        if len(submeshes) == 0:
            mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
        else:
            mesh = reduce(trimesh.util.concatenate, submeshes)

        material_database = pl.DataFrame(material_rows)

        # --- extract TX/RX coordinates if present in scene ---
        if len(scene.receivers.values()) > 0:
            rx_coords = np.concatenate([rx.position for rx in scene.receivers.values()]).reshape(-1, 3)
        else:
            rx_coords = np.empty((0, 3), dtype=np.float32)

        if len(scene.transmitters.values()) > 0:
            tx_coords = np.concatenate([tx.position for tx in scene.transmitters.values()]).reshape(-1, 3)
        else:
            tx_coords = np.empty((0, 3), dtype=np.float32)

        antenna_database = AntennaDatabase.from_coords(tx_coords, rx_coords)

        return Scene(
            mesh=mesh,
            material_database=material_database,
            antenna_database=antenna_database,
            face2material=face_to_material,
        )
    
    
    def to_sionna_geometry(self, frequency: float):
        """
        Build geometry + radio materials ONLY.
        Does NOT add transmitters/receivers.
        """
        if self.sionna_scene_geometry is not None and self.sionna_scene_geometry.frequency == frequency:
            return self.sionna_scene_geometry

        material2face = defaultdict(list)
        for face_id, material_id in self.face2material.items():
            material2face[material_id].append(face_id)

        meshes = []
        for material_id, face_list in material2face.items():
            material_data = (
                self.material_database
                .filter((pl.col("id") == material_id) & (pl.col("frequency") == frequency))
                .head(1)
            )
            if material_data.height == 0:
                raise ValueError(f"No material row for id={material_id}, frequency={frequency}")

            sionna_material = sionna.rt.RadioMaterial(
                name=material_data[0, "name"],
                thickness=material_data[0, "thickness"],
                relative_permittivity=material_data[0, "permittivity"],
                conductivity=material_data[0, "conductivity"],
            )

            single_material_meshes = self.mesh.submesh([face_list], append=False)
            assert isinstance(single_material_meshes, list)
            meshes.extend([trimesh2mitsuba(m, sionna_material) for m in single_material_meshes])

        assert len(meshes) > 0

        mi_scene = mi.load_dict(
            {"type": "scene", "integrator": {"type": "path"}}
            | {f"mesh_{i}": mesh for i, mesh in enumerate(meshes)}
        )
        assert isinstance(mi_scene, mi.Scene)

        si_scene = sionna.rt.Scene(mi_scene)
        si_scene.tx_array = sionna.rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        si_scene.rx_array = si_scene.tx_array
        si_scene.frequency = frequency

        self.sionna_scene_geometry = si_scene
        return si_scene




def mitsuba2trimesh(mesh: mi.Mesh) -> Trimesh:
    num_faces = mesh.face_count()
    faces = np.asarray(mesh.faces_buffer()).reshape(num_faces, 3)

    num_vertices = mesh.vertex_count()
    vertices = np.asarray(mesh.vertex_positions_buffer()).reshape(num_vertices, 3)
    vertex_normals = (
        np.asarray(mesh.vertex_normals_buffer()).reshape(num_vertices, 3)
        if mesh.has_vertex_normals()
        else None
    )

    return Trimesh(vertices, faces, vertex_normals=vertex_normals)


def trimesh2mitsuba(mesh: Trimesh, material: sionna.rt.RadioMaterialBase) -> mi.Mesh:
    num_vertices = mesh.vertices.shape[0]
    num_faces = mesh.faces.shape[0]

    has_vertex_normals = mesh.vertex_normals is not None

    # holder = sionna.rt.HolderMaterial(mi.Properties())
    # holder.radio_material = material

    props = mi.Properties()
    props["bsdf"] = material

    mi_mesh = mi.Mesh(
        "mesh", num_vertices, num_faces, props, has_vertex_normals=has_vertex_normals
    )
    mesh_params = mi.traverse(mi_mesh)
    for k in mesh_params.keys():
        if k == "faces":
            mesh_params[k] = np.asarray(mesh.faces, dtype=np.int32).flatten()
        elif k == "vertex_positions":
            mesh_params[k] = np.asarray(mesh.vertices, dtype=np.float32).flatten()
        elif has_vertex_normals and k == "vertex_normals":
            mesh_params[k] = np.asarray(mesh.vertex_normals, dtype=np.float32).flatten()

    mesh_params.update()
    mi_mesh.initialize()
    return mi_mesh
