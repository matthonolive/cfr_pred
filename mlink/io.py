import csv
import re
from collections import defaultdict
from enum import auto, Enum
from pathlib import Path

import numpy as np
import polars as pl
from trimesh import Trimesh

from mlink.shapely_ops import (
    build_polygon,
    build_basis,
    extract_vertices,
    is_triangle,
    projected_triangulate,
    projected_difference,
)


class IdaSection(Enum):
    IGNORE = auto()
    MATERIAL = auto()
    WALLS = auto()


class IdaParser:
    def __init__(self, path: Path, chunk_size: int = -1):
        self.file_obj = open(path, mode="r", buffering=chunk_size, encoding="utf-8")

        self.state = IdaSection.IGNORE
        self.static_properties = dict()
        self.freq_properties = defaultdict(list)
        self.triangles = []

    def parse(self) -> None:
        """
        This method parses the IDA file pointed at by `self.file_obj`.
        It effectively populates the `self.static_properties`, `self.freq_properties`, and `self.triangles` attributes.
        By default, we use no buffering to load the file in.

        The actual parsing behaviour is implemented in `self._parse_ignore`, `self._parse_material` and `self._parse_wall`.
        """
        while True:
            line = self.file_obj.readline()
            if len(line) == 0:
                break
            line = line.strip()

            if self.state == IdaSection.IGNORE:
                self._parse_ignore(line)
            elif self.state == IdaSection.MATERIAL:
                self._parse_material(line)
            elif self.state == IdaSection.WALLS:
                self._parse_wall(line)
            else:
                pass

    def export(self) -> tuple[Trimesh, pl.DataFrame, dict[int, int]]:
        """
        This method takes a populated parser object and adjusts the data formatting for `Scene` construction.

        Returns
        -------
        mesh : trimesh.Trimesh
        material_database : pl.DataFrame
        face2material : dict[int, int]
        """
        # build material_database
        for material_id in self.freq_properties["id"]:
            name, thickness = self.static_properties[material_id]
            self.freq_properties["name"].append(name)
            self.freq_properties["thickness"].append(thickness)

        material_database = pl.DataFrame(data=self.freq_properties)

        # build trimesh mesh + face2mat dict
        face2material = dict()
        coords2idx = dict()
        num_unique_coords = 0
        triangles_with_idxs = []

        for i, triangle_data in enumerate(self.triangles):
            face_id, material_id, triangle_coords = triangle_data
            face2material[i] = material_id
            triangle_coords_idxs = []

            for coords in triangle_coords[:-1]:
                coords_tuple = tuple(coords.tolist())
                if coords_tuple in coords2idx:
                    triangle_coords_idxs.append(coords2idx[coords_tuple])
                else:
                    triangle_coords_idxs.append(num_unique_coords)
                    coords2idx[coords_tuple] = num_unique_coords
                    num_unique_coords += 1

            triangles_with_idxs.append(triangle_coords_idxs)

        vertices = np.asarray(list(coords2idx.keys()))
        faces = np.asarray(triangles_with_idxs)

        mesh = Trimesh(vertices, faces)
        return mesh, material_database, face2material

    def __close__(self):
        self.file_obj.close()

    def _parse_ignore(self, line: str) -> None:
        if line == "BEGIN_MATERIAL":
            self.state = IdaSection.MATERIAL
        elif line == "BEGIN_WALLS":
            self.state = IdaSection.WALLS

        return

    def _parse_material(self, line: str) -> None:
        if len(line) == 0 or line.startswith("*"):
            return
        elif line == "END_MATERIAL":
            self.state = IdaSection.IGNORE
            return

        fields = IdaParser._tokenize(line)
        id = int(fields[1])
        entry_type = fields[2]

        if entry_type == "GENERAL":
            name = fields[3]
            thickness = float(fields[4]) * 1e-2
            self.static_properties[id] = name, thickness

        elif entry_type == "FREQUENCY":
            vals = map(float, fields[3:13])
            self.freq_properties["id"].append(id)
            self.freq_properties["frequency"].append(next(vals) * 1e6)
            self.freq_properties["permittivity"].append(next(vals))
            self.freq_properties["permeability"].append(next(vals))
            self.freq_properties["conductivity"].append(next(vals))
            self.freq_properties["transmission_loss_vertical"].append(next(vals))
            self.freq_properties["transmission_loss_horizontal"].append(next(vals))
            self.freq_properties["diffraction_loss_min"].append(next(vals))
            self.freq_properties["diffraction_loss_max"].append(next(vals))
            self.freq_properties["diffraction_loss"].append(next(vals))
        else:
            raise IOError(f"Invalid material entry type: {entry_type}.")

        return

    def _parse_wall(self, line: str) -> None:
        if len(line) == 0 or line.startswith("*"):
            return
        elif line == "END_WALLS":
            self.state = IdaSection.IGNORE
            return

        fields = re.split(pattern=r"[,\s]+", string=line)
        # early return for number of defined walls
        if len(fields) == 1:
            return

        wall_id = int(fields[0])
        num_coords = int(fields[1])
        coords = np.asarray(fields[2 : 2 + 3 * num_coords], dtype=float).reshape(-1, 3)
        material_id = int(fields[2 + 3 * num_coords])
        num_subdivisions = int(fields[3 + 3 * num_coords])
        polygon_parent = build_polygon(exterior=coords, interior=np.empty(shape=(0, 3)))
        basis = build_basis(polygon_parent)

        children = []
        for i in range(num_subdivisions):
            line = self.file_obj.readline()
            if len(line) == 0:
                raise IOError("Incorrect number of subdivisions.")

            subdivision_fields = re.split(pattern=r"[,\s]+", string=line)
            subdivision_wall_id = int(subdivision_fields[0])
            subdivision_num_coords = int(subdivision_fields[1])
            subdivision_coords = np.asarray(
                subdivision_fields[2 : 2 + 3 * subdivision_num_coords], dtype=float
            ).reshape(-1, 3)
            subdivision_material_id = int(
                subdivision_fields[2 + 3 * subdivision_num_coords]
            )
            children.append(
                (
                    subdivision_wall_id,
                    subdivision_material_id,
                    build_polygon(
                        exterior=subdivision_coords, interior=np.empty(shape=(0, 3))
                    ),
                )
            )

        for child in children:
            child_wall_id, child_material_id, polygon_child = child
            polygon_parent = projected_difference(polygon_parent, polygon_child, basis)

            if not is_triangle(polygon_child):
                triangles_child = projected_triangulate(polygon_child, basis)
                for triangle in triangles_child:
                    exterior, _ = extract_vertices(triangle)
                    self.triangles.append(child_wall_id, child_material_id, exterior)
            else:
                exterior, _ = extract_vertices(polygon_child)
                self.triangles.append((child_wall_id, child_material_id, coords))

        if not is_triangle(polygon_parent):
            triangles_parent = projected_triangulate(polygon_parent, basis)
            for triangle in triangles_parent:
                exterior, _ = extract_vertices(triangle)
                self.triangles.append((wall_id, material_id, exterior))
        else:
            exterior, _ = extract_vertices(polygon_parent)
            self.triangles.append((wall_id, material_id, exterior))

        return

    @staticmethod
    def _tokenize(s: str) -> list[str]:
        reader = csv.reader([s], delimiter=" ", quotechar='"', skipinitialspace=True)
        x = next(reader)
        return x
