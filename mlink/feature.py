from functools import wraps
from typing import Any, Callable, NamedTuple

import mitsuba as mi
import numpy as np
import numpy.typing as npt
from sionna.rt import RadioMapSolver
from trimesh.intersections import mesh_plane

from mlink.constants import FREE_SPACE_CONSTS
from mlink.cost import _compute_wall_losses, build_transmission_coeffs
from mlink.scene import Scene

"""
Our amazing feature system!

We use a simple DFS algorithm and a feature decorator to construct feature graphs.
This allows us to cache features that might be used in further computations.

Users can implement new features, just make sure to add it to the `REGISTRY` and use the decorator.

All feature functions should have the following signature:

`def feature(scene: mlink.Scene, frequency: float, ...)`
"""


class Specification(NamedTuple):
    name: str
    requires: tuple[str, ...]
    fn: Callable[..., npt.NDArray[np.floating | np.integer]]


REGISTRY: dict[str, Specification] = dict()


def feature(*, name: str, requires: tuple[str, ...] = ()):
    def decorator(fn: Callable[..., npt.NDArray[np.floating | np.integer]]):
        spec = Specification(name, requires, fn)
        REGISTRY[name] = spec

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorator


@feature(name="electrical_distance")
def electrical_distance(scene: Scene, frequency: float):
    tx_coords = scene.antenna_database.tx_coords
    rx_coords = scene.antenna_database.rx_coords

    rx_grid = scene.antenna_database.rx_grid
    if rx_grid is None:
        raise Exception(
            "Receiver coordinates must be initialized with an `AntennaGrid` object for tensorial features."
        )

    distances = np.linalg.norm(tx_coords[:, None, :] - rx_coords[None, :, :], axis=-1)
    distances *= frequency / FREE_SPACE_CONSTS.c
    distances = distances.reshape(-1, 1, *rx_grid.shape)
    return distances


@feature(name="binary_walls")
def binary_walls(scene: Scene, frequency: float) -> npt.NDArray[np.floating]:
    mesh = scene.mesh
    z_normal = np.asarray([0, 0, 1])

    rx_grid = scene.antenna_database.rx_grid
    if rx_grid is None:
        raise Exception("Receivers must be initialized with a grid!")

    def f(lines):
        o = []
        for i, line in enumerate(lines):
            src = rx_grid.xyz2ijk(line[0, :])
            dst = rx_grid.xyz2ijk(line[1, :])
            dif = np.asarray([x - y for x, y in zip(dst, src)])
            dist = np.abs(np.sum(dif)) + 1
            o.append(
                np.linspace(
                    start=src,
                    stop=dst,
                    num=dist,
                    endpoint=True,
                ).astype(np.int32)
            )

        return np.concatenate(o, axis=0)

    wall_tensor_lst = []
    for i in range(rx_grid.shape[0]):
        plane_origin = rx_grid.origin + i * rx_grid.deltas[2]
        lines = mesh_plane(
            mesh,
            plane_normal=z_normal,
            plane_origin=plane_origin,
            return_faces=False,
        )
        wall_idxs = f(lines)
        walls = np.zeros(shape=rx_grid.shape[1:])
        walls[wall_idxs[:, 0], wall_idxs[:, 1]] = 1
        wall_tensor_lst.append(walls)

    wall_tensor = np.stack(wall_tensor_lst, axis=0).astype(np.float32)  # (K,H,W)

    # shape to (1,1,K,H,W) then repeat over tx -> (tx,1,K,H,W)
    wall_maps = wall_tensor[None, None, :, :, :]
    wall_maps = np.repeat(wall_maps, repeats=scene.antenna_database.tx_coords.shape[0], axis=0)
    
    return wall_maps


@feature(name="ray_features", requires=("electrical_distance",))
def ray_features(
    scene: Scene, frequency: float, electrical_distance: npt.NDArray[np.floating]
):
    tx_coords = scene.antenna_database.tx_coords
    rx_coords = scene.antenna_database.rx_coords

    rx_grid = scene.antenna_database.rx_grid
    if rx_grid is None:
        raise Exception(
            "Receiver coordinates must be initialized with an `AntennaGrid` object for tensorial features."
        )

    mi_scene = scene.to_sionna(frequency).mi_scene
    assert isinstance(mi_scene, mi.Scene)

    per_wall_loss = build_transmission_coeffs(mi_scene, frequency)
    total_wall_loss, num_obstructions = _compute_wall_losses(
        mi_scene, tx_coords, rx_coords, per_wall_loss
    )

    total_wall_loss = total_wall_loss.reshape(-1, 1, *rx_grid.shape)
    free_space_loss = 20 * np.log10(4 * np.pi * electrical_distance)
    total_path_loss = -free_space_loss + 10 * np.log10(total_wall_loss)

    num_obstructions = num_obstructions.reshape(-1, 1, *rx_grid.shape)
    return np.concatenate((total_path_loss, num_obstructions), axis=1)


@feature(name="cost", requires=("ray_features",))
def cost(scene: Scene, frequency: float, ray_features: npt.NDArray[np.floating]):
    return ray_features[:, 0:1, ...]


@feature(name="num_obstructions", requires=("ray_features",))
def num_obstructions(
    scene: Scene, frequency: float, ray_features: npt.NDArray[np.floating]
):
    return ray_features[:, 1:2, ...]


@feature(name="rss", requires=())
def rss(scene: Scene, frequency: float):
    rx_grid = scene.antenna_database.rx_grid
    if rx_grid is None:
        raise Exception("`rx_grid` must be defined for tensorial features.")

    rx_coords = scene.antenna_database.rx_coords
    rm_solver = RadioMapSolver()

    sionna=scene.to_sionna(frequency)

    UNIFORM_TX_POWER_DBM = 44

    txs = sionna.transmitters
    tx_iter = txs.values() if isinstance(txs, dict) else txs

    for tx in tx_iter:
        tx.power_dbm = UNIFORM_TX_POWER_DBM

    radio_maps = []
    for i in range(rx_grid.shape[0]):
        rx_plane = rx_coords.reshape((*rx_grid.shape, 3))[i, ...]
        rm = rm_solver(
            scene=sionna,
            samples_per_tx=100000000,
            center=mi.Point3f(rx_plane.mean(axis=(0, 1))),
            size=mi.Point2f(rx_grid.deltas[:2, :2] @ np.asarray(rx_grid.shape[1:])),
            cell_size=mi.Point2f(rx_grid.deltas[:2, :2] @ np.array([1, 1])),
            orientation=mi.Point3f([0, 0, 0]),
            max_depth=32,
            seed=431,
        )
        rm = 10.0 * np.log10(rm.rss.numpy()) + 30.0
        rm = rm.reshape(-1, 1, 1, *rx_grid.shape[1:])
        radio_maps.append(rm)

    radio_maps = np.concatenate(radio_maps, axis=2)
    return np.transpose(radio_maps, axes=(0, 1, 2, 4, 3))


@feature(name = "height_cond")
def height_cond(scene: Scene, frequency: float) -> npt.NDArray[np.floating]:
    rx_grid = scene.antenna_database.rx_grid 
    if rx_grid is None:
        raise Exception("`rx_grid` must be defined for tensorial features.")
    
    num_tx = scene.antenna_database.tx_coords.shape[0]
    K, H, W = rx_grid.shape

    z_min = float(scene.mesh.bounds[0, 2])
    z_max = float(scene.mesh.bounds[1, 2])
    room_h = max(z_max - z_min, 1e-6)

    k = np.arange(K, dtype = np.float32)
    z_slices = (rx_grid.origin[None, :] + k[:, None] * rx_grid.deltas[2][None, :])[:, 2]

    z_rel = (z_slices - z_min)/room_h 
    d_floor = (z_slices - z_min)
    d_ceil = (z_max - z_slices)

    cond = np.stack([z_rel, d_floor, d_ceil], axis = 0)
    cond = cond[None, :, :, None, None]
    cond = np.broadcast_to(cond, (num_tx, 3, K, H, W)).astype(np.float32)
    return cond


def build_feature_tensor(scene: Scene, frequency: float, requested: list[str]):
    """
    Build the feature tensor using DFS. Any feature name in the `requested` list must be present in the `REGISTRY`.

    Parameters
    ----------
    scene : `mlink.Scene`
    frequency : `float`
    requested : `list[str]`

    Returns
    -------
    `npt.NDArray[np.floating]`
    """
    if not all([request in REGISTRY for request in requested]):
        raise Exception("Requested feature is not implemented.")

    visited = set()
    order = []

    def dfs(name: str) -> None:
        if name in visited:
            return

        visited.add(name)
        for dep in REGISTRY[name].requires:
            dfs(dep)

        order.append(name)

    for name in requested:
        dfs(name)

    cache: dict[str, npt.NDArray[Any]] = dict()
    for name in order:
        spec = REGISTRY[name]
        kwargs = {k: cache[k] for k in spec.requires}
        cache[name] = spec.fn(scene, frequency, **kwargs)

    feature_tensor = np.concatenate([cache[name] for name in requested], axis=1)
    return feature_tensor
