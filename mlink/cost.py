from math import pi

import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt

from mlink.constants import FREE_SPACE_CONSTS


def build_transmission_coeffs(mi_scene: mi.Scene, frequency: float) -> dr.auto.ad.Float:
    """
    Compute the transmission coefficients using the finite-thickness formula from ITU. We compute this from Mitsuba's scene object.

    Parameters
    ----------
    mi_scene : `mi.Scene`
    frequency : float

    Returns
    -------
    `dr.auto.ad.Float`
    """
    transmission_coeffs = np.empty(shape=len(mi_scene.shapes()))
    for i, shape in enumerate(mi_scene.shapes()):
        params = mi.traverse(shape)
        eta = params["bsdf.eta_r"][0]
        sigma = params["bsdf.sigma"][0]
        d = params["bsdf.d"][0]

        eta_complex = complex(
            eta, -sigma / (2 * pi * frequency * FREE_SPACE_CONSTS.eps)
        )
        r = (1 - np.sqrt(eta_complex)) / (1 + np.sqrt(eta_complex))
        q = 2 * pi * d * frequency / FREE_SPACE_CONSTS.c
        t = (np.square(1 - r) * np.exp(-1j * q)) / (1 - np.square(r) * np.exp(-2j * q))
        transmission_coeffs[i] = np.abs(t)

    return dr.auto.ad.Float(transmission_coeffs)


def build_rays(
    tx_coords_np: npt.NDArray[np.floating], rx_coords_np: npt.NDArray[np.floating]
) -> tuple[mi.Ray3f, mi.Point3f]:
    """
    Construct a Mitsuba ray object for each receiver and transmitter pairing.

    Parameters
    ----------
    tx_coords_np : `npt.NDArray[np.floating]`
    rx_coords_np : `npt.NDArray[np.floating]`

    Returns
    -------
    `tuple[mi.Ray3f, mi.Point3f]`
    """
    num_transmitters = tx_coords_np.shape[0]
    num_receivers = rx_coords_np.shape[0]

    tx_coords = mi.Point3f(tx_coords_np.T)
    rx_coords = mi.Point3f(rx_coords_np.T)

    srcs = dr.repeat(tx_coords, num_receivers)
    dsts = dr.tile(rx_coords, num_transmitters)

    dvec = dsts - srcs
    dist = dr.norm(dvec)
    dirs = mi.Vector3f(dvec * dr.rcp(dist))

    rays = mi.Ray3f(o=srcs, d=dirs)
    rays = mi.Ray3f(rays, maxt=dist)
    return rays, dsts


def _compute_wall_losses(
    scene: mi.Scene,
    tx_coords: npt.NDArray[np.floating],
    rx_coords: npt.NDArray[np.floating],
    per_wall_loss: dr.auto.ad.Float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
    """
    Compute the wall losses using the COST-231 model.

    Parameters
    ----------
    scene : `mi.Scene`
    tx_coords : `npt.NDArray[np.floating]`
    rx_coords : `npt.NDArray[np.floating]`
    per_wall_loss : `dr.auto.ad.Float`

    Returns
    -------
    `tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]`
    """
    rays, dsts = build_rays(tx_coords, rx_coords)

    is_ray_active = dr.ones(dr.auto.ad.Bool, rays.o.shape[1])
    total_wall_losses = dr.ones(dr.auto.ad.Float, rays.o.shape[1])

    # not good memory usage.
    obstruction_coeffs = dr.ones(dr.auto.ad.Int, rays.o.shape[1])
    num_obstructions = dr.zeros(dr.auto.ad.Int, rays.o.shape[1])

    while dr.any(is_ray_active):
        intersections = scene.ray_intersect_preliminary(rays, is_ray_active)
        hit = is_ray_active & intersections.is_valid()

        if not dr.any(hit):
            break

        num_obstructions += dr.gather(
            dtype=dr.auto.ad.Int,
            source=obstruction_coeffs,
            index=intersections.shape_index,
            active=hit,
        )
        total_wall_losses[hit] *= dr.gather(
            dtype=dr.auto.ad.Float,
            source=per_wall_loss,
            index=intersections.shape_index,
            active=hit,
        )

        # need full ray intersection to property compute new rays
        surface_interaction = intersections.compute_surface_interaction(
            rays, active=hit
        )
        rays = surface_interaction.spawn_ray_to(dsts)
        is_ray_active = hit

    return np.asarray(total_wall_losses), np.asarray(num_obstructions)
