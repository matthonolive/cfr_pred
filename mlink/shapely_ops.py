import numpy as np
import numpy.typing as npt
from shapely import Polygon
from shapely.ops import triangulate


def is_triangle(polygon: Polygon) -> bool:
    """
    Boolean test if a polygon is a simple triangle.

    Parameters
    ----------
    polygon : shapely.Polygon

    Returns
    -------
    bool
    """
    return polygon.is_simple and len(polygon.exterior.coords) == 4


def extract_vertices(
    polygon: Polygon,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Extracts vertices of polygon into NumPy array.

    Parameters
    ----------
    polygon : shapely.Polygon

    Returns
    -------
    exterior_vertices : npt.NDArray[np.floating]
    interior_vertices : npt.NDArray[np.floating]
    """
    exterior = np.asarray(polygon.exterior.coords)

    if len(polygon.interiors) > 0:
        interior = np.asarray([list(ring.coords) for ring in polygon.interiors])
    else:
        interior = np.empty(shape=(0, exterior.shape[1]))

    return exterior, interior


def build_polygon(
    exterior: npt.NDArray[np.floating], interior: npt.NDArray[np.floating]
):
    """
    Build a Polygon object from exterior and interior coordinates.

    Parameters
    __________
    exterior : npt.NDArray[np.floating]
    interior : npt.NDArray[np.floating]

    Returns
    -------
    shapely.Polygon
    """
    return (
        Polygon(shell=exterior)
        if interior.size == 0
        else Polygon(shell=exterior, holes=interior)
    )


def build_basis(polygon: Polygon) -> npt.NDArray[np.floating]:
    """
    Build 2-D basis for surface defined on simple polygon.
    1. Compute the normal of the polygon face.
    2. Compute the unit vector between the 1st and 2nd vertices.
    3. Compute the cross product between (1) and (2).

    Parameters
    ----------
    polygon : shapely.Polygon

    Returns
    -------
    npt.NDArray[np.floating]
    """
    exterior, _ = extract_vertices(polygon)
    p0, p1, p2 = np.vsplit(exterior[:3], 3)
    v1, v2 = p1 - p0, p2 - p0
    normal = np.cross(v1, v2)

    if np.allclose(normal, 0):
        raise Exception("Normal vector is too close to zero.")

    normal /= np.linalg.norm(normal)
    u = v1 / np.linalg.norm(v1)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    return np.asarray([p0, u, v]).squeeze()


def project_to_2d(polygon: Polygon, basis: npt.NDArray[np.floating] | None = None):
    """
    Project the exterior and interior coordinates of a Polygon into a 2-D space.

    Parameters
    ----------
    polygon : shapely.Polygon
    basis : npt.NDArray[np.floating] | None

    Returns
    -------
    shapely.Polygon
    """
    basis = basis if basis is not None else build_basis(polygon)
    exterior, interior = extract_vertices(polygon)
    exterior_projected = (exterior - basis[0]) @ basis[1:].T
    interior_projected = (interior - basis[0]) @ basis[1:].T
    return build_polygon(exterior_projected, interior_projected)


def project_to_3d(polygon: Polygon, basis: npt.NDArray[np.floating]) -> Polygon:
    """
    Project the exterior and interior coordinates of a Polygon into a 3-D space.

    Parameters
    ----------
    polygon : shapely.Polygon
    basis : npt.NDArray[np.floating] | None

    Returns
    -------
    shapely.Polygon
    """
    exterior, interior = extract_vertices(polygon)
    exterior_projected = (exterior @ basis[1:]) + basis[0]
    interior_projected = (interior @ basis[1:]) + basis[0]
    return build_polygon(exterior_projected, interior_projected)


def projected_triangulate(
    polygon: Polygon,
    basis: npt.NDArray[np.floating] | None = None,
) -> list[Polygon]:
    """
    Extended triangulate that works on Polygons with 3-D coordinates.

    Parameters
    ----------
    polygon : shapely.Polygon
    basis : npt.NDArray[np.floating]

    Returns
    -------
    list[shapely.Polygon]
    """
    basis = basis if basis is not None else build_basis(polygon)
    polygon_2d = project_to_2d(polygon, basis)
    triangles = triangulate(polygon_2d)
    return [project_to_3d(triangle, basis) for triangle in triangles]


def projected_difference(
    parent: Polygon, child: Polygon, basis: npt.NDArray[np.floating] | None = None
) -> Polygon:
    """
    Extended difference that works on Polygons with 3-D coordinates.

    Parameters
    ----------
    polygon : shapely.Polygon
    basis : npt.NDArray[np.floating]

    Returns
    -------
    list[shapely.Polygon]
    """
    basis = basis if basis is not None else build_basis(parent)
    parent_2d = project_to_2d(parent, basis)
    child_2d = project_to_2d(child, basis)
    diff_2d = parent_2d.difference(child_2d)
    assert isinstance(diff_2d, Polygon)
    return project_to_3d(diff_2d, basis)
