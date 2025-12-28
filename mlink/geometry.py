from queue import SimpleQueue

import numpy as np
import numpy.typing as npt
import trimesh


def walls_to_mesh(
    walls: npt.NDArray[np.integer],
    floor_height: float = 0.0,
    ceiling_height: float = 9.0,
) -> trimesh.Trimesh:
    # extracting wall segments
    vwalls = np.pad(walls.astype(np.uint8), pad_width=[(1, 1), (0, 0)], mode="constant")
    vwalls = vwalls[1:, :] - vwalls[:-1, :]

    row_start, col_start = np.nonzero(vwalls == 1)
    row_end, col_end = np.nonzero(vwalls == 255)

    order_start = np.lexsort((row_start, col_start))
    order_end = np.lexsort((row_end, col_end))

    row_start, col_start = row_start[order_start], col_start[order_start]
    row_end, col_end = row_end[order_end], col_end[order_end]

    starts = np.stack([row_start, col_start], axis=1)
    ends = np.stack([row_end - 1, col_end], axis=1)
    vert_segments = np.concatenate((starts, ends), axis=-1)
    vert_segments = vert_segments[vert_segments[:, 2] - vert_segments[:, 0] > 1]

    hwalls = np.pad(walls.astype(np.uint8), pad_width=[(0, 0), (1, 1)], mode="constant")
    hwalls = hwalls[:, 1:] - hwalls[:, :-1]

    row_start, col_start = np.nonzero(hwalls == 1)
    row_end, col_end = np.nonzero(hwalls == 255)

    order_start = np.lexsort((col_start, row_start))
    order_end = np.lexsort((col_end, row_end))

    row_start, col_start = row_start[order_start], col_start[order_start]
    row_end, col_end = row_end[order_end], col_end[order_end]

    starts = np.stack([row_start, col_start], axis=1)
    ends = np.stack([row_end, col_end - 1], axis=1)
    hori_segments = np.concatenate((starts, ends), axis=-1)
    hori_segments = hori_segments[hori_segments[:, 3] - hori_segments[:, 1] > 1]

    wall_segments = np.concatenate((vert_segments, hori_segments), axis=0).reshape(-1, 2, 2)

    # increase dimensionality of wall segments from 2-D to 3-D
    # hard-coded heights for now...
    wall_segments = np.repeat(wall_segments, repeats=2, axis=1)
    num_segments = wall_segments.shape[0]
    # wall_segments = np.concatenate(
    #     (wall_segments, np.zeros(shape=(num_segments, 4, 1))), axis=2
    # )
    # wall_segments[:, 1:3, -1] = ceiling_height

    z = np.full((num_segments, 4, 1), fill_value=floor_height, dtype=np.float32)
    wall_segments = np.concatenate((wall_segments, z), axis=2)
    wall_segments[:, 1:3, -1] = ceiling_height

    # add ceiling + roof
    xmin, xmax = np.min(wall_segments[:, :, 0]), np.max(wall_segments[:, :, 0])
    ymin, ymax = np.min(wall_segments[:, :, 1]), np.max(wall_segments[:, :, 1])

    # for now, we will hard-code the heights of the floor/ceiling.
    # this function should be extended to accept height levels.
    # ALSO, we assume that the floors/ceilings are rectangular.
    wall_segments = np.concatenate(
        (
            wall_segments,
            np.asarray(
                [
                    [
                        [xmin, ymin, floor_height],
                        [xmin, ymax, floor_height],
                        [xmax, ymax, floor_height],
                        [xmax, ymin, floor_height],
                    ],
                    [
                        [xmin, ymin, ceiling_height],
                        [xmin, ymax, ceiling_height],
                        [xmax, ymax, ceiling_height],
                        [xmax, ymin, ceiling_height],
                    ],
                ]
            ),
        ),
        axis=0,
    )

    # convert wall-segments to structures required by trimesh
    wall_segments = wall_segments.reshape(-1, 3)
    vertices, inverse = np.unique(wall_segments, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 4)

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        merge_tex=True,
        merge_norm=True,
    )


def _initialize_wall_map():
    walls = np.zeros(shape=(64, 64), dtype=np.uint8)
    walls[(0, -1), :] = 1
    walls[:, (0, -1)] = 1
    return walls


def partition(
    arr: npt.NDArray[np.uint8],
    min_wall_length: int,
    min_door_length: int,
    max_partitions: int,
    rng: np.random.Generator,
):
    height, width = arr.shape
    queue = SimpleQueue()
    queue.put((0, height, 0, width))
    num_partitions = 0

    while not queue.empty() and num_partitions < max_partitions:
        row0, row1, col0, col1 = queue.get()
        height, width = row1 - row0, col1 - col0

        # cannot split any further
        if min(height, width) <= 2 * min_wall_length:
            continue

        axes = [
            axis
            for axis, size in enumerate([height, width])
            if size > 2 * min_wall_length
        ]
        axis = int(rng.choice(axes))

        if axis == 0:
            cut = rng.integers(row0 + min_wall_length, row1 - min_wall_length)
            queue.put((row0, cut, col0, col1))
            queue.put((cut, row1, col0, col1))

            door_start = rng.integers(col0, col1 - min_door_length)
            door_end = rng.integers(door_start + min_door_length, col1)
            arr[cut, col0:door_start] = 1
            arr[cut, door_end:col1] = 1
            num_partitions += 1
        else:
            cut = rng.integers(col0 + min_wall_length, col1 - min_wall_length)
            queue.put((row0, row1, col0, cut))
            queue.put((row0, row1, cut, col1))
            door_start = rng.integers(row0, row1 - min_door_length)
            door_end = rng.integers(door_start + min_door_length, row1)
            arr[row0:door_start, cut] = 1
            arr[door_end:row1, cut] = 1
            num_partitions += 1

    return


def generate_wall_map(
    dims: tuple[int, int],
    min_wall_length: int,
    min_door_length: int,
    max_partitions: int,
    rng: np.random.Generator,
):
    walls = np.zeros(shape=dims, dtype=np.uint8)
    walls[(0, -1), :] = 1
    walls[:, (0, -1)] = 1

    partition(walls, min_wall_length, min_door_length, max_partitions, rng)
    return walls
