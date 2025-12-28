from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class AntennaGrid:
    origin: npt.NDArray[np.floating]
    deltas: npt.NDArray[np.floating]
    shape: tuple[int, int, int]

    def ijk2xyz(
        self,
        i: npt.NDArray[np.floating],
        j: npt.NDArray[np.floating],
        k: npt.NDArray[np.floating],
    ):
        """
        Convert array indices to Cartesian coordinates.

        Parameters
        ----------
        i : `npt.NDArray[np.floating]`
        j : `npt.NDArray[np.floating]`
        k : `npt.NDArray[np.floating]`

        Returns
        -------
        `npt.NDArray[np.floating]`
        """
        return (
            self.origin
            + i[..., None] * self.deltas[0]
            + j[..., None] * self.deltas[1]
            + k[..., None] * self.deltas[2]
        )

    def xyz2ijk(self, xyz: npt.NDArray[np.floating]):
        """
        Convert Cartesian coordinates to array indices.

        Parameters
        ----------
        i : `npt.NDArray[np.floating]`
        j : `npt.NDArray[np.floating]`
        k : `npt.NDArray[np.floating]`

        Returns
        -------
        `npt.NDArray[np.floating]`
        """
        frac = self.inv @ (xyz - self.origin)
        i, j, k = np.rint(frac).astype(int)
        return i, j, k

    @cached_property
    def inv(self):
        return np.linalg.inv(self.deltas)

    @classmethod
    def from_bbox(cls, bbox_extends):
        raise NotImplementedError()


class AntennaDatabase(NamedTuple):
    tx_coords: npt.NDArray[np.floating]
    rx_coords: npt.NDArray[np.floating]

    tx_grid: AntennaGrid | None = None
    rx_grid: AntennaGrid | None = None

    @classmethod
    def from_grid(cls, tx_grid: AntennaGrid, rx_grid: AntennaGrid):
        """
        Constructs an instance of `AntennaDatabase` from two `AntennaGrid` objects.
        The `AntennaGrid` objects defined a grid of coordinates for both transmitter and receivers.
        These integer grids are converted to Cartesian coordinates and used to populate `tx_coords` and `rx_coords`.

        Parameters
        ----------
        tx_grid : `AntennaGrid`
        rx_grid : `AntennaGrid`

        Returns
        -------
        `AntennaDatabase`
        """
        d, h, w = tx_grid.shape
        k, i, j = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
        tx_coords = tx_grid.ijk2xyz(i, j, k).reshape(-1, 3)

        d, h, w = rx_grid.shape
        k, i, j = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
        rx_coords = rx_grid.ijk2xyz(i, j, k).reshape(-1, 3)

        return cls(tx_coords, rx_coords, tx_grid, rx_grid)

    @classmethod
    def from_coords(
        cls, tx_coords: npt.NDArray[np.floating], rx_coords: npt.NDArray[np.floating]
    ):
        """
        Constructs an instance of `AntennaDatabase` directly from two coordinate array objects.
        The coordinate arrays should each have shape (Nx, 3), where Nx corresponds to either the number of receivers or number of transmitters.

        Parameters
        ----------
        tx_coords : `npt.NDArray[np.floating]`
        rx_coords : `npt.NDArray[np.floating]`

        Returns
        -------
        `AntennaDatabase`
        """
        return cls(tx_coords, rx_coords, None, None)
