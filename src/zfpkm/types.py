from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["ApproxResult", "DensityResult", "ZFPKMResult"]


@dataclass(slots=True, frozen=True)
class ApproxResult:
    x: npt.NDArray[np.floating]
    y: npt.NDArray[np.floating]


@dataclass(slots=True, frozen=True)
class DensityResult:
    """Results from a kernel density estimation.

    :param x: The x-coordinates of the density estimate.
    :param y: The estimated density values at the x-coordinates.
    :param bw: The bandwidth used.
    :param n: The number of points in the output grid.
    """

    x: npt.NDArray[np.floating]
    y: npt.NDArray[np.floating]
    bw: float
    n: int


@dataclass(slots=True, frozen=True)
class ZFPKMResult:
    name: str
    density: DensityResult
    mu: float
    sd: float
    fpkm_at_mu: float
