"""Shared numeric helpers for model ``guess_initial`` implementations."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from core.numerics import find_closest_index


def edge_intensities(
    x: NDArray,
    y: NDArray,
    start: int | float,
    stop: int | float,
    mode: Literal["value", "index"] = "index",
    avg_on: int = 3,
) -> dict[str, float]:
    """
    Average intensities just outside a region slice.

    Parameters
    ----------
    x : NDArray
        Full spectrum x-axis.
    y : NDArray
        Full spectrum intensities.
    start : int or float
        Region start (index or axis value).
    stop : int or float
        Region stop (index or axis value).
    mode : {"value", "index"}, optional
        How to interpret ``start`` / ``stop``.
    avg_on : int, optional
        Number of points to average before ``start`` and after ``stop``.

    Returns
    -------
    dict[str, float]
        Keys ``i1`` (near start) and ``i2`` (near stop).
    """
    if mode == "value":
        start = find_closest_index(float(start), x)
        stop = find_closest_index(float(stop), x)

    start_i = int(start)
    stop_i = int(stop)

    i1_arr = y[max(start_i - avg_on, 0) : start_i]
    i2_arr = y[stop_i : min(stop_i + avg_on, len(y))]

    i1 = float(np.mean(i1_arr)) if len(i1_arr) > 0 else float(y[start_i])
    i2 = float(np.mean(i2_arr)) if len(i2_arr) > 0 else float(y[stop_i - 1])

    return {"i1": i1, "i2": i2}


def peak_index_from_residuals(residuals: NDArray) -> int:
    """
    Return the index of the maximum residual.

    Parameters
    ----------
    residuals : NDArray
        Residual intensities over a region.

    Returns
    -------
    int
        Index of the peak maximum in ``residuals``.
    """
    return int(np.argmax(residuals))


def half_max_sigma(x: NDArray, y: NDArray, peak_index: int) -> float:
    """
    Guess peak half-width from half-maximum crossings around ``peak_index``.

    When either side of the half-maximum cannot be found, returns ``1.0``.

    Parameters
    ----------
    x : NDArray
        Region x-axis.
    y : NDArray
        Region intensities.
    peak_index : int
        Index of the peak maximum.

    Returns
    -------
    float
        Approximate sigma (half of the FWHM span).
    """
    half_max = (y[peak_index] - y.min()) / 2 + y.min()
    left = np.where(y[:peak_index] <= half_max)[0]
    right = np.where(y[peak_index:] <= half_max)[0]
    if left.size == 0 or right.size == 0:
        return 1.0
    l_hm_idx = int(left[-1])
    r_hm_idx = int(right[0]) + peak_index
    return float((x[r_hm_idx] - x[l_hm_idx]) / 2)


def amp_from_height(y: NDArray, peak_index: int, sig: float, frac: float) -> float:
    """
    Guess pseudo-Voigt amplitude from peak height, sigma, and fraction.

    Parameters
    ----------
    y : NDArray
        Region intensities.
    peak_index : int
        Index of the peak maximum.
    sig : float
        Width parameter.
    frac : float
        Lorentzian fraction.

    Returns
    -------
    float
        Amplitude estimate.
    """
    shape_mult = frac / np.pi + (1 - frac) * np.sqrt(np.log(2) / np.pi)
    return float((y[peak_index] - y.min()) * sig / shape_mult)
