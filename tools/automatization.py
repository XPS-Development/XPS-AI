"""
Set of useful methods for routine tasks.

Provides functions for guessing parameters.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from core.types import ComponentLike, RegionLike

from .evaluation import region_bundle


def guess_pseudo_voigt_sig_parameter(x: NDArray, y: NDArray, max_idx: int) -> float:
    """Guess sigma parameter for pseudo-voigt model at max_idx.

    Parameters
    ----------
    x : NDArray
        X values.
    y : NDArray
        Y values.
    max_idx : int
        Index of the peak maximum.

    Returns
    -------
    float
        Sigma parameter for pseudo-voigt model at max_idx.
    """

    half_max = (y[max_idx] - y.min()) / 2 + y.min()
    try:
        l_hm_idx = np.where(y[:max_idx] <= half_max)[0][-1]
        r_hm_idx = np.where(y[max_idx:] <= half_max)[0][0] + max_idx
        return (x[r_hm_idx] - x[l_hm_idx]) / 2
    except IndexError:
        return 1.0


def guess_pseudo_voigt_amp_parameter(y: NDArray, max_idx: int, sig: float, frac: float) -> float:
    """Guess amplitude parameter for pseudo-voigt model at max_idx.

    Parameters
    ----------
    y : NDArray
        Y values.
    max_idx : int
        Index of the peak maximum.
    sig : float
        Sigma of the peak.
    frac : float
        Fraction of the peak.

    Returns
    -------
    float
        Amplitude parameter for pseudo-voigt model at max_idx.
    """
    shape_mult = frac / np.pi + (1 - frac) * np.sqrt(np.log(2) / np.pi)
    amp = (y[max_idx] - y.min()) * sig / shape_mult
    return amp


def calculate_background_intensities(
    x: NDArray,
    y: NDArray,
    start: int | float,
    stop: int | float,
    mode: Literal["value", "index"] = "index",
    avg_on: int = 3,
) -> dict[str, float]:
    """Calculate intensities at start and stop indices.

    Useful for region with background.
    This method can be used to update background parameters for a region according to the new slice.

    Parameters
    ----------
    x : NDArray
        X values.
    y : NDArray
        Y values.
    start : int | float
        Start value or index.
    stop : int | float
        Stop value or index.
    mode: Literal["value", "index"]
        Mode to convert start and stop to indices.
    avg_on : int, default=3
        Number of points to average on the start and stop indices.

    Returns
    -------
    dict[str, float]
        Parameters i1 and i2 for background model.
    """

    if mode == "value":
        start = np.searchsorted(x, start)
        stop = np.searchsorted(x, stop)

    i1 = np.mean(y[max(start - avg_on, 0) : start])
    i2 = np.mean(y[stop : min(stop + avg_on, len(y))])
    return dict(i1=i1, i2=i2)


def guess_pseudo_voigt_params_at_max(
    x: NDArray,
    y: NDArray,
    max_idx: int,
    frac: float = 0.5,
) -> tuple[float, float, float, float]:
    """Guess peak parameters for pseudo-voigt model.

    Parameters
    ----------
    x : NDArray
        X values.
    y : NDArray
        Y values.
    max_idx : int
        Index of the peak maximum.
    frac : float, default=0.5
        Fraction of the peak.

    Returns
    -------
    amp : float
        Amplitude of the peak.
    cen : float
        Center of the peak.
    sig : float
        Sigma of the peak.
    frac : float
        Fraction of the peak.
    """
    sig = guess_pseudo_voigt_sig_parameter(x, y, max_idx)
    amp = guess_pseudo_voigt_amp_parameter(y, max_idx, sig, frac)
    return amp, x[max_idx], sig, frac


def guess_peak_position_by_residuals(x: NDArray, residuals: NDArray) -> int:
    """Guess peak position by residuals.
    The peak position is the index of the maximum residual.

    Parameters
    ----------
    x : NDArray
        X values.
    residuals : NDArray
        Residuals.

    Returns
    -------
    int
        Index of the peak maximum.
    """
    return np.argmax(residuals)


def create_pseudo_voigt_peak_parameters(
    region: RegionLike, components: tuple[ComponentLike, ...]
) -> dict[str, float]:
    """Create pseudo-voigt peak parameters for a region.
    This method can be used to create initial peak parameters for pseudo-voigt model.

    Parameters
    ----------
    region: RegionLike
        Region.
    components: tuple[ComponentLike, ...]
        Components.

    Returns
    -------
    dict[str, float]
        Peak parameters.
    """
    region_eval = region_bundle(region, components)
    max_idx = guess_peak_position_by_residuals(region_eval.x, region_eval.residuals)
    amp, cen, sig, frac = guess_pseudo_voigt_params_at_max(region_eval.x, region_eval.y, max_idx)
    return {"amp": amp, "cen": cen, "sig": sig, "frac": frac}
