"""
Set of useful methods for routine tasks.

Provides functions for guessing parameters.
"""

import numpy as np
from numpy.typing import NDArray


def guess_pseudo_voigt_sig_paramater(x: NDArray, y: NDArray, max_idx: int) -> float:
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
    l_hm_idx = np.where(y[:max_idx] <= half_max)[0][-1]
    r_hm_idx = np.where(y[max_idx:] <= half_max)[0][0] + max_idx
    return (x[r_hm_idx] - x[l_hm_idx]) / 2


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
    y: NDArray,
    start: int,
    stop: int,
    avg_on: int = 3,
) -> tuple[float, float]:
    """Calculate intensities at start and stop indices.

    Parameters
    ----------
    y : NDArray
        Y values.
    start : int
        Start index.
    stop : int
        Stop index.
    avg_on : int, default=3
        Number of points to average on the start and stop indices.

    Returns
    -------
    tuple[float, float]
        Intensities at start and stop indices.
    """

    i1 = np.mean(y[max(start - avg_on, 0) : start])
    i2 = np.mean(y[stop : min(stop + avg_on, len(y))])
    return i1, i2


def guess_pseudo_voigt_peak_parameters_at_max_idx(
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
    sig = guess_pseudo_voigt_sig_paramater(x, y, max_idx)
    amp = guess_pseudo_voigt_amp_parameter(y, max_idx, sig, frac)
    return amp, x[max_idx], sig, frac


def guess_peak_position_at_max_idx(x: NDArray, diff: NDArray) -> float:
    """Guess peak position by difference of Y values and model values.
    The peak position is the index of the maximum difference.

    Parameters
    ----------
    x : NDArray
        X values.
    diff : NDArray
        Difference of Y values and model values.

    Returns
    -------
    float
        Peak position.
    """
    return x[np.argmax(diff)]
