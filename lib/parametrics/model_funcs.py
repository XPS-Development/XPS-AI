from typing import Sequence, List

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid


def gauss(x: NDArray, center: float, sigma: float) -> NDArray:
    """
    Normalized Gaussian function.

    Parameters
    ----------
    x : NDArray
        Array of x-values.
    center : float
        Center of the Gaussian peak.
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    NDArray
        Gaussian function evaluated at x.
    """
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def lorentz(x: NDArray, center: float, sigma: float) -> NDArray:
    """
    Normalized Lorentzian function.

    Parameters
    ----------
    x : NDArray
        Array of x-values.
    center : float
        Center of the Lorentzian peak.
    sigma : float
        Half-width at half-maximum of the Lorentzian.

    Returns
    -------
    NDArray
        Lorentzian function evaluated at x.
    """
    return 1 / np.pi * sigma / ((x - center) ** 2 + sigma**2)


def pvoigt(x: NDArray, amplitude: float, center: float, sigma: float, fraction: float) -> NDArray:
    """
    Pseudo-Voigt function: linear combination of normalized Gaussian and Lorentzian.

    Parameters
    ----------
    x : NDArray
        Array of x-values.
    amplitude : float
        Peak amplitude.
    center : float
        Peak center position.
    sigma : float
        Width parameter (Gaussian and Lorentzian combined).
    fraction : float
        Lorentzian fraction (0 = pure Gaussian, 1 = pure Lorentzian).

    Returns
    -------
    pvoigt : NDArray
        Pseudo-Voigt function evaluated at x.
    """
    sigma_g = sigma / np.sqrt(2 * np.log(2))  # convert to Gaussian sigma for same FWHM
    return amplitude * ((1 - fraction) * gauss(x, center, sigma_g) + fraction * lorentz(x, center, sigma))


def npvoigt(params: dict[str, float], x: NDArray, combine: Sequence[int]) -> NDArray:
    """
    Sum multiple pseudo-Voigt peaks on the same x-array.

    Parameters
    ----------
    params : dict[str, float]
        Dictionary containing parameters of all peaks, keys like '0_amp', '0_cen', etc.
    x : NDArray
        Array of x-values.
    combine : Sequence[str]
        Indexes of peaks to sum.

    Returns
    -------
    NDArray
        Combined peak function evaluated at x.
    """
    y = np.zeros_like(x)
    for i in combine:
        y += pvoigt(
            x,
            params[f"{i}_amp"],
            params[f"{i}_cen"],
            params[f"{i}_sig"],
            params[f"{i}_frac"],
        )
    return y


def ndpvoigt(
    params: dict[str, float],
    x: Sequence[NDArray],
    combinations: Sequence[Sequence[int]],
) -> List[NDArray]:
    """
    Calculate multiple peaks datasets simultaneously.

    Parameters
    ----------
    params : dict[str, float]
        Dictionary containing parameters of all peaks.
    x : Sequence[NDArray]
        Sequence of x-arrays for each dataset.
    combinations : Sequence[Sequence[int]]
        Sequence of sequences of peak indices to apply to each dataset.

    Returns
    -------
    List[NDArray]
        List of evaluated pseudo-Voigt datasets.
    """
    y_list: List[NDArray] = []
    for row, combination in zip(x, combinations):
        y_list.append(npvoigt(params, row, combination))
    return y_list


def static_shirley_background(x: NDArray, y: NDArray, i1: float, i2: float, iters: int = 8) -> NDArray:
    """
    Calculate iterative Shirley background.

    Parameters
    ----------
    x : NDArray
        X-data points.
    y : NDArray
        Y-data points corresponding to x.
    i1 : float
        Starting intensity (baseline).
    i2 : float
        Ending intensity (baseline).
    iters : int, default=8
        Number of iterations.

    Returns
    -------
    NDArray
        Shirley background evaluated at x.
    """
    background = np.zeros_like(x, dtype=np.float32)
    for _ in range(iters):
        y_adj = y - i1 - background
        k = (i2 - i1) / trapezoid(y_adj, x)
        shirley_to_i = lambda i: k * trapezoid(y_adj[: i + 1], x[: i + 1])
        background = np.array([shirley_to_i(i) for i in range(len(x))])
    return background + i1


def linear_background(x: NDArray, i1: float, i2: float) -> NDArray:
    """
    Calculate a linear background between two intensity points.

    Parameters
    ----------
    x : NDArray
        Array of x-values.
    i1 : float
        Intensity at the start of x (x[0]).
    i2 : float
        Intensity at the end of x (x[-1]).

    Returns
    -------
    NDArray
        Linear background evaluated at each point in x.
    """
    return i1 + (i2 - i1) * (x - x[0]) / (x[-1] - x[0])
