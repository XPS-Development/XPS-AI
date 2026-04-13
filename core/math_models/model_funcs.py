import numpy as np
from scipy.integrate import trapezoid

from typing import Sequence, List
from numpy.typing import NDArray


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
