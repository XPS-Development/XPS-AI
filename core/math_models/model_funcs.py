"""Closed-form peak and background evaluation functions."""

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid


def gauss(x: NDArray, center: float, sigma: float) -> NDArray:
    """Return a normalized Gaussian evaluated at ``x``.

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
    """Return a normalized Lorentzian evaluated at ``x``.

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
    return amplitude * (
        (1 - fraction) * gauss(x, center, sigma_g) + fraction * lorentz(x, center, sigma)
    )


_ASYM_EPS = 1e-8


def asym_pvoigt(
    x: NDArray,
    amplitude: float,
    center: float,
    sigma: float,
    fraction: float,
    asymmetry: float,
) -> NDArray:
    """
    Pseudo-Voigt with a log-warped axis (Stancik-Brauns).

    Positive ``asymmetry`` broadens the high-x side and narrows the low-x side.
    When ``x`` is binding energy, that is the high-binding-energy side. The
    Jacobian of the warp keeps the integral equal to ``amplitude`` while the
    log argument stays positive. Past that pole the profile is zero. At
    ``asymmetry == 0`` the result is :func:`pvoigt`.

    Parameters
    ----------
    x : NDArray
        Array of x-values.
    amplitude : float
        Peak area.
    center : float
        Peak center.
    sigma : float
        Width parameter, as in :func:`pvoigt`.
    fraction : float
        Lorentzian fraction.
    asymmetry : float
        Warp strength. Zero is symmetric.

    Returns
    -------
    NDArray
        Asymmetric pseudo-Voigt evaluated at ``x``.
    """
    if abs(asymmetry) < _ASYM_EPS:
        return pvoigt(x, amplitude, center, sigma, fraction)
    # Positive asymmetry stretches x > center.
    # The log warp is defined only while z > 0; past that pole the profile is 0.
    warp = asymmetry
    dx = x - center
    z = 1.0 + 2.0 * warp * dx / sigma
    valid = z > 1e-4
    z_safe = np.where(valid, z, 1.0)
    u = center + (sigma / (2.0 * warp)) * np.log(z_safe)
    shaped = pvoigt(u, amplitude, center, sigma, fraction) / z_safe
    return np.where(valid, shaped, 0.0)


def tail_pvoigt(
    x: NDArray,
    amplitude: float,
    center: float,
    sigma: float,
    fraction: float,
    tail_scale: float,
    tail_length: float,
) -> NDArray:
    """
    Pseudo-Voigt plus an exponential tail on the high-x side.

    The symmetric core is :func:`pvoigt`. The tail lifts the high-x wing toward
    the peak height and leaves ``x <= center`` unchanged. ``tail_scale == 0``
    returns the symmetric core. The tail adds area beyond ``amplitude``.

    Parameters
    ----------
    x : NDArray
        Array of x-values.
    amplitude : float
        Area of the symmetric core.
    center : float
        Peak center.
    sigma : float
        Width parameter, as in :func:`pvoigt`.
    fraction : float
        Lorentzian fraction.
    tail_scale : float
        Tail strength from 0 (none) to 1 (wing lifted fully toward the peak height).
    tail_length : float
        Exponential decay length of the tail, in x units.

    Returns
    -------
    NDArray
        Tailed pseudo-Voigt evaluated at ``x``.
    """
    core = pvoigt(x, amplitude, center, sigma, fraction)
    if tail_scale == 0.0 or tail_length <= 0.0:
        return core
    height = float(pvoigt(np.asarray([center]), amplitude, center, sigma, fraction)[0])
    if height == 0.0:
        return core
    unit = core / height
    dx = x - center
    decay = np.exp(-np.maximum(dx, 0.0) / tail_length)
    extra = np.where(dx > 0.0, (1.0 - unit) * tail_scale * decay, 0.0)
    return core + extra * height


def static_shirley_background(
    x: NDArray, y: NDArray, i1: float, i2: float, iters: int = 8
) -> NDArray:
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

        def shirley_to_i(i: int, k_val: float = k, y_val: NDArray = y_adj) -> float:
            return k_val * trapezoid(y_val[: i + 1], x[: i + 1])

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
