from typing import Iterable, Generator, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


def batch(iterable: Iterable, n: int = 1) -> Generator:
    """
    Yield successive n-sized chunks from an iterable.

    Parameters
    ----------
    iterable : Iterable
        The iterable to split into batches.
    n : int, default=1
        Size of each batch.

    Yields
    ------
    Generator
        Subsequent chunks of the iterable.
    """
    for idx in range(0, len(iterable), n):
        yield iterable[idx : idx + n]


def norm_with_coefs(value: float | NDArray, norm_coefs: Tuple[float, float]) -> float | NDArray:
    """
    Normalize a value using the provided coefficients.

    Parameters
    ----------
    value : float | NDArray
        The value to normalize.
    norm_coefs : Tuple[float, float]
        The normalization coefficients. (min, max)

    Returns
    -------
    float | NDArray
        The normalized value.
    """
    return (value - norm_coefs[0]) / (norm_coefs[1] - norm_coefs[0])


def denorm_with_coefs(value: float, norm_coefs: Tuple[float, float]) -> float:
    """
    Denormalize a value using the provided coefficients.

    Parameters
    ----------
    value : float | NDArray
        The value to denormalize.
    norm_coefs : Tuple[float, float]
        The normalization coefficients. (min, max)

    Returns
    -------
    float | NDArray
        The denormalized value.
    """
    return value * (norm_coefs[1] - norm_coefs[0]) + norm_coefs[0]


def interpolate(x: NDArray, y: NDArray, num: int = 256) -> Tuple[NDArray, NDArray]:
    """
    Linearly interpolate y-values over a new x-grid.

    Parameters
    ----------
    x : NDArray
        Original x-values.
    y : NDArray
        Original y-values.
    num : int, default=256
        Number of points in the interpolated grid.

    Returns
    -------
    Tuple[NDArray, NDArray]
        new_x : NDArray
            Interpolated x-values.
        new_y : NDArray
            Interpolated y-values.
    """
    f = interp1d(x, y, kind="linear")
    new_x = np.linspace(x[0], x[-1], num, dtype=np.float32)
    new_y = f(new_x)
    return new_x, new_y
