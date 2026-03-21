import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


def interpolate(x: NDArray, y: NDArray, num: int = 256) -> tuple[NDArray, NDArray]:
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
    tuple[NDArray, NDArray]
        new_x : NDArray
            Interpolated x-values.
        new_y : NDArray
            Interpolated y-values.
    """
    f = interp1d(x, y, kind="linear", fill_value="extrapolate")
    new_x = np.linspace(x[0], x[-1], num, dtype=np.float32)
    new_y = f(new_x)
    return new_x, new_y


def recalculate_idx(idx: int, array_1: NDArray, array_2: NDArray) -> int:
    """Map index from interpolated grid (array_1) to original grid (array_2).

    Parameters
    ----------
    idx : int
        Index in the interpolated grid.
    array_1 : NDArray
        Interpolated grid.
    array_2 : NDArray
        Original grid.

    Returns
    -------
    int
        Index in the original grid.
    """
    if idx >= len(array_1):
        return len(array_2)
    val = array_1[idx]
    return int(np.abs(array_2 - val).argmin())


def find_closest_index(value: float, array: NDArray) -> int:
    """Find the closest index to a value in an array.
    Works for sorted arrays.

    Parameters
    ----------
    value : float
        Value to find the closest index to.
    array : NDArray
        Array to find the closest index in.

    Returns
    -------
    int
        Index of the closest value in the array.
    """
    return int(np.abs(array - value).argmin())
