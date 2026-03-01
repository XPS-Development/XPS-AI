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
    f = interp1d(x, y, kind="linear")
    new_x = np.linspace(x[0], x[-1], num, dtype=np.float32)
    new_y = f(new_x)
    return new_x, new_y
