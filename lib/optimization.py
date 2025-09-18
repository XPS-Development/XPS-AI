from itertools import chain
from typing import Sequence, Tuple, Optional, Callable

import numpy as np
from numpy.typing import NDArray
from lmfit import Parameters, minimize

from .tools import batch
from .funcs import ndpvoigt


class Optimizer:
    """Optimizer for multi-spectrum peak fitting.

    Parameters
    ----------
    x : NDArray | Sequence[NDArray]
        X-data. Either a single 1D array or a sequence of 1D arrays.
    y : NDArray | Sequence[NDArray]
        Y-data, aligned with x.
    peaks_parameters : Parameters
        lmfit Parameters object, number of parameters must be divisible by 4.
    combinations : tuple[tuple[int, ...]], optional
        Defines which peaks belong to each spectrum.
        If None, parameters are split uniformly.
    model : callable, default=ndpvoigt
        Model function: (params, x, combinations) -> list[NDArray]
    """

    def __init__(
        self,
        x: NDArray | Sequence[NDArray],
        y: NDArray | Sequence[NDArray],
        peaks_parameters: Parameters,
        combinations: Optional[Tuple[Tuple[str, ...], ...]] = None,
        model: Callable = None,
    ) -> None:

        self._validate_peaks(peaks_parameters)
        self.x, self.y = self._process_xy(x, y)
        self.combinations = self._process_combinations(combinations, len(self.x), len(peaks_parameters) // 4)

        self.init_params = peaks_parameters
        self.model = model if model is not None else ndpvoigt

    def _validate_peaks(self, params: Parameters) -> None:
        if len(params) % 4 != 0:
            raise ValueError(f"Expected number of parameters divisible by 4, got {len(params)}")

    def _process_xy(
        self,
        x: NDArray | Sequence[NDArray],
        y: NDArray | Sequence[NDArray],
    ) -> Tuple[Tuple[NDArray, ...], Tuple[NDArray, ...]]:

        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if len(x) != len(y):
                raise ValueError("x and y must have the same length.")
            return (x,), (y,)
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        return tuple(x), tuple(y)

    def _process_combinations(
        self,
        combinations: Optional[Tuple[Tuple[int, ...], ...]],
        n_spectra: int,
        n_peaks: int,
    ) -> Tuple[Tuple[int, ...], ...]:

        if combinations is None:  # split peaks uniformly
            peaks_per_spectrum = n_peaks // n_spectra
            return tuple(tuple(i) for i in batch(range(n_peaks), peaks_per_spectrum))
        if len(combinations) != n_spectra:
            raise ValueError("Number of combinations must match number of spectra.")
        if len(tuple(chain(*combinations))) != n_peaks:
            raise ValueError("Total peaks in combinations must match number of peaks.")
        return combinations

    # def _check_normalization(self, y: Sequence[NDArray], is_norm: bool) -> None:
    #     if is_norm and any(np.max(arr) > 1 for arr in y):
    #         raise ValueError("is_norm=True, but y contains values greater than 1.")

    def ndresid(
        self,
        params: Parameters,
        x: Sequence[NDArray],
        y: Sequence[NDArray],
        combinations: Tuple[Tuple[int, ...], ...],
    ) -> NDArray:
        """Calculate residuals for multiple spectra simultaneously."""
        y_true = np.concatenate(y).astype(float)
        y_model = np.concatenate(self.model(params, x, combinations)).astype(float)
        return y_model - y_true

    def fit(self, return_result: bool = False, **kwargs) -> Parameters | MinimizerResult:
        """Run optimization.

        Parameters
        ----------
        return_result : bool, default=False
            If True, returns full lmfit.MinimizerResult, else only params.
        kwargs : dict
            Extra arguments passed to lmfit.minimize.

        Returns
        -------
        Parameters or MinimizerResult
        """
        res = minimize(
            self.ndresid,
            self.init_params,
            args=(self.x, self.y, self.combinations),
            **kwargs,
        )
        return res if return_result else res.params

    def __repr__(self) -> str:
        return (
            f"Optimizer(n_spectra={len(self.x)}, "
            f"n_peaks={len(self.init_params)//4}, "
            f"norm={self.is_norm}, "
            f"model={self.model.__name__ if self.model else None})"
        )
