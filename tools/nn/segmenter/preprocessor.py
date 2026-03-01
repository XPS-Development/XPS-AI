"""
Preprocessor layer: SpectrumLike -> model-specific input.

Preprocessors produce model input (e.g. dict of arrays for ONNX). For the
segmenter, input y is expected to be already normalized (caller responsibility).
"""

import numpy as np
from numpy.typing import NDArray

from core.types import SpectrumLike

from ..._tools import interpolate
from ..types import ModelInputT
from .adapter import ONNXSegmenterAdapter


class SegmenterPreprocessor:
    """
    Preprocessor for the segmenter model: interpolate, log/min-max normalize, stack.

    Consumes SpectrumLike (x, y); y should be normalized. Optionally accepts
    precomputed x_int, y_int to avoid duplicate interpolation when used in a pipeline.
    """

    def __init__(self, num: int = 256) -> None:
        """
        Parameters
        ----------
        num : int, optional
            Number of points for interpolation (default 256).
        """
        self._num = num

    def __call__(
        self,
        data: SpectrumLike,
    ) -> ModelInputT:
        """
        Produce segmenter model input from spectrum-like data.

        Parameters
        ----------
        data : SpectrumLike
            Spectrum with .x and .y (y expected normalized for segmenter).

        Returns
        -------
        ModelInputT
            Dict with key INPUT_KEY, value shape (1, 2, num).
        metadata: dict[str, NDArray]
            Dictionary with metadata.
        """
        x_int, y_int = interpolate(data.x, data.y, num=self._num)

        tensor = self._prepare_input(y_int)
        return {ONNXSegmenterAdapter.INPUT_KEY: tensor}, {"x": data.x, "x_int": x_int}

    def _prepare_input(self, y: NDArray) -> NDArray:
        """Log-normalize and min-max scale, stack into (1, 2, n) for ONNX."""
        y_log = np.log(10.0 * y + 1.0)
        y_log_min, y_log_max = y_log.min(), y_log.max()
        if y_log_max > y_log_min:
            y_log = (y_log - y_log_min) / (y_log_max - y_log_min)
        x_inp = np.stack((y.astype(np.float32), y_log.astype(np.float32)), axis=0)
        return x_inp[np.newaxis, :, :]
