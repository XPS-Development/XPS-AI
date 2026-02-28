"""
Adapter layer: run inference on model input.

Adapters load the model and run inference; input/output are model-specific
(tensors / dicts). Framework details (ONNX, etc.) stay in this layer.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort

from ..types import ModelInputT, ModelOutputT


class ONNXSegmenterAdapter:
    """
    ONNX adapter for the segmenter: runs inference and returns region/max masks.

    Holds an InferenceSession; no knowledge of spectra or regions.
    """

    INPUT_SHAPE: tuple[int, int, int] = (1, 2, 256)
    INPUT_KEY: str = "l_x_"
    CHANNEL_MASK_KEYS: tuple[str, str] = ("region_mask", "max_mask")

    def __init__(self, model_path: str | Path | None = None) -> None:
        """
        Parameters
        ----------
        model_path : str or Path or None, optional
            Path to ONNX model file. If None or empty, session is not created
            until load() or run() with a loaded session.
        """
        self._model_path: Path | None = Path(model_path) if model_path and str(model_path).strip() else None
        self._session = None
        if self._model_path and self._model_path.exists():
            self._session = ort.InferenceSession(str(self._model_path), providers=["CPUExecutionProvider"])

    @property
    def session(self) -> ort.InferenceSession | None:
        """ONNX InferenceSession; may be None if no model loaded."""
        return self._session

    def load(self, model_path: str | Path) -> None:
        """Load or reload the ONNX model from the given path."""
        self._model_path = Path(model_path)
        self._session = ort.InferenceSession(str(self._model_path), providers=["CPUExecutionProvider"])

    def run(self, model_input: ModelInputT) -> ModelOutputT:
        """
        Run segmenter inference; return region_mask and max_mask arrays.

        Parameters
        ----------
        model_input : ModelInputT
            Dict with INPUT_KEY and value shape (1, 2, n).

        Returns
        -------
        ModelOutputT
            Dict with CHANNEL_MASK_KEYS as keys and 1d arrays as values.
        """
        if self._session is None:
            raise RuntimeError("ONNXSegmenterAdapter: no model loaded")
        inp = model_input.get(self.INPUT_KEY)
        if inp is None:
            raise KeyError(f"Model input must contain key {self.INPUT_KEY!r}")
        out: list[NDArray] = self._session.run(None, {self.INPUT_KEY: inp})
        arr = out[0]
        return {key: np.asarray(arr[0, i, :]) for i, key in enumerate(self.CHANNEL_MASK_KEYS)}
