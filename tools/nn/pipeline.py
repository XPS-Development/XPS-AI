"""
Pipeline orchestration: preprocess -> adapter.run -> postprocess.

Composes preprocessor, adapter, and postprocessor. Stays free of app concepts
(no spectrum_id); the app layer maps RegionBounds to CreateRegion / CreatePeak.
"""

from typing import Any, Protocol
from pathlib import Path
from numpy.typing import NDArray

from core.types import SpectrumLike

from .segmenter import (
    ONNXSegmenterAdapter,
    SegmenterPostprocessor,
    SegmenterPreprocessor,
    SegmenterResult,
)
from .types import ModelInputT, ModelOutputT


class PreprocessorLike(Protocol):
    """Protocol for preprocessor callables used by the pipeline."""

    def __call__(
        self,
        data: SpectrumLike,
        *,
        x_int: NDArray | None = None,
        y_int: NDArray | None = None,
    ) -> ModelInputT: ...


class AdapterLike(Protocol):
    """Protocol for adapters used by the pipeline."""

    def run(self, model_input: ModelInputT) -> ModelOutputT: ...


class PostprocessorLike(Protocol):
    """Protocol for postprocessor callables that accept x and x_int."""

    def __call__(
        self,
        model_output: ModelOutputT,
        *,
        x: NDArray,
        x_int: NDArray,
    ) -> Any: ...


class InferencePipeline:
    """
    Generic pipeline: preprocess -> adapter.run -> postprocess.

    Accepts optional preprocessor, adapter, and postprocessor instances;
    returns the postprocessor result type (e.g. list[RegionBounds]).
    """

    def run(self, spectrum: SpectrumLike) -> Any:
        """
        Run the full pipeline on a spectrum-like input.
        """
        raise NotImplementedError("Subclasses must implement run method")


class SegmenterPipeline(InferencePipeline):
    """
    Concrete segmenter pipeline: SegmenterPreprocessor + ONNXSegmenterAdapter + SegmenterPostprocessor.

    Exposes run(spectrum) returning list[RegionBounds]. Model path and
    threshold/smooth options are configured at construction.
    """

    def __init__(
        self,
        model_path: str | None = None,
        pred_threshold: float = 0.5,
        smooth: bool = True,
        interp_num: int = 256,
    ) -> None:
        """
        Parameters
        ----------
        model_path : str or None, optional
            Path to segmenter ONNX model; if None, adapter must be loaded later.
        pred_threshold : float, optional
            Mask probability threshold (default 0.5).
        smooth : bool, optional
            Whether to smooth the peak mask (default True).
        interp_num : int, optional
            Interpolation points (default 256).
        """
        self.preprocessor = SegmenterPreprocessor(num=interp_num)
        self.adapter = ONNXSegmenterAdapter(model_path=model_path)
        self.postprocessor = SegmenterPostprocessor(threshold=pred_threshold, smooth=smooth)

    def run(
        self, normalized_spectrum: SpectrumLike, original_spectrum: SpectrumLike
    ) -> list[SegmenterResult]:
        """
        Run the full pipeline on a spectrum-like input.

        Parameters
        ----------
        normalized_spectrum : SpectrumLike
            Input with .x and .y (for segmenter, y should be normalized).
        original_spectrum : SpectrumLike
            Original spectrum with .x and .y.

        Returns
        -------
        list[SegmenterResult]
            List of SegmenterResult objects.
        """
        model_input, metadata = self.preprocessor(normalized_spectrum)
        model_output = self.adapter.run(model_input)
        return self.postprocessor(model_output, **metadata, y=original_spectrum.y)

    def load_model(self, model_path: str | Path) -> None:
        """Load or reload the ONNX model from the given path."""
        self.adapter.load(model_path)
