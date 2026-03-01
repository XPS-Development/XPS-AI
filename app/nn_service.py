"""
App-layer NN service: load pipelines, run inference, produce Change objects.

Holds a SegmenterPipeline instance, loads ONNX models, and converts
pipeline outputs to CreateRegion, CreateBackground, CreatePeak changes
via the SegmenterPipelineAdapter.
"""

from pathlib import Path

from core.types import SpectrumLike
from tools.nn import SegmenterPipeline

from .command.changes import CompositeChange
from .nn_adapter import segmenter_results_to_changes


class NNService:
    """
    App-level service for NN model pipelines.

    Loads ONNX models into pipelines (e.g. SegmenterPipeline), accepts
    spectrum DTOs, runs inference, and returns Change objects for
    CommandExecutor.
    """

    def __init__(
        self,
        model_path: str | None = None,
        pred_threshold: float = 0.5,
        smooth: bool = True,
        interp_num: int = 256,
    ) -> None:
        """
        Initialize the NN service with a segmenter pipeline.

        Parameters
        ----------
        model_path : str or None, optional
            Path to segmenter ONNX model; if None, call load_model before run.
        pred_threshold : float, optional
            Mask probability threshold (default 0.5).
        smooth : bool, optional
            Whether to smooth the peak mask (default True).
        interp_num : int, optional
            Interpolation points (default 256).
        """
        self._pipeline = SegmenterPipeline(
            model_path=model_path,
            pred_threshold=pred_threshold,
            smooth=smooth,
            interp_num=interp_num,
        )

    def load_model(self, model_path: str | Path) -> None:
        """
        Load or reload the ONNX model into the segmenter pipeline.

        Parameters
        ----------
        model_path : str or Path
            Path to the segmenter ONNX model file.
        """
        self._pipeline.load_model(model_path)

    def run_segmenter(
        self,
        spectrum_id: str,
        normalized_spectrum: SpectrumLike,
        original_spectrum: SpectrumLike,
    ) -> CompositeChange:
        """
        Run the segmenter pipeline and return Change objects.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the parent spectrum for CreateRegion.
        normalized_spectrum : SpectrumLike
            Spectrum with normalized y (e.g. from DTOService.get_spectrum(..., normalized=True)).
        original_spectrum : SpectrumLike
            Original spectrum with raw x/y (e.g. from DTOService.get_spectrum(..., normalized=False)).

        Returns
        -------
        CompositeChange
            CreateRegion, CreateBackground, CreatePeak changes for CommandExecutor.
        """
        results = self._pipeline.run(normalized_spectrum, original_spectrum)
        return segmenter_results_to_changes(spectrum_id, results)
