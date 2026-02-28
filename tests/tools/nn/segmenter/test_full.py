"""
Integration test: run real ONNX segmenter on example spectrum.

Place model.onnx at project root for this test to run; otherwise it is skipped.
"""

from pathlib import Path

import pytest

from tools.nn.pipeline import SegmenterPipeline
from tools.nn.segmenter import SegmenterResult
from tools.nn.types import BackgroundDetectionResult, PeakDetectionResult


# model.onnx at project root (skip if missing)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "model.onnx"


def test_segmenter_pipeline_real_model(
    dto_service,
    spectrum_id: str,
) -> None:
    """
    Run SegmenterPipeline with real ONNX model on spectrum from dto_service.

    Skips when model.onnx is not found at project root.
    """
    if not MODEL_PATH.exists():
        pytest.skip("model.onnx not found at project root")

    seg = SegmenterPipeline(model_path=str(MODEL_PATH))
    norm_spec = dto_service.get_spectrum(spectrum_id, normalized=True)
    orig_spec = dto_service.get_spectrum(spectrum_id, normalized=False)

    results = seg.run(norm_spec, orig_spec)

    assert isinstance(results, list)
    for sr in results:
        assert isinstance(sr, SegmenterResult)
        assert hasattr(sr, "region") and hasattr(sr, "peaks")
        assert sr.region.start < sr.region.stop
        assert isinstance(sr.peaks, tuple)
        assert sr.background is not None
        assert isinstance(sr.background, BackgroundDetectionResult)
        for peak in sr.peaks:
            assert isinstance(peak, PeakDetectionResult)
