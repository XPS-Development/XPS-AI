"""
Tests for tools.nn.pipeline: SegmenterPipeline end-to-end.
"""

import numpy as np
import pytest

from tools.nn.pipeline import SegmenterPipeline
from tools.nn.segmenter import ONNXSegmenterAdapter, SegmenterResult


class _SpectrumLike:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y


@pytest.fixture
def fixture_spectrum() -> _SpectrumLike:
    """Spectrum-like fixture: Gaussian-like bump, positive y."""
    x = np.linspace(0.0, 10.0, 200, dtype=np.float64)
    y = np.exp(-((x - 5.0) ** 2) / 2.0) + 0.1
    return _SpectrumLike(x, y)


class _MockAdapter:
    """Mock adapter that returns fixed masks for pipeline e2e without ONNX."""

    def run(self, model_input: dict) -> dict:
        n = model_input[ONNXSegmenterAdapter.INPUT_KEY].shape[2]
        region_mask = np.zeros(n)
        region_mask[50:200] = 1.0
        max_mask = np.zeros(n)
        max_mask[120:125] = 1.0
        return {"region_mask": region_mask, "max_mask": max_mask}


def test_segmenter_pipeline_run_without_model_raises(fixture_spectrum: _SpectrumLike) -> None:
    """SegmenterPipeline with no model path raises when run (adapter has no session)."""
    pipeline = SegmenterPipeline(model_path=None)
    with pytest.raises(RuntimeError, match="no model loaded"):
        pipeline.run(fixture_spectrum, fixture_spectrum)


def test_segmenter_pipeline_run_with_mock_adapter(fixture_spectrum: _SpectrumLike) -> None:
    """SegmenterPipeline with patched adapter returns list[SegmenterResult]."""
    pipeline = SegmenterPipeline(model_path=None)
    pipeline.adapter = _MockAdapter()

    result = pipeline.run(fixture_spectrum, fixture_spectrum)

    assert isinstance(result, list)
    assert len(result) >= 1
    for sr in result:
        assert isinstance(sr, SegmenterResult)
        assert hasattr(sr, "region") and hasattr(sr, "peaks")
        assert sr.region.start < sr.region.stop
        assert sr.region.start >= 0 and sr.region.stop <= len(fixture_spectrum.x)
        assert isinstance(sr.peaks, tuple)
