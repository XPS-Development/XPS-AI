"""
Tests for tools.nn.segmenter: SegmenterPostprocessor with fixed masks -> list[SegmenterResult].
"""

import numpy as np
import pytest

from tools.nn.segmenter import ONNXSegmenterAdapter, SegmenterPostprocessor, SegmenterResult
from tools.nn.types import PeakDetectionResult, RegionDetectionResult


@pytest.fixture
def x_original() -> np.ndarray:
    return np.linspace(0.0, 10.0, 101)


@pytest.fixture
def x_interp() -> np.ndarray:
    return np.linspace(0.0, 10.0, 256)


@pytest.fixture
def y_original(x_original: np.ndarray) -> np.ndarray:
    """Original spectrum y for peak parameter computation."""
    return np.exp(-((x_original - 5.0) ** 2) / 2.0) + 0.1


def test_postprocessor_one_region(
    x_original: np.ndarray,
    x_interp: np.ndarray,
    y_original: np.ndarray,
) -> None:
    """Single peak/max run produces one SegmenterResult with region and peaks."""
    n = len(x_interp)
    region_mask = np.zeros(n)
    region_mask[50:150] = 1.0
    max_mask = np.zeros(n)
    max_mask[100:101] = 1.0  # single max at 100
    model_output = {
        ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[0]: region_mask,
        ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[1]: max_mask,
    }
    post = SegmenterPostprocessor(threshold=0.5, smooth=False)
    result = post(model_output, x=x_original, x_int=x_interp, y=y_original)
    assert len(result) == 1
    sr = result[0]
    assert isinstance(sr, SegmenterResult)
    assert isinstance(sr.region, RegionDetectionResult)
    assert sr.region.start < sr.region.stop
    assert len(sr.peaks) >= 1
    for peak in sr.peaks:
        assert isinstance(peak, PeakDetectionResult)


def test_postprocessor_no_region_when_empty_max(
    x_original: np.ndarray,
    x_interp: np.ndarray,
    y_original: np.ndarray,
) -> None:
    """When max_mask has no run inside peak borders, region is skipped."""
    n = len(x_interp)
    region_mask = np.zeros(n)
    region_mask[50:150] = 1.0
    max_mask = np.zeros(n)  # no max
    model_output = {
        ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[0]: region_mask,
        ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[1]: max_mask,
    }
    post = SegmenterPostprocessor(threshold=0.5, smooth=False)
    result = post(model_output, x=x_original, x_int=x_interp, y=y_original)
    assert len(result) == 0


def test_postprocessor_missing_keys_raises(
    x_original: np.ndarray,
    x_interp: np.ndarray,
    y_original: np.ndarray,
) -> None:
    """Missing region_mask or max_mask raises KeyError."""
    post = SegmenterPostprocessor(threshold=0.5, smooth=False)
    with pytest.raises(KeyError):
        post({}, x=x_original, x_int=x_interp, y=y_original)
    with pytest.raises(KeyError):
        post(
            {ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[0]: np.zeros(256)},
            x=x_original,
            x_int=x_interp,
            y=y_original,
        )


def test_postprocessor_smooth_vs_no_smooth(
    x_original: np.ndarray,
    x_interp: np.ndarray,
    y_original: np.ndarray,
) -> None:
    """With smooth=True and same masks, result is consistent."""
    n = len(x_interp)
    region_mask = np.zeros(n)
    region_mask[50:150] = 1.0
    max_mask = np.zeros(n)
    max_mask[99:102] = 1.0
    model_output = {
        ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[0]: region_mask,
        ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[1]: max_mask,
    }
    post_no = SegmenterPostprocessor(threshold=0.5, smooth=False)
    post_yes = SegmenterPostprocessor(threshold=0.5, smooth=True)
    result_no = post_no(model_output, x=x_original, x_int=x_interp, y=y_original)
    result_yes = post_yes(model_output, x=x_original, x_int=x_interp, y=y_original)
    assert len(result_no) >= 1
    assert len(result_yes) >= 1
    for sr in result_no + result_yes:
        assert sr.region.start >= 0 and sr.region.stop <= len(x_original)
        for peak in sr.peaks:
            assert "cen" in peak.parameters
            assert x_original.min() <= peak.parameters["cen"] <= x_original.max()
