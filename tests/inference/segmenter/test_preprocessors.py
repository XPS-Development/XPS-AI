"""
Tests for inference.segmenter: SegmenterPreprocessor output shape and value ranges.
"""

import numpy as np
import pytest

from core.dto import SpectrumDTO
from inference.segmenter import ONNXSegmenterAdapter, SegmenterPreprocessor


def _spectrum(x: np.ndarray, y: np.ndarray) -> SpectrumDTO:
    return SpectrumDTO(id_="s", parent_id=None, normalized=True, x=x, y=y)


@pytest.fixture
def spectrum_like() -> SpectrumDTO:
    x = np.linspace(0.0, 10.0, 100, dtype=np.float64)
    y = np.exp(-((x - 5.0) ** 2) / 2.0) + 0.1  # positive, normalized-like
    return _spectrum(x, y)


def test_segmenter_preprocessor_output_shape(spectrum_like: SpectrumDTO) -> None:
    """Preprocessor output has key INPUT_KEY and value shape (1, 2, num)."""
    pre = SegmenterPreprocessor(num=64)
    model_input, _ = pre(spectrum_like)
    assert ONNXSegmenterAdapter.INPUT_KEY in model_input
    arr = model_input[ONNXSegmenterAdapter.INPUT_KEY]
    assert arr.shape == (1, 2, 64)
    assert arr.dtype == np.float32


def test_segmenter_preprocessor_value_ranges(spectrum_like: SpectrumDTO) -> None:
    """First channel is original-like; second channel is log-scaled in [0, 1]."""
    pre = SegmenterPreprocessor(num=128)
    model_input, _ = pre(spectrum_like)
    arr = model_input[ONNXSegmenterAdapter.INPUT_KEY][0]  # (2, num)
    ch0, ch1 = arr[0], arr[1]
    assert ch0.min() >= 0
    assert ch0.max() <= spectrum_like.y.max() + 1e-5
    assert ch1.min() >= 0
    assert ch1.max() <= 1.0 + 1e-5


def test_segmenter_preprocessor_default_num() -> None:
    """Default interpolation number is ONNXSegmenterAdapter.INPUT_SHAPE[2]."""
    x = np.linspace(0, 1, 50)
    y = np.ones(50)
    pre = SegmenterPreprocessor()
    model_input, _ = pre(_spectrum(x, y))
    assert model_input[ONNXSegmenterAdapter.INPUT_KEY].shape == ONNXSegmenterAdapter.INPUT_SHAPE
