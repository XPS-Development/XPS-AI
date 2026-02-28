"""
Tests for tools.nn.segmenter: ONNXSegmenterAdapter output shape (with mock input).
"""

import numpy as np
import pytest

from tools.nn.segmenter import ONNXSegmenterAdapter


def test_onnx_adapter_run_requires_loaded_model() -> None:
    """Adapter without loaded model raises on run()."""
    adapter = ONNXSegmenterAdapter(model_path=None)
    inp = {ONNXSegmenterAdapter.INPUT_KEY: np.zeros((1, 2, 256), dtype=np.float32)}
    with pytest.raises(RuntimeError, match="no model loaded"):
        adapter.run(inp)


def test_onnx_adapter_run_requires_input_key() -> None:
    """Adapter run() with missing input key raises KeyError."""

    class MockSession:
        def run(self, _out_names: list, feed_dict: dict) -> list:
            return []  # never called when KeyError raised first

    adapter = ONNXSegmenterAdapter(model_path=None)
    adapter._session = MockSession()
    with pytest.raises(KeyError, match=ONNXSegmenterAdapter.INPUT_KEY):
        adapter.run({"wrong_key": np.zeros((1, 2, 256), dtype=np.float32)})


def test_onnx_adapter_output_shape_with_mock_session() -> None:
    """When run succeeds, output has PEAK_MASK_KEY and MAX_MASK_KEY with 1d arrays."""

    class MockSession:
        def run(self, _out_names: list, feed_dict: dict) -> list:
            inp = feed_dict[ONNXSegmenterAdapter.INPUT_KEY]
            _, _, n = inp.shape
            # Return shape (1, 2, n) as real model would
            out = np.zeros((1, 2, n), dtype=np.float32)
            out[0, 0, :] = 0.6  # region mask
            out[0, 1, :] = 0.0
            out[0, 1, n // 2] = 0.7  # max mask
            return [out]

    adapter = ONNXSegmenterAdapter(model_path=None)
    adapter._session = MockSession()
    inp = {ONNXSegmenterAdapter.INPUT_KEY: np.zeros((1, 2, 64), dtype=np.float32)}
    out = adapter.run(inp)
    assert ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[0] in out
    assert ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[1] in out
    assert out[ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[0]].shape == (64,)
    assert out[ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[1]].shape == (64,)
