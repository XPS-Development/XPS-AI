"""Smoke tests for core.numerics helpers."""

import numpy as np

from core.numerics import find_closest_index, interpolate, recalculate_idx


def test_interpolate_returns_requested_length() -> None:
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    y = np.array([0.0, 10.0, 20.0], dtype=np.float32)
    new_x, new_y = interpolate(x, y, num=5)
    assert len(new_x) == 5
    assert len(new_y) == 5
    assert new_x[0] == x[0]
    assert new_x[-1] == x[-1]
    np.testing.assert_allclose(new_y[0], 0.0)
    np.testing.assert_allclose(new_y[-1], 20.0)


def test_find_closest_index() -> None:
    array = np.array([1.0, 2.0, 3.0, 4.0])
    assert find_closest_index(2.1, array) == 1
    assert find_closest_index(0.0, array) == 0
    assert find_closest_index(10.0, array) == 3


def test_recalculate_idx_maps_interpolated_to_original() -> None:
    original = np.array([0.0, 10.0, 20.0, 30.0])
    interpolated = np.linspace(0.0, 30.0, 7)
    mid = len(interpolated) // 2
    mapped = recalculate_idx(mid, interpolated, original)
    assert mapped == find_closest_index(float(interpolated[mid]), original)


def test_recalculate_idx_out_of_range_returns_len_original() -> None:
    original = np.array([0.0, 1.0, 2.0])
    interpolated = np.array([0.0, 0.5, 1.0])
    assert recalculate_idx(len(interpolated), interpolated, original) == len(original)
