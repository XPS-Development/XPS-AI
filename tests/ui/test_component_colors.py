"""Tests for sequential peak color helpers."""

from ui.component_colors import PEAK_COLORS, color_for_component


def test_peak_palette_has_twelve_colors() -> None:
    """Peak colors come from a 12-color pool."""
    assert len(PEAK_COLORS) == 12
    assert PEAK_COLORS[0] == "#1f77b4"
    assert PEAK_COLORS[1] == "#ff7f0e"


def test_peak_colors_cycle_by_index() -> None:
    """The same peak index maps to the same palette color, wrapping the pool."""
    assert color_for_component(index=0) == PEAK_COLORS[0]
    assert color_for_component(index=1) == PEAK_COLORS[1]
    assert color_for_component(index=len(PEAK_COLORS)) == PEAK_COLORS[0]


def test_color_for_background_is_fixed() -> None:
    """Background components use a fixed gray, independent of index."""
    assert color_for_component(kind="background", index=0) == color_for_component(
        kind="background", index=7
    )
