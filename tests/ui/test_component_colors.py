"""Tests for stable peak color helpers."""

from ui.component_colors import PEAK_COLORS, color_for_component


def test_peak_palette_has_twelve_colors() -> None:
    """Peak colors come from a 12-color pool."""
    assert len(PEAK_COLORS) == 12
    assert PEAK_COLORS[0] == "#1f77b4"
    assert PEAK_COLORS[1] == "#ff7f0e"


def test_peak_color_stays_with_component_id() -> None:
    """The same peak id keeps its color; another id does not take it over."""
    first = color_for_component(component_id="peak-a")
    second = color_for_component(component_id="peak-b")
    assert first in PEAK_COLORS
    assert second in PEAK_COLORS
    assert color_for_component(component_id="peak-a") == first
    assert color_for_component(component_id="peak-b") == second
    assert color_for_component(component_id=None) == PEAK_COLORS[0]


def test_color_for_background_is_fixed() -> None:
    """Background components use a fixed gray, independent of id."""
    assert color_for_component(kind="background", component_id="bg-a") == color_for_component(
        kind="background", component_id="bg-b"
    )
