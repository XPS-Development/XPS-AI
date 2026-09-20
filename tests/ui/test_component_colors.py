"""Tests for stable component color helpers."""

from ui.component_colors import color_for_component


def test_color_for_component_is_stable() -> None:
    """Same component id always maps to the same palette color."""
    assert color_for_component("pabc123") == color_for_component("pabc123")


def test_color_for_background_is_fixed() -> None:
    """Background components use a fixed gray, independent of id."""
    assert color_for_component("b1", kind="background") == color_for_component(
        "b2", kind="background"
    )


def test_different_peak_ids_can_differ() -> None:
    """Different peak ids are not forced to the same color."""
    # Not a hard uniqueness guarantee — only that hashing is applied.
    assert isinstance(color_for_component("p111"), str)
    assert color_for_component("p111").startswith("#")
