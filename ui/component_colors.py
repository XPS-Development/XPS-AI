"""Stable colors for peak/background components in plot and properties UI."""

from __future__ import annotations

# Peak palette: first is blue, second orange, then cycled by peak index.
PEAK_COLORS: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#393b79",
    "#e7ba52",
    "#843c39",
]

BACKGROUND_COLOR = "#444444"
ID_SUFFIX_HEX = "#888888"

# Structural status dots for the spectrum tree.
STATUS_COLOR_EMPTY = "#c0c0c0"
STATUS_COLOR_REGIONS = "#e6a817"
STATUS_COLOR_PEAKS = "#2ca02c"


def color_for_component(*, kind: str = "peak", index: int = 0) -> str:
    r"""
    Return a CSS-like color string for a component.

    Peak colors are taken from :data:`PEAK_COLORS` by ``index`` modulo the
    palette length, so the first peak is always blue, the second orange, and
    so on, independent of spectrum or component id.

    Parameters
    ----------
    kind : str, optional
        ``\"peak\"`` or ``\"background\"``. Backgrounds use a fixed gray.
    index : int, optional
        Zero-based peak order within a region. Ignored for backgrounds.

    Returns
    -------
    str
        Hex color string.
    """
    if kind == "background":
        return BACKGROUND_COLOR
    return PEAK_COLORS[index % len(PEAK_COLORS)]
