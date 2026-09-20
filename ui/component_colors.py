"""Stable colors for peak/background components in plot and properties UI."""

from __future__ import annotations

import zlib

# Peak palette (cycled by stable hash of component id).
PEAK_COLORS: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]

BACKGROUND_COLOR = "#444444"
ID_SUFFIX_HEX = "#888888"

# Structural status dots for the spectrum tree.
STATUS_COLOR_EMPTY = "#c0c0c0"
STATUS_COLOR_REGIONS = "#e6a817"
STATUS_COLOR_PEAKS = "#2ca02c"


def color_for_component(component_id: str, *, kind: str = "peak") -> str:
    r"""
    Return a stable CSS-like color string for a component.

    Parameters
    ----------
    component_id : str
        Full component identifier.
    kind : str, optional
        ``\"peak\"`` or ``\"background\"``. Backgrounds use a fixed gray.

    Returns
    -------
    str
        Hex color string.
    """
    if kind == "background":
        return BACKGROUND_COLOR
    digest = zlib.crc32(component_id.encode("utf-8")) & 0xFFFFFFFF
    return PEAK_COLORS[digest % len(PEAK_COLORS)]
