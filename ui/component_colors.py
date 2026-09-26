"""Stable colors for peak/background components in plot and properties UI."""

from __future__ import annotations

import zlib

# Peak palette. A peak's color is a stable function of its id, not its order.
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


def color_for_component(*, kind: str = "peak", component_id: str | None = None) -> str:
    r"""
    Return a CSS-like color string for a component.

    Peak colors come from :data:`PEAK_COLORS` hashed from ``component_id``.
    Adding, removing, or replacing another peak does not recolor this one,
    and replacing a peak's model keeps the color because the id stays.

    Parameters
    ----------
    kind : str, optional
        ``\"peak\"`` or ``\"background\"``. Backgrounds use a fixed gray.
    component_id : str or None, optional
        Peak identifier. Ignored for backgrounds. Missing ids use the first
        palette color.

    Returns
    -------
    str
        Hex color string.
    """
    if kind == "background":
        return BACKGROUND_COLOR
    if not component_id:
        return PEAK_COLORS[0]
    slot = zlib.crc32(component_id.encode("utf-8")) % len(PEAK_COLORS)
    return PEAK_COLORS[slot]
