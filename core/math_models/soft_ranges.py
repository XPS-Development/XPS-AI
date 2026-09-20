"""Soft UI ranges for interactive parameter controls (sliders)."""

from __future__ import annotations

import math
from typing import Literal


def soft_parameter_range(
    name: str,
    value: float,
    lower: float,
    upper: float,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    y_max: float | None = None,
) -> tuple[float, float]:
    """
    Return a finite ``(lo, hi)`` range suitable for a UI slider.

    Prefer schema bounds when both are finite. Otherwise derive a soft window
    from the parameter name and optional spectrum context.

    Parameters
    ----------
    name : str
        Parameter name (e.g. ``amp``, ``cen``, ``sig``, ``frac``).
    value : float
        Current parameter value.
    lower, upper : float
        Hard fit bounds (may be ``±inf``).
    x_min, x_max : float or None, optional
        Spectrum / region x extent (used for ``cen``).
    y_max : float or None, optional
        Peak intensity scale hint (used for amplitudes / intensities).

    Returns
    -------
    tuple[float, float]
        Inclusive soft range with ``lo < hi``.
    """
    if math.isfinite(lower) and math.isfinite(upper) and upper > lower:
        return (float(lower), float(upper))

    key = name.lower()
    v = float(value) if math.isfinite(value) else 0.0

    if key == "frac":
        return (0.0, 1.0)

    if key == "cen":
        if x_min is not None and x_max is not None and x_max != x_min:
            lo, hi = (float(x_min), float(x_max)) if x_min < x_max else (float(x_max), float(x_min))
            return (lo, hi)
        span = max(abs(v) * 0.1, 5.0)
        return (v - span, v + span)

    if key in {"amp", "sig"}:
        lo = 0.0 if (not math.isfinite(lower) or lower < 0) else float(lower)
        hi_candidates = [abs(v) * 2.0, abs(v) + 1.0, 1.0]
        if y_max is not None and math.isfinite(y_max):
            hi_candidates.append(abs(float(y_max)) * (2.0 if key == "amp" else 0.5))
        hi = max(hi_candidates)
        if math.isfinite(upper):
            hi = min(hi, float(upper))
        if hi <= lo:
            hi = lo + 1.0
        return (lo, hi)

    if key in {"const", "i1", "i2"}:
        span = max(abs(v) * 0.5, abs(y_max or 0.0) * 0.25, 1.0)
        lo = v - span
        hi = v + span
        if math.isfinite(lower):
            lo = max(lo, float(lower))
        if math.isfinite(upper):
            hi = min(hi, float(upper))
        if hi <= lo:
            hi = lo + 1.0
        return (lo, hi)

    # Generic fallback: window around the current value.
    span = max(abs(v) * 0.5, 1.0)
    lo, hi = v - span, v + span
    if math.isfinite(lower):
        lo = max(lo, float(lower))
    if math.isfinite(upper):
        hi = min(hi, float(upper))
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def soft_region_bound_range(
    *,
    mode: Literal["value", "index"] = "value",
    x_min: float | None = None,
    x_max: float | None = None,
    index_count: int | None = None,
) -> tuple[float, float]:
    """
    Return a finite ``(lo, hi)`` range for region start/stop sliders.

    Parameters
    ----------
    mode : {"value", "index"}, optional
        Whether bounds are axis values or sample indices.
    x_min, x_max : float or None, optional
        Spectrum x extent (value mode).
    index_count : int or None, optional
        Number of spectrum samples (index mode).

    Returns
    -------
    tuple[float, float]
        Inclusive soft range with ``lo < hi``.
    """
    if mode == "index":
        n = max(int(index_count or 2), 2)
        return (0.0, float(n - 1))
    if x_min is not None and x_max is not None and x_max != x_min:
        lo, hi = (float(x_min), float(x_max)) if x_min < x_max else (float(x_max), float(x_min))
        return (lo, hi)
    return (0.0, 1.0)
