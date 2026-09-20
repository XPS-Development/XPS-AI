"""Helpers for soft UI slider ranges (region bounds and shared clipping)."""

from __future__ import annotations

import math
from typing import Literal


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


def prefer_finite_hard_bounds(lower: float, upper: float) -> tuple[float, float] | None:
    """Return ``(lower, upper)`` when both hard bounds are finite and ordered."""
    if math.isfinite(lower) and math.isfinite(upper) and upper > lower:
        return (float(lower), float(upper))
    return None


def clip_soft_range(
    lo: float,
    hi: float,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """
    Clip a soft ``(lo, hi)`` window by hard bounds and ensure ``lo < hi``.

    Parameters
    ----------
    lo, hi : float
        Proposed soft slider bounds.
    lower, upper : float
        Hard fit bounds (may be ``±inf``).

    Returns
    -------
    tuple[float, float]
        Inclusive soft range with ``lo < hi``.
    """
    if math.isfinite(lower):
        lo = max(lo, float(lower))
    if math.isfinite(upper):
        hi = min(hi, float(upper))
    if hi <= lo:
        hi = lo + 1.0
    return (float(lo), float(hi))


def soft_window_around(
    value: float,
    *,
    span: float,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Return a soft window centered on ``value`` with half-width ``span``."""
    v = float(value) if math.isfinite(value) else 0.0
    return clip_soft_range(v - span, v + span, lower, upper)


def intensity_soft_range(
    value: float,
    lower: float,
    upper: float,
    *,
    y_max: float | None = None,
    non_negative: bool = False,
) -> tuple[float, float]:
    """
    Soft slider range for intensity-like parameters (``amp``, ``const``, ``i1``/``i2``).

    Parameters
    ----------
    value : float
        Current parameter value.
    lower, upper : float
        Hard fit bounds.
    y_max : float or None, optional
        Peak intensity scale hint.
    non_negative : bool, optional
        When True, soft lower bound defaults to 0 if hard lower is open.
    """
    hard = prefer_finite_hard_bounds(lower, upper)
    if hard is not None:
        return hard
    v = float(value) if math.isfinite(value) else 0.0
    if non_negative:
        lo = 0.0 if (not math.isfinite(lower) or lower < 0) else float(lower)
        hi_candidates = [abs(v) * 2.0, abs(v) + 1.0, 1.0]
        if y_max is not None and math.isfinite(y_max):
            hi_candidates.append(abs(float(y_max)) * 2.0)
        hi = max(hi_candidates)
        return clip_soft_range(lo, hi, lower, upper)
    span = max(abs(v) * 0.5, abs(y_max or 0.0) * 0.25, 1.0)
    return soft_window_around(v, span=span, lower=lower, upper=upper)
