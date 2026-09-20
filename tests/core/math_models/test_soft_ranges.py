"""Tests for soft UI parameter ranges."""

import math

from core.math_models.soft_ranges import soft_parameter_range


def test_frac_uses_unit_interval() -> None:
    """frac always maps to [0, 1]."""
    assert soft_parameter_range("frac", 0.5, -math.inf, math.inf) == (0.0, 1.0)


def test_finite_schema_bounds_preferred() -> None:
    """Finite lower/upper from the schema win over heuristics."""
    assert soft_parameter_range("amp", 1.0, 0.0, 2.5) == (0.0, 2.5)


def test_cen_uses_x_extent_when_provided() -> None:
    """cen soft range follows the region x window."""
    lo, hi = soft_parameter_range("cen", 100.0, -math.inf, math.inf, x_min=90.0, x_max=110.0)
    assert lo == 90.0
    assert hi == 110.0


def test_amp_is_non_negative() -> None:
    """amp soft range starts at zero and extends above the current value."""
    lo, hi = soft_parameter_range("amp", 10.0, -math.inf, math.inf, y_max=20.0)
    assert lo == 0.0
    assert hi >= 20.0
