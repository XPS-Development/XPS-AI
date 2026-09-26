"""Tests for soft UI parameter ranges on models and region helpers."""

import math

from core.math_models import ModelRegistry
from core.math_models.models import (
    ConstantBackgroundModel,
    PseudoVoigtPeakModel,
    ShirleyBackgroundModel,
)
from core.math_models.soft_ranges import soft_region_bound_range


def test_frac_uses_unit_interval() -> None:
    """frac always maps to [0, 1]."""
    assert PseudoVoigtPeakModel.soft_parameter_range("frac", 0.5, -math.inf, math.inf) == (
        0.0,
        1.0,
    )


def test_finite_schema_bounds_preferred() -> None:
    """Finite lower/upper from the schema win over heuristics."""
    assert PseudoVoigtPeakModel.soft_parameter_range("amp", 1.0, 0.0, 2.5) == (0.0, 2.5)


def test_cen_uses_x_extent_when_provided() -> None:
    """cen soft range follows the region x window."""
    lo, hi = PseudoVoigtPeakModel.soft_parameter_range(
        "cen", 100.0, -math.inf, math.inf, x_min=90.0, x_max=110.0
    )
    assert lo == 90.0
    assert hi == 110.0


def test_amp_is_non_negative() -> None:
    """amp soft range starts at zero and extends above the current value."""
    lo, hi = PseudoVoigtPeakModel.soft_parameter_range("amp", 10.0, -math.inf, math.inf, y_max=20.0)
    assert lo == 0.0
    assert hi >= 20.0


def test_sig_uses_fixed_soft_window() -> None:
    """sig soft slider stays within 0.1 .. 8 when hard bounds are open."""
    assert PseudoVoigtPeakModel.soft_parameter_range("sig", 1.5, -math.inf, math.inf) == (
        0.1,
        8.0,
    )


def test_sig_respects_tighter_hard_bounds() -> None:
    """Finite hard bounds clip the default sig soft window, and never exceed 8."""
    assert PseudoVoigtPeakModel.soft_parameter_range("sig", 2.0, 0.5, 10.0) == (0.5, 8.0)
    assert PseudoVoigtPeakModel.soft_parameter_range("sig", 2.0, 0.5, 4.0) == (0.5, 4.0)


def test_registry_soft_range_matches_model_class() -> None:
    """Registry instances expose the same soft_parameter_range as the class."""
    model = ModelRegistry.get("pseudo-voigt")
    assert model.soft_parameter_range("sig", 1.0, -math.inf, math.inf) == (0.1, 8.0)


def test_background_intensity_soft_range() -> None:
    """Background endpoint intensities use an intensity soft window."""
    lo, hi = ShirleyBackgroundModel.soft_parameter_range(
        "i1", 100.0, -math.inf, math.inf, y_max=200.0
    )
    assert lo < 100.0
    assert hi > 100.0
    lo_c, hi_c = ConstantBackgroundModel.soft_parameter_range(
        "const", 50.0, -math.inf, math.inf, y_max=200.0
    )
    assert lo_c < 50.0
    assert hi_c > 50.0


def test_region_bound_value_mode_uses_x_extent() -> None:
    """Value-mode region sliders follow the spectrum x window."""
    assert soft_region_bound_range(mode="value", x_min=10.0, x_max=50.0) == (10.0, 50.0)


def test_region_bound_index_mode_uses_sample_count() -> None:
    """Index-mode region sliders span ``[0, n-1]``."""
    assert soft_region_bound_range(mode="index", index_count=100) == (0.0, 99.0)
