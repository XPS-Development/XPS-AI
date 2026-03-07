"""
Tests for tools.automatization module (parameter guessing and AutomaticEvaluator).

Tool-level tests: pure functions use numpy arrays only; AutomaticEvaluator
uses CoreContext and DTOService from conftest fixtures.
"""

import numpy as np
import pytest

from tools.automatization import (
    guess_pseudo_voigt_sig_paramater,
    guess_pseudo_voigt_amp_parameter,
    calculate_background_intensities,
    guess_pseudo_voigt_params_at_max,
    guess_peak_position_by_residuals,
    AutomaticEvaluator,
)


# --- Pure function tests (no context) ---


def test_guess_pseudo_voigt_sig_paramater_symmetric_peak() -> None:
    """Sigma is half of FWHM for a symmetric peak."""
    x = np.linspace(-5, 5, 101)
    # Gaussian-like: max at center, sigma ~ 1
    y = np.exp(-(x**2) / 2)
    max_idx = 50
    sig = guess_pseudo_voigt_sig_paramater(x, y, max_idx)
    assert sig > 0
    # FWHM for exp(-x^2/2) is 2*sqrt(2*ln2) ~ 2.35; half is ~1.17
    assert 0.5 < sig < 2.0


def test_guess_pseudo_voigt_amp_parameter() -> None:
    """Amplitude scales with peak height and sigma."""
    y = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    max_idx = 2
    sig = 1.0
    frac = 0.5
    amp = guess_pseudo_voigt_amp_parameter(y, max_idx, sig, frac)
    assert amp > 0
    assert np.isfinite(amp)


def test_calculate_background_intensities() -> None:
    """Averages over avg_on points at start and stop."""
    y = np.arange(10.0, 20.0)  # 10..19
    start, stop = 2, 7
    avg_on = 2
    i1, i2 = calculate_background_intensities(y, start, stop, avg_on=avg_on)
    # i1 = mean(y[0:2]) = (10+11)/2 = 10.5
    # i2 = mean(y[7:9]) = (17+18)/2 = 17.5
    assert i1 == 10.5
    assert i2 == 17.5


def test_calculate_background_intensities_clamps_to_bounds() -> None:
    """Start/stop near boundaries use available points only."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # start=1: mean of y[max(0,0):1] = mean([1]) = 1; stop=4: mean of y[4:5] = 5
    i1, i2 = calculate_background_intensities(y, start=1, stop=4, avg_on=3)
    assert np.isfinite(i1) and np.isfinite(i2)
    assert i1 == 1.0
    assert i2 == 5.0


def test_guess_pseudo_voigt_params_at_max() -> None:
    """Returns (amp, cen, sig, frac) from x, y, max_idx."""
    x = np.linspace(-3, 3, 61)
    y = np.exp(-(x**2) / 2) + 0.1
    max_idx = 30
    frac = 0.5
    amp, cen, sig, frac_out = guess_pseudo_voigt_params_at_max(x, y, max_idx, frac=frac)
    assert amp > 0
    assert np.isfinite(cen)
    assert sig > 0
    assert frac_out == frac
    assert np.isclose(cen, x[max_idx])


def test_guess_peak_position_by_residuals() -> None:
    """Peak position is argmax of residuals."""
    x = np.linspace(0, 1, 5)
    residuals = np.array([0.0, 0.1, 1.0, 0.2, 0.0])
    idx = guess_peak_position_by_residuals(x, residuals)
    assert idx == 2


# --- AutomaticEvaluator tests (use ctx and dto_service from conftest) ---


@pytest.fixture
def evaluator(ctx, dto_service) -> AutomaticEvaluator:
    return AutomaticEvaluator(ctx, dto=dto_service)


def test_automatic_evaluator_calculate_background_parameters_constant(
    evaluator: AutomaticEvaluator,
    region_id: str,
) -> None:
    """With constant background, returns dict with 'const'."""
    start, stop = 25, 175
    result = evaluator.calculate_background_parameters(region_id, start, stop, mode="index", avg_on=3)
    assert result is not None
    assert "const" in result
    assert isinstance(result["const"], (int, float))


def test_automatic_evaluator_calculate_background_parameters_value_mode(
    evaluator: AutomaticEvaluator,
    region_id: str,
    dto_service,
) -> None:
    """Value mode converts start/stop via searchsorted on spectrum x."""
    spectrum = dto_service.get_spectrum(dto_service.get_region(region_id).parent_id)
    x = spectrum.x
    start_val, stop_val = float(x[25]), float(x[175])
    result = evaluator.calculate_background_parameters(
        region_id, start_val, stop_val, mode="value", avg_on=3
    )
    assert result is not None
    assert "const" in result


def test_automatic_evaluator_create_pseudo_voigt_peak_parameters(
    evaluator: AutomaticEvaluator,
    region_id: str,
) -> None:
    """Returns dict with amp, cen, sig, frac."""
    result = evaluator.create_pseudo_voigt_peak_parameters(region_id)
    assert "amp" in result
    assert "cen" in result
    assert "sig" in result
    assert "frac" in result
    assert all(np.isfinite(v) for v in result.values())


def test_automatic_evaluator_create_i1_i2_parameters(
    evaluator: AutomaticEvaluator,
    region_id: str,
) -> None:
    """Returns i1 and i2 from region slice and spectrum y."""
    result = evaluator.create_i1_i2_parameters(region_id, avg_on=3)
    assert "i1" in result
    assert "i2" in result
    assert all(isinstance(v, (int, float)) for v in result.values())
