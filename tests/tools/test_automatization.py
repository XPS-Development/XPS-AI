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
    create_pseudo_voigt_peak_parameters,
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
    x = np.arange(y.size, dtype=float)
    start, stop = 2, 7
    avg_on = 2
    params = calculate_background_intensities(x, y, start, stop, avg_on=avg_on)
    # i1 = mean(y[0:2]) = (10+11)/2 = 10.5
    # i2 = mean(y[7:9]) = (17+18)/2 = 17.5
    assert params["i1"] == 10.5
    assert params["i2"] == 17.5


def test_calculate_background_intensities_clamps_to_bounds() -> None:
    """Start/stop near boundaries use available points only."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = np.arange(y.size, dtype=float)
    # start=1: mean of y[max(0,0):1] = mean([1]) = 1; stop=4: mean of y[4:5] = 5
    params = calculate_background_intensities(x, y, start=1, stop=4, avg_on=3)
    assert np.isfinite(params["i1"]) and np.isfinite(params["i2"])
    assert params["i1"] == 1.0
    assert params["i2"] == 5.0


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


class _DummyModel:
    def evaluate(  # type: ignore[override]
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        **params: float,
    ) -> np.ndarray:
        return np.zeros_like(x)


class _DummyParameter:
    def __init__(self, value: float) -> None:
        self.value = value


class _DummyComponent:
    def __init__(self, component_id: str, parent_id: str) -> None:
        self.id_ = component_id
        self.parent_id = parent_id
        self.normalized = False
        self.kind = "background"
        self.model = _DummyModel()
        self.parameters = {"p": _DummyParameter(0.0)}


class _DummyRegion:
    def __init__(self, region_id: str, parent_id: str, x: np.ndarray, y: np.ndarray) -> None:
        self.id_ = region_id
        self.parent_id = parent_id
        self.normalized = False
        self.x = x
        self.y = y


def test_create_pseudo_voigt_peak_parameters_uses_region_bundle() -> None:
    """create_pseudo_voigt_peak_parameters returns peak params from residuals."""
    x = np.linspace(-5.0, 5.0, 201)
    y = np.exp(-(x**2) / 2.0)
    region = _DummyRegion("region-1", "spectrum-1", x=x, y=y)
    components = (_DummyComponent("bg-1", region.id_),)

    params = create_pseudo_voigt_peak_parameters(region, components)

    assert set(params.keys()) == {"amp", "cen", "sig", "frac"}
    assert params["amp"] > 0.0
    assert params["sig"] > 0.0
    assert 0.0 < params["frac"] <= 1.0
    max_idx = int(np.argmax(y))
    assert np.isclose(params["cen"], x[max_idx])
