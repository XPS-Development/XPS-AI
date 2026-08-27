"""Tests for guess_helpers and model guess_initial."""

import numpy as np
import pytest

from core.math_models import ModelRegistry
from core.math_models.guess_helpers import (
    amp_from_height,
    edge_intensities,
    half_max_sigma,
    peak_index_from_residuals,
)
from core.math_models.models import (
    ConstantBackgroundModel,
    LinearBackgroundModel,
    PseudoVoigtPeakModel,
    ShirleyBackgroundModel,
)


def test_half_max_sigma_symmetric_peak() -> None:
    """Sigma is half of FWHM for a symmetric peak."""
    x = np.linspace(-5, 5, 101)
    y = np.exp(-(x**2) / 2)
    max_idx = 50
    sig = half_max_sigma(x, y, max_idx)
    assert sig > 0
    assert 0.5 < sig < 2.0


def test_amp_from_height() -> None:
    """Amplitude scales with peak height and sigma."""
    y = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    max_idx = 2
    sig = 1.0
    frac = 0.5
    amp = amp_from_height(y, max_idx, sig, frac)
    assert amp > 0
    assert np.isfinite(amp)


def test_edge_intensities() -> None:
    """Averages over avg_on points at start and stop."""
    y = np.arange(10.0, 20.0)
    x = np.arange(y.size, dtype=float)
    start, stop = 2, 7
    avg_on = 2
    params = edge_intensities(x, y, start, stop, avg_on=avg_on)
    assert params["i1"] == 10.5
    assert params["i2"] == 17.5


def test_edge_intensities_clamps_to_bounds() -> None:
    """Start/stop near boundaries use available points only."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = np.arange(y.size, dtype=float)
    params = edge_intensities(x, y, start=1, stop=4, avg_on=3)
    assert np.isfinite(params["i1"])
    assert np.isfinite(params["i2"])
    assert params["i1"] == 1.0
    assert params["i2"] == 5.0


def test_peak_index_from_residuals() -> None:
    """Peak position is argmax of residuals."""
    residuals = np.array([0.0, 0.1, 1.0, 0.2, 0.0])
    idx = peak_index_from_residuals(residuals)
    assert idx == 2


def test_pseudo_voigt_guess_initial_at_max() -> None:
    """Returns amp/cen/sig/frac from x, y, peak_index."""
    x = np.linspace(-3, 3, 61)
    y = np.exp(-(x**2) / 2) + 0.1
    max_idx = 30
    frac = 0.5
    params = PseudoVoigtPeakModel.guess_initial(x, y, peak_index=max_idx, frac=frac)
    assert params["amp"] > 0
    assert np.isfinite(params["cen"])
    assert params["sig"] > 0
    assert params["frac"] == frac
    assert np.isclose(params["cen"], x[max_idx])


def test_pseudo_voigt_guess_initial_from_residuals_like_data() -> None:
    """Guessed peak sits near the residual/data maximum."""
    x = np.linspace(-5.0, 5.0, 201)
    y = np.exp(-(x**2) / 2.0)
    max_idx = peak_index_from_residuals(y)
    params = PseudoVoigtPeakModel.guess_initial(x, y, peak_index=max_idx)
    assert set(params.keys()) == {"amp", "cen", "sig", "frac"}
    assert params["amp"] > 0.0
    assert params["sig"] > 0.0
    assert 0.0 < params["frac"] <= 1.0
    assert np.isclose(params["cen"], x[max_idx])


@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (PseudoVoigtPeakModel, {"peak_index": 50}),
        (ConstantBackgroundModel, {"start": 10, "stop": 90}),
        (LinearBackgroundModel, {"start": 10, "stop": 90}),
        (ShirleyBackgroundModel, {"start": 10, "stop": 90}),
    ],
)
def test_guess_initial_keys_match_parameter_schema(model_cls, kwargs) -> None:
    """Every registered model returns keys matching parameter_schema."""
    x = np.linspace(0.0, 10.0, 101)
    y = np.exp(-((x - 5.0) ** 2) / 2.0) + 0.1
    model = ModelRegistry.get(model_cls.name)
    params = model.guess_initial(x, y, **kwargs)
    assert set(params.keys()) == {spec.name for spec in model.parameter_schema}


def test_constant_guess_initial_uses_min_edge() -> None:
    """Constant background uses min(i1, i2)."""
    y = np.linspace(10.0, 20.0, 100)
    x = np.arange(y.size, dtype=float)
    params = ConstantBackgroundModel.guess_initial(x, y, start=10, stop=90, avg_on=2)
    edges = edge_intensities(x, y, 10, 90, avg_on=2)
    assert params == {"const": min(edges["i1"], edges["i2"])}
