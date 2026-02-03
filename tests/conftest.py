import numpy as np
import pytest

from core.math_models.model_funcs import pvoigt

RNG = np.random.default_rng(seed=42)

X_AXIS_START: float = -10.0
X_AXIS_STOP: float = 10.0
X_AXIS_NUM_POINTS: int = 200


@pytest.fixture
def x_axis() -> np.ndarray:
    return np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)


@pytest.fixture
def noise() -> np.ndarray:
    return RNG.normal(loc=0, scale=0.01, size=X_AXIS_NUM_POINTS)


@pytest.fixture
def simple_gauss(x_axis: np.ndarray) -> np.ndarray:
    amp = 1
    cen = 0
    sig = 1
    frac = 0

    y = pvoigt(x_axis, amp, cen, sig, frac)

    return y


@pytest.fixture
def simple_gauss_spectrum(x_axis, simple_gauss, noise):
    background = 1
    return x_axis, simple_gauss + noise + background
