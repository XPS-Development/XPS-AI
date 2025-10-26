import pytest
import numpy as np
from scipy.stats import norm

from lib.funcs import static_shirley_background, linear_background


def test_shirley_background_converges_with_background():
    # сетка
    x = np.linspace(0, 10, 200)

    # сигнал: пик + плавный фон
    peak = np.exp(-((x - 5) ** 2) / (2 * 0.5**2))  # гауссиан
    background = 0.2 + 0.5 * norm.cdf(x, loc=5, scale=0.5)  # фон как CDF
    y = peak + background

    bg = static_shirley_background(x, y, i_1=y[0], i_2=y[-1], iters=8)

    assert bg.shape == x.shape
    # фон должен быть в пределах интенсивностей
    assert np.all(bg >= min(y[0], y[-1]) - 1e-6)
    assert np.all(bg <= y.max() + 1e-6)
    # фон должен совпадать с CDF
    assert np.allclose(bg, background, atol=1e-3)


def test_linear_background():
    x = np.linspace(0, 10, 5)
    bg = linear_background(x, i_1=0, i_2=10)
    # linear interpolation: bg[0]=0, bg[-1]=10
    assert np.allclose(bg[0], 0)
    assert np.allclose(bg[-1], 10)
    # Should be strictly increasing
    assert np.all(np.diff(bg) > 0)
