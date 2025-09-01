import sys
import os

# Добавляем корень проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from lmfit import Parameters
from lib.optimization import Optimizer
from lib.funcs import ndpvoigt


@pytest.fixture
def make_data_single():
    """Создаёт синтетический спектр с двумя пиками (один x, один y)."""
    x = np.linspace(-5, 5, 200)

    params_true = Parameters()
    params_true.add_many(
        ("p0_amp", 1.0, True, 0, None),
        ("p0_cen", 0.0, True, -5, 5),
        ("p0_sig", 1.0, True, 0.1, 5),
        ("p0_frac", 0.5, True, 0, 1),
        ("p1_amp", 0.8, True, 0, None),
        ("p1_cen", 2.0, True, -5, 5),
        ("p1_sig", 0.5, True, 0.1, 5),
        ("p1_frac", 0.3, True, 0, 1),
    )

    combinations = (("p0", "p1"),)

    y_clean = ndpvoigt(params_true, (x,), combinations)[0]
    y_noisy = y_clean + 0.05 * np.random.normal(size=len(x))

    return x, y_noisy, params_true, combinations


@pytest.fixture
def make_data_multi():
    """Создаёт данные для двух спектров с 4 пиками."""
    x1 = np.linspace(-5, 5, 200)
    x2 = np.linspace(-3, 7, 200)

    params_true = Parameters()
    params_true.add_many(
        # Спектр 1 (пики 0,1)
        ("p0_amp", 1.0, True, 0, None),
        ("p0_cen", 0.0, True, -5, 5),
        ("p0_sig", 1.0, True, 0.1, 5),
        ("p0_frac", 0.5, True, 0, 1),
        ("p1_amp", 0.8, True, 0, None),
        ("p1_cen", 2.0, True, -5, 5),
        ("p1_sig", 0.5, True, 0.1, 5),
        ("p1_frac", 0.3, True, 0, 1),
        # Спектр 2 (пики 2,3)
        ("p2_amp", 1.2, True, 0, None),
        ("p2_cen", 1.0, True, -3, 7),
        ("p2_sig", 0.7, True, 0.1, 5),
        ("p2_frac", 0.4, True, 0, 1),
        ("p3_amp", 0.6, True, 0, None),
        ("p3_cen", 4.0, True, -3, 7),
        ("p3_sig", 1.0, True, 0.1, 5),
        ("p3_frac", 0.6, True, 0, 1),
    )

    combinations = (("p0", "p1"), ("p2", "p3"))
    y1_clean = ndpvoigt(params_true, (x1,), combinations[:1])[0]
    y2_clean = ndpvoigt(params_true, (x2,), combinations[1:])[0]

    y1 = y1_clean + 0.05 * np.random.normal(size=len(x1))
    y2 = y2_clean + 0.05 * np.random.normal(size=len(x2))

    return (x1, x2), (y1, y2), params_true, combinations


def test_init_and_validation(make_data_single):
    x, y, params_true, combinations = make_data_single

    # Ошибка если параметры не делятся на 4
    with pytest.raises(ValueError):
        bad_params = Parameters()
        bad_params.add("p4_amp", value=1)
        Optimizer(x, y, bad_params, model=ndpvoigt)

    # Ошибка если x и y разные по длине
    with pytest.raises(ValueError):
        Optimizer(x, y[:-1], params_true, combinations, model=ndpvoigt)


def test_ndresid_shape(make_data_single):
    x, y, params_true, combinations = make_data_single

    # создаём слегка неправильные параметры
    fit_params = params_true.copy()
    fit_params["p0_cen"].set(value=-0.5)

    opt = Optimizer(x, y, fit_params, combinations, model=ndpvoigt)
    resid = opt.ndresid(fit_params, (x,), (y,), combinations)

    assert resid.shape == y.shape
    assert np.isfinite(resid).all()


def test_fit_improves_residual(make_data_single):
    x, y, params_true, combinations = make_data_single

    # испортим стартовые параметры
    fit_params = Parameters()
    fit_params.add_many(
        ("p0_amp", 0.5, True, 0, None),
        ("p0_cen", -2.0, True, -5, 5),
        ("p0_sig", 2.0, True, 0.1, 5),
        ("p0_frac", 0.1, True, 0, 1),
        ("p1_amp", 0.5, True, 0, None),
        ("p1_cen", 3.0, True, -5, 5),
        ("p1_sig", 1.5, True, 0.1, 5),
        ("p1_frac", 0.9, True, 0, 1),
    )

    opt = Optimizer(x, y, fit_params, combinations, model=ndpvoigt)

    resid_before = np.sum(opt.ndresid(fit_params, (x,), (y,), combinations) ** 2)
    res = opt.fit(return_result=True)
    resid_after = np.sum(opt.ndresid(res.params, (x,), (y,), combinations) ** 2)

    assert resid_after < resid_before


def test_multi_spectrum_fit(make_data_multi):
    (x1, x2), (y1, y2), params_true, combinations = make_data_multi

    # Начальные (искажённые) параметры
    fit_params = params_true.copy()
    fit_params["p0_cen"].set(value=-1.0)
    fit_params["p1_cen"].set(value=3.0)
    fit_params["p2_cen"].set(value=0.0)
    fit_params["p3_cen"].set(value=5.0)

    opt = Optimizer(
        (x1, x2),
        (y1, y2),
        fit_params,
        combinations,
        model=ndpvoigt,
    )

    resid_before = np.sum(
        opt.ndresid(fit_params, (x1, x2), (y1, y2), combinations) ** 2
    )
    res = opt.fit(return_result=True)
    resid_after = np.sum(opt.ndresid(res.params, (x1, x2), (y1, y2), combinations) ** 2)

    assert resid_after < resid_before


def test_linked_parameters_across_spectra(make_data_multi):
    (x1, x2), (y1, y2), params, combinations = make_data_multi

    # Добавим зависимость: амплитуда пика 3 = половина амплитуды пика 0
    params["p3_amp"].set(expr="p0_amp * 0.5")

    opt = Optimizer(
        (x1, x2),
        (y1, y2),
        params,
        combinations,
        model=ndpvoigt,
    )

    res = opt.fit(return_result=True)

    assert np.isclose(
        res.params["p3_amp"].value, res.params["p0_amp"].value * 0.5, rtol=1e-6
    )


def test_invalid_expression_reference(make_data_single):
    x, y, params, combinations = make_data_single

    # expr указывает на несуществующий параметр
    params["p1_amp"].set(expr="nonexistent_param * 2")

    opt = Optimizer(x, y, params, combinations, model=ndpvoigt)

    with pytest.raises(NameError):
        _ = opt.fit(return_result=True)
