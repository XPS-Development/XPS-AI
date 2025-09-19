import os
import sys

# Добавляем корень проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from lib.spectra import SpectrumCollection, Spectrum, Region, Peak, PeakParameter
from lib.optimization import OptimizationManager


@pytest.fixture
def mock_peak():
    peak = MagicMock()
    peak.id = "p123"
    peak.region_id = "r1"
    peak.amp_par.name = "amp"
    peak.amp_par.value = 5.0
    peak.amp_par.vary = True
    peak.amp_par.min = 0
    peak.amp_par.max = 10
    peak.amp_par.expr = None

    peak.cen_par.name = "cen"
    peak.cen_par.value = 1.0
    peak.cen_par.vary = True
    peak.cen_par.min = 0
    peak.cen_par.max = 2
    peak.cen_par.expr = None

    peak.sig_par.name = "sig"
    peak.sig_par.value = 0.5
    peak.sig_par.vary = True
    peak.sig_par.min = 0.1
    peak.sig_par.max = 2
    peak.sig_par.expr = None

    peak.frac_par.name = "frac"
    peak.frac_par.value = 0.3
    peak.frac_par.vary = True
    peak.frac_par.min = 0
    peak.frac_par.max = 1
    peak.frac_par.expr = None
    return peak


@pytest.fixture
def mock_region(mock_peak):
    region = MagicMock()
    region.id = "r1"
    region.x = np.array([1, 2, 3])
    region.y = np.array([10, 20, 30])
    region.y_norm = np.array([0.1, 0.2, 0.3])
    region.norm_coefs = (10, 100)
    region.peaks = [mock_peak]
    return region


@pytest.fixture
def mock_collection(mock_region, mock_peak):
    collection = MagicMock()
    collection.get_region.return_value = mock_region
    collection.get_peak.return_value = mock_peak
    return collection


def test_set_collection(mock_collection):
    mgr = OptimizationManager()
    mgr.set_collection(mock_collection)
    assert mgr.collection is mock_collection


def test_parse_expr():
    mgr = OptimizationManager()
    expr = "p123 + 2"
    parsed = mgr.parse_expr(expr, "amp")
    assert parsed == "p123_amp + 2"


@patch("lib.optimization.Parameter")
@patch("lib.optimization.norm_with_coefs", return_value=42)
def test_peakparam_to_param(mock_norm, mock_param, mock_peak):
    mgr = OptimizationManager()
    param = mgr.peakparam_to_param("p123", mock_peak.amp_par, (0, 100))
    mock_param.assert_called_once()
    assert mock_param.call_args[0][0] == "p123_amp"
    assert mock_param.call_args[1]["value"] == 42


def test_get_peak_params(mock_peak):
    mgr = OptimizationManager()
    amp, cen, sig, frac = mgr.get_peak_params(mock_peak)
    assert amp.name == "amp"
    assert frac.name == "frac"


@patch("lib.optimization.denorm_with_coefs", return_value=77)
def test_update_peak_param_values(mock_denorm, mock_collection):
    mgr = OptimizationManager()
    mgr.set_collection(mock_collection)

    parameters = {"p123_amp": MagicMock(value=5.0)}
    mgr.update_peak_param_values(parameters, from_norm=True)

    mock_collection.get_peak.assert_called_with("p123")
    mock_collection.get_region.assert_called_with("r1")
    mock_denorm.assert_called_once()


@patch.object(OptimizationManager, "peakparam_to_param")
def test_peak_to_params(mock_peakparam_to_param, mock_peak):
    mgr = OptimizationManager()
    mock_peakparam_to_param.return_value = "param"
    params = mgr.peak_to_params(mock_peak)
    assert len(params) == 4
    assert all(p == "param" for p in params)


@patch.object(OptimizationManager, "peak_to_params", return_value=["a", "b"])
def test_peaks_to_params(mock_peak_to_params, mock_peak):
    mgr = OptimizationManager()
    peaks = [mock_peak, mock_peak]
    params = mgr.peaks_to_params(peaks)
    assert params == ["a", "b", "a", "b"]


def test_get_combinations(mock_peak):
    mgr = OptimizationManager()
    combos = mgr.get_combinations([mock_peak])
    assert combos == ("p123",)


@patch.object(OptimizationManager, "peaks_to_params", return_value=["param"])
def test_prepare_region(mock_peaks_to_params, mock_region):
    mgr = OptimizationManager()
    x, y, combos, params = mgr.prepare_region(mock_region, normalize=True)
    assert np.allclose(x, mock_region.x)
    assert combos == ("p123",)
    assert params == ["param"]


def test_resolve_dependencies():
    mgr = OptimizationManager()
    param1 = MagicMock(name="p123_amp")
    param1.name = "p123_amp"
    param1.expr = "p999"
    mgr.resolve_dependencies([param1])
    assert param1.expr is None


@patch("lib.optimization.Optimizer")
@patch("lib.optimization.Parameters")
def test_get_regions_opt(mock_params_cls, mock_optimizer, mock_collection, mock_region):
    mgr = OptimizationManager()
    mgr.set_collection(mock_collection)

    mock_params = MagicMock()
    mock_params_cls.return_value = mock_params

    opt = mgr.get_regions_opt(["r1"], normalize=True)
    mock_optimizer.assert_called_once()
    assert opt == mock_optimizer.return_value


@patch.object(OptimizationManager, "get_regions_opt")
@patch.object(OptimizationManager, "update_peak_param_values")
def test_proceed_regions_opt(mock_update, mock_get_regions_opt, mock_collection):
    mgr = OptimizationManager()
    mgr.set_collection(mock_collection)

    mock_opt = MagicMock()
    mock_opt.fit.return_value = {"p123_amp": MagicMock(value=5.0)}
    mock_get_regions_opt.return_value = mock_opt

    mgr.proceed_regions_opt(["r1"], normalize=True)
    mock_opt.fit.assert_called_once_with(return_result=False)
    mock_update.assert_called_once()


# === Real tests ===


@pytest.fixture
def simple_collection():
    # Создаём коллекцию с одним спектром, одним регионом и одним пиком
    collection = SpectrumCollection()
    x = np.linspace(0, 10, 100)
    y = np.exp(-((x - 5) ** 2))  # простая "гауссиана"
    spectrum = Spectrum(id="s1", x=x, y=y, norm_coefs=(0, 1))
    region = Region(id="r1", x=x, y=y, norm_coefs=(0, 1), spectrum_id="s1")

    peak = Peak()
    peak.set("amp", value=1.0, min_val=0.0, max_val=2.0, vary=True)
    peak.set("cen", value=5.0, min_val=4.0, max_val=6.0, vary=True)
    peak.set("sig", value=1.0, min_val=0.1, max_val=2.0, vary=True)
    peak.set("frac", value=0.0, min_val=0.0, max_val=1.0, vary=False)
    peak.id = "p123"

    region.add_peak(peak)
    spectrum.add_region(region)
    collection.add_spectrum(spectrum)
    return collection, spectrum, region, peak


def test_get_regions_opt_builds_optimizer(simple_collection):
    collection, _, region, peak = simple_collection

    mgr = OptimizationManager()
    mgr.set_collection(collection)

    opt = mgr.get_regions_opt([region.id], normalize=True)

    # Проверяем что оптимизатор получил правильные данные
    assert len(opt.x) == 1
    assert len(opt.y) == 1
    assert peak.id in opt.combinations[0]
    assert any(p.name.startswith(peak.id) for p in opt.init_params)


def test_proceed_regions_opt_updates_peak_values(simple_collection, monkeypatch):
    collection, _, region, peak = simple_collection

    mgr = OptimizationManager()
    mgr.set_collection(collection)

    # Подменяем Optimizer.fit, чтобы вернуть "подогнанные" значения
    class DummyParams(dict):
        def __getitem__(self, item):
            return super().__getitem__(item)

    dummy_params = DummyParams(
        {
            f"{peak.id}_amp": type("Obj", (), {"value": 1.5}),
            f"{peak.id}_cen": type("Obj", (), {"value": 5.2}),
            f"{peak.id}_sig": type("Obj", (), {"value": 0.8}),
            f"{peak.id}_frac": type("Obj", (), {"value": 0.0}),
        }
    )

    class DummyOpt:
        def fit(self, return_result=False):
            return dummy_params

    monkeypatch.setattr("lib.optimization.Optimizer", lambda *a, **kw: DummyOpt())

    # Запускаем оптимизацию
    mgr.proceed_regions_opt([region.id], normalize=False)

    # Проверяем что значения пика обновились
    assert peak.amp_par.value == pytest.approx(1.5)
    assert peak.cen_par.value == pytest.approx(5.2)
    assert peak.sig_par.value == pytest.approx(0.8)
    assert peak.frac_par.value == pytest.approx(0.0)
