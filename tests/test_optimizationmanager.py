import pytest
from uuid import uuid4
import numpy as np

from lmfit import Parameter
from lib.spectra import SpectrumCollection, Spectrum, Region, Peak, PeakParameter
from lib.optimization import OptimizationManager


@pytest.fixture
def simple_param():
    param = PeakParameter("param", 1.0, vary=True, min_val=0, max_val=10)
    return param


@pytest.fixture
def simple_peak():
    peak = Peak()
    return peak


@pytest.fixture
def simple_collection_factory(monkeypatch):
    monkeypatch.setattr(Region, "background", property(lambda self: np.zeros(len(self.x))))

    def _make_collection(
        n_spectra: int, regions_per_spectrum: int, peaks_per_region: int
    ) -> SpectrumCollection:

        collection = SpectrumCollection()

        for i in range(n_spectra):
            x = np.linspace(0, 10, 100)
            y = np.sin(x + i)
            s = Spectrum(name=f"S_{i}", x=x, y=y)
            collection.register(s)

            for r in range(regions_per_spectrum):
                r = s.create_region(r * 10, (r + 1) * 10)
                collection.add_link(s, r)

                for p in range(peaks_per_region):
                    peak = Peak(cen=p)
                    collection.add_link(r, peak)

        return collection

    return _make_collection


@pytest.fixture
def sin_spectrum():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    return Spectrum(x=x, y=y, name="test_spectrum")


@pytest.fixture
def simple_collection(sin_spectrum):
    spec = sin_spectrum
    collection = SpectrumCollection()
    collection.register(spec)
    collection.add_link(
        spec,
    )
    return collection


def test_parse_expression():
    opt = OptimizationManager()
    expr = "p123 + 2"
    parsed = opt.parse_expr(expr, "amp")
    assert parsed == "p123_amp + 2"


def test_resolve_dependencies():
    opt = OptimizationManager()
    # valid
    params = [
        Parameter("p1_amp", expr=None),
        Parameter("p2_amp", expr="p1_amp + 1"),
    ]
    opt.resolve_dependencies(params)
    assert params[1].expr == "p1_amp + 1"

    # invalid
    params = [
        Parameter("p1_amp", expr=None),
        Parameter("p2_amp", expr="p3_amp + 1"),
    ]
    opt.resolve_dependencies(params)
    assert params[1].expr is None

    # real valid
    f_id = uuid4().hex
    s_id = uuid4().hex
    t_id = uuid4().hex
    params = [
        Parameter(f"p{f_id}_amp", expr=None),
        Parameter(f"p{s_id}_amp", expr=f"p{f_id}_amp + 1"),
        Parameter(f"p{t_id}_amp", expr=f"p{s_id}_amp + 1"),
    ]
    opt.resolve_dependencies(params)
    assert params[1].expr == f"p{f_id}_amp + 1"
    assert params[2].expr == f"p{s_id}_amp + 1"

    # real invalid
    f_id = uuid4().hex
    s_id = uuid4().hex
    t_id = uuid4().hex
    params = [
        Parameter(f"p{f_id}_amp", expr=None),
        Parameter(f"p{s_id}_amp", expr=f"p{t_id}_amp + 1"),
    ]
    opt.resolve_dependencies(params)
    assert params[1].expr is None


def test_get_peak_params(simple_peak):
    mgr = OptimizationManager()
    amp, cen, sig, frac = mgr.get_peak_params(simple_peak)
    assert amp.name == "amp"
    assert frac.name == "frac"


def test_peakparam_to_param(simple_param):
    opt = OptimizationManager()
    param = opt.peakparam_to_param("p123", simple_param)
    assert param.name == "p123_param"
    assert param.value == simple_param.value


def test_peak_to_params(simple_peak):
    opt = OptimizationManager()
    params = opt.peak_to_params(simple_peak)
    assert params[0].value == simple_peak.amp_par.value


def test_peak_to_params_with_norm(simple_peak):
    opt = OptimizationManager()
    params = opt.peak_to_params(simple_peak, norm_coefs=(0, 10))
    assert params[0].value == simple_peak.amp / 10
    assert params[1].value == simple_peak.cen


@pytest.mark.parametrize("peaks_per_region", [1, 2, 3], ids=["1", "2", "3"])
def test_prepare_region(simple_collection_factory, peaks_per_region):
    collection = simple_collection_factory(1, 1, peaks_per_region)
    opt = OptimizationManager()
    opt.set_collection(collection)
    example_region = tuple(collection.regions_index.keys())[0]
    x, y, reg_combination, params = opt.prepare_region(example_region)
    assert len(reg_combination) == peaks_per_region
    assert len(params) == 4 * peaks_per_region
