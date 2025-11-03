import pytest
import random

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from lib.spectra import SpectrumCollection, Spectrum, Peak
from lib.optimization import OptimizationManager
from lib.funcs import pvoigt


SPECTRUM_LEN = 200
SPECTRUM_START = 5
SPECTRUM_END = 35
PERTUBATION = 0.5
NOISE = 5
RTOL = 0.25

IGNOR_PERTUB_FOR = "frac"
LOW_PERTUBE_FOR = "cen"

CREATE_PEAK_PARAMS = dict(pertube=PERTUBATION, ignore_for=IGNOR_PERTUB_FOR, low_pertube_for=LOW_PERTUBE_FOR)

RUNS_NUM = pytest.mark.parametrize("run", range(3))


def get_xy():
    x = np.linspace(SPECTRUM_START, SPECTRUM_END, SPECTRUM_LEN, dtype=np.float32)
    y = np.zeros_like(x)
    return x, y


def add_param_perturb(
    true_params: dict[str, float],
    pertube: float | None = None,
    low_pertube_for: tuple[str] | str | None = None,
    ignore_for: tuple[str] | str | None = None,
) -> dict[str, float]:
    if ignore_for is None:
        ignore_for = ()
    if low_pertube_for is None:
        low_pertube_for = ()

    params = true_params.copy()

    if pertube is not None and pertube != 0:
        for k in params:
            if k in ignore_for:
                continue
            elif k in low_pertube_for:
                params[k] *= random.uniform(1 - pertube / 10, 1 + pertube / 10)
            else:
                params[k] *= random.uniform(1 - pertube, 1 + pertube)
    elif pertube == 0:
        for k in params:
            if k in ignore_for:
                continue
            else:
                params[k] = 0

    return params


def create_peak(
    x: NDArray,
    true_params: dict[str, float],
    pertube: float | None = None,
    low_pertube_for: tuple[str] | str | None = None,
    ignore_for: tuple[str] | str | None = None,
) -> tuple[NDArray, Peak, dict[str, float]]:
    amp = true_params["amp"]
    cen = true_params["cen"]
    sig = true_params["sig"]
    frac = true_params["frac"]

    y_peak = pvoigt(x, amp, cen, sig, frac)

    params = add_param_perturb(true_params, pertube, low_pertube_for, ignore_for)

    peak_obj = Peak(**params)

    return y_peak, peak_obj


def gen_peak(x, pertube=0.2):
    amp = random.uniform(100, 2000)
    cen = x[100] + random.uniform(-5, 5)
    sig = random.uniform(0.5, 3)
    frac = random.uniform(0.1, 0.9)

    true_params = {"amp": amp, "cen": cen, "sig": sig, "frac": frac}
    y_p, p_obj = create_peak(x, true_params, pertube=pertube, ignore_for="frac")

    return y_p, p_obj, true_params


def s1r1p1(x, true_params_d):
    y_p, p_obj, t_p_1 = gen_peak(x, pertube=0.2)
    true_params_d[p_obj.id] = t_p_1

    return y_p, p_obj


def add_bg(x, true_params_d: dict, bg_type="none"):
    if bg_type == "constant":
        y = random.uniform(100, 1000)
    elif bg_type == "linear":
        y = random.uniform(-0.01, 0.01) * x + random.uniform(100, 1000)
    elif bg_type == "shirley":
        y = np.zeros_like(x) + random.uniform(100, 1000)
        mult = random.uniform(0.02, 0.1)
        for true_params in true_params_d.values():
            back = (
                mult
                * true_params["amp"]
                * stats.norm(loc=true_params["cen"], scale=true_params["sig"]).cdf(x)
            )
            y += back
    else:
        y = 0

    return y


def add_noise(level=10):
    return np.random.normal(0, level, 200)


@pytest.fixture
def s1r1p1_constbg():
    x, y = get_xy()
    true_params_d = {}

    y_p, p_obj = s1r1p1(x, true_params_d)
    y += y_p

    y += add_bg(x, true_params_d, bg_type="constant")
    y += add_noise(level=NOISE)

    coll = SpectrumCollection()
    spec = Spectrum(x=x, y=y)
    coll.register(spec)
    reg = spec.create_region(0, len(x) - 1, background_type="linear")
    coll.add_link(spec, reg)
    coll.add_link(reg, p_obj)

    return coll, true_params_d


@pytest.fixture
def s1r1p1_linbg():
    x, y = get_xy()
    true_params_d = {}

    y_p, p_obj = s1r1p1(x, true_params_d)
    y += y_p

    y += add_bg(x, true_params_d, bg_type="linear")
    y += add_noise(level=NOISE)

    coll = SpectrumCollection()
    spec = Spectrum(x=x, y=y)
    coll.register(spec)
    reg = spec.create_region(0, len(x) - 1, background_type="linear")
    coll.add_link(spec, reg)
    coll.add_link(reg, p_obj)

    return coll, true_params_d


@pytest.fixture
def s1r1p1_shirleybg():
    x, y = get_xy()
    true_params_d = {}

    y_p, p_obj = s1r1p1(x, true_params_d)
    y += y_p

    y += add_bg(x, true_params_d, bg_type="shirley")
    y += add_noise(level=NOISE)

    coll = SpectrumCollection()
    spec = Spectrum(x=x, y=y)
    coll.register(spec)
    reg = spec.create_region(0, len(x) - 1, background_type="shirley")
    coll.add_link(spec, reg)
    coll.add_link(reg, p_obj)

    return coll, true_params_d


def run(coll):
    opt = OptimizationManager()
    opt.set_collection(coll)
    regs = list(coll.regions_index.keys())
    opt.proceed_regions_opt(regs)


def get_params(peak: Peak):
    return peak.amp, peak.cen, peak.sig, peak.frac


@pytest.mark.parametrize("_run", range(10))
def test_s1r1p1_constbg(s1r1p1_constbg, _run):
    coll, true_params_d = s1r1p1_constbg
    run(coll)

    for p_id, p in coll.peaks_index.items():
        p_true = tuple(true_params_d[p_id].values())[1:3]  # ignore amp and frac
        p_params = get_params(p)[1:3]

        assert np.allclose(p_true, p_params, rtol=RTOL)


@pytest.mark.parametrize("_run", range(10))
def test_s1r1p1_linbg(s1r1p1_linbg, _run):
    coll, true_params_d = s1r1p1_linbg
    run(coll)

    for p_id, p in coll.peaks_index.items():
        p_true = tuple(true_params_d[p_id].values())[1:3]  # ignore amp and frac
        p_params = get_params(p)[1:3]

        assert np.allclose(p_true, p_params, rtol=RTOL)


@pytest.mark.parametrize("_run", range(10))
def test_s1r1p1_shirleybg(s1r1p1_shirleybg, _run):
    coll, true_params_d = s1r1p1_shirleybg
    run(coll)

    for p_id, p in coll.peaks_index.items():
        p_true = tuple(true_params_d[p_id].values())[1:3]  # ignore amp and frac
        p_params = get_params(p)[1:3]

        assert np.allclose(p_true, p_params, rtol=RTOL)
