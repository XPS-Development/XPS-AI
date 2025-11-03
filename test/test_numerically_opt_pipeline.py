"""
Test numerically spectra optimization

NOTE: Some cases may fail due to random perturbations and high noise.
It's normal behavior. You may try to run test multiple times.
"""

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


def gen_params(x):
    amp = random.uniform(100, 2000)
    cen = x[SPECTRUM_LEN // 2] + random.uniform(-5, 5)
    sig = random.uniform(0.5, 3)
    frac = random.uniform(0.1, 0.9)
    true_params = {"amp": amp, "cen": cen, "sig": sig, "frac": frac}
    return true_params


def s1r1p1(x, true_params_d: dict):
    t_p_1 = gen_params(x)
    y_p, p_obj = create_peak(x, t_p_1, **CREATE_PEAK_PARAMS)
    true_params_d[p_obj.id] = t_p_1

    return y_p, p_obj


def s1r1p2(x, true_params_d: dict):
    t_p_1 = gen_params(x)
    y_p_1, p_obj1 = create_peak(x, t_p_1, **CREATE_PEAK_PARAMS)
    true_params_d[p_obj1.id] = t_p_1

    t_p_2 = t_p_1.copy()
    t_p_2["amp"] /= 2
    t_p_2["cen"] += 5

    y_p_2, p_obj2 = create_peak(x, t_p_2, **CREATE_PEAK_PARAMS)
    true_params_d[p_obj2.id] = t_p_2
    p_obj2.set(name="cen", expr=f"{p_obj1.id} + 5")

    y_p = y_p_1 + y_p_2
    p_objs = (p_obj1, p_obj2)
    return y_p, p_objs


def s1r2p1(x, true_params_d: dict):
    t_p_1 = gen_params(x)
    t_p_1["cen"] = x[SPECTRUM_LEN // 4]
    y_p_1, p_obj1 = create_peak(x, t_p_1, **CREATE_PEAK_PARAMS)
    true_params_d[p_obj1.id] = t_p_1

    t_p_2 = t_p_1.copy()
    t_p_2["amp"] /= 2
    t_p_2["cen"] += 15
    y_p_2, p_obj2 = create_peak(x, t_p_2, **CREATE_PEAK_PARAMS)
    p_obj2.set(name="cen", expr=f"{p_obj1.id} + 15")
    true_params_d[p_obj2.id] = t_p_2

    y_p = y_p_1 + y_p_2
    p_objs = (p_obj1, p_obj2)
    return y_p, p_objs


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
    reg = spec.create_region(0, len(x), background_type="linear")
    coll.add_link(spec, reg)
    coll.add_link(reg, p_obj)

    return coll, true_params_d


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
    reg = spec.create_region(0, len(x), background_type="linear")
    coll.add_link(spec, reg)
    coll.add_link(reg, p_obj)

    return coll, true_params_d


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
    reg = spec.create_region(0, len(x), background_type="shirley")
    coll.add_link(spec, reg)
    coll.add_link(reg, p_obj)

    return coll, true_params_d


def s1r1p2_linbg():
    x, y = get_xy()
    true_params_d = {}

    y_p, p_objs = s1r1p2(x, true_params_d)
    y += y_p

    y += add_bg(x, true_params_d, bg_type="linear")
    y += add_noise(level=NOISE)

    coll = SpectrumCollection()
    spec = Spectrum(x=x, y=y)
    coll.register(spec)
    reg = spec.create_region(0, len(x), background_type="linear")
    coll.add_link(spec, reg)
    for p_obj in p_objs:
        coll.add_link(reg, p_obj)

    return coll, true_params_d


def s1r1p2_shirleybg():
    x, y = get_xy()
    true_params_d = {}

    y_p, p_objs = s1r1p2(x, true_params_d)
    y += y_p

    y += add_bg(x, true_params_d, bg_type="shirley")
    y += add_noise(level=NOISE)

    coll = SpectrumCollection()
    spec = Spectrum(x=x, y=y)
    coll.register(spec)
    reg = spec.create_region(0, len(x), background_type="shirley")
    coll.add_link(spec, reg)
    for p_obj in p_objs:
        coll.add_link(reg, p_obj)

    return coll, true_params_d


def s1r2p1_linbg():
    x, y = get_xy()
    true_params_d = {}

    y_p, p_objs = s1r2p1(x, true_params_d)
    y += y_p

    y += add_bg(x, true_params_d, bg_type="linear")
    y += add_noise(level=NOISE)

    coll = SpectrumCollection()
    spec = Spectrum(x=x, y=y)
    coll.register(spec)
    reg_idxs = ((0, len(x) // 2), (len(x) // 2, len(x)))
    for p_obj, (s_i, e_i) in zip(p_objs, reg_idxs):
        reg = spec.create_region(s_i, e_i, background_type="linear")
        coll.add_link(spec, reg)
        coll.add_link(reg, p_obj)

    return coll, true_params_d


def s1r2p1_shirleybg():
    x, y = get_xy()
    true_params_d = {}

    y_p, p_objs = s1r2p1(x, true_params_d)
    y += y_p

    y += add_bg(x, true_params_d, bg_type="shirley")
    y += add_noise(level=NOISE)

    coll = SpectrumCollection()
    spec = Spectrum(x=x, y=y)
    coll.register(spec)
    reg_idxs = ((0, len(x) // 2), (len(x) // 2, len(x)))
    for p_obj, (s_i, e_i) in zip(p_objs, reg_idxs):
        reg = spec.create_region(s_i, e_i, background_type="shirley")
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


def construct_case(type_: str, bg: str):
    f = globals()[f"{type_}_{bg}bg"]
    coll, true_params_d = f()
    return coll, true_params_d


def run_test(coll, true_params_d):
    run(coll)

    for p_id, p in coll.peaks_index.items():
        p_true = tuple(true_params_d[p_id].values())[1:3]  # ignore amp and frac
        p_params = get_params(p)[1:3]

        assert np.allclose(p_true, p_params, rtol=RTOL)


@RUNS_NUM
@pytest.mark.parametrize("bg", ("const", "lin", "shirley"))
def test_default(bg, run):
    coll, true_params_d = construct_case(type_="s1r1p1", bg=bg)
    run_test(coll, true_params_d)


@RUNS_NUM
@pytest.mark.parametrize("bg", ("lin", "shirley"))
@pytest.mark.parametrize("type_", ("s1r1p2", "s1r2p1"))
def test_with_links(type_, bg, run):
    coll, true_params_d = construct_case(type_=type_, bg=bg)
    run_test(coll, true_params_d)
