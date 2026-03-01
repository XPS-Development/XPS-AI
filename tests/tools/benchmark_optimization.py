"""
Benchmarks for tools.optimization: quality (parameter recovery) and latency.

Uses pvoigt from core.math_models.model_funcs, PseudoVoigtPeakModel,
and the same component creation pattern as test_optimization.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.dto import ParameterDTO, ComponentDTO
from core.math_models import PseudoVoigtPeakModel
from core.math_models.model_funcs import pvoigt

from tools.optimization import OptimizationContext, optimize

RNG = np.random.default_rng(42)
X_AXIS_START: float = -10.0
X_AXIS_STOP: float = 10.0
X_AXIS_NUM_POINTS: int = 200

# Representative parameter sets for 1-3 peaks
TRUE_PARAMS_1PEAK: list[dict[str, float]] = [
    {"amp": 1.0, "cen": 0.0, "sig": 1.0, "frac": 0.5},
]
TRUE_PARAMS_2PEAKS: list[dict[str, float]] = [
    {"amp": 1.0, "cen": -2.0, "sig": 0.8, "frac": 0.3},
    {"amp": 0.8, "cen": 2.0, "sig": 1.2, "frac": 0.7},
]
TRUE_PARAMS_3PEAKS: list[dict[str, float]] = [
    {"amp": 1.0, "cen": -3.0, "sig": 0.9, "frac": 0.4},
    {"amp": 0.9, "cen": 0.0, "sig": 1.0, "frac": 0.5},
    {"amp": 0.7, "cen": 3.0, "sig": 1.1, "frac": 0.6},
]


def generate_pvoigt_spectrum(
    x: np.ndarray,
    peak_params: list[dict[str, float]],
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate synthetic spectrum as sum of pvoigt peaks plus Gaussian noise.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-values (>= 200 points).
    peak_params : list[dict[str, float]]
        List of parameter dicts per peak (amp, cen, sig, frac).
    noise_scale : float
        Standard deviation of additive Gaussian noise.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        y = sum(pvoigt(...)) + noise.
    """
    y = np.zeros_like(x, dtype=float)
    for params in peak_params:
        y += pvoigt(
            x,
            params["amp"],
            params["cen"],
            params["sig"],
            params["frac"],
        )
    noise = rng.normal(loc=0, scale=noise_scale, size=len(x))
    return y + noise


def perturb_params(
    true_params: list[dict[str, float]],
    scale: float,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """
    Perturb parameters for use as initial guess.

    Parameters
    ----------
    true_params : list[dict[str, float]]
        True parameter dicts per peak.
    scale : float
        Perturbation scale (e.g. 0.1 for ~10%).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    list[dict[str, float]]
        Perturbed parameter dicts.
    """
    result: list[dict[str, float]] = []
    for params in true_params:
        perturbed: dict[str, float] = {}
        for name, val in params.items():
            delta = scale * rng.uniform(-1, 1)
            pval = val * (1 + delta)
            if name == "amp":
                pval = max(0, pval)
            elif name == "frac":
                pval = max(0, min(1, pval))
            elif name == "sig":
                pval = max(1e-6, pval)
            perturbed[name] = pval
        result.append(perturbed)
    return result


def _make_component(
    comp_id: str,
    parent_id: str,
    params: dict[str, float],
    *,
    amp_expr: str | None = None,
) -> ComponentDTO:
    """Create a minimal ComponentDTO for benchmarking."""
    param_dtos: dict[str, ParameterDTO] = {}
    for name, val in params.items():
        param_dtos[name] = ParameterDTO(
            name=name,
            value=val,
            lower=-np.inf if name != "amp" else 0,
            upper=np.inf if name != "frac" else 1,
            vary=True,
            expr=amp_expr if name == "amp" else None,
        )
    return ComponentDTO(
        id_=comp_id,
        parent_id=parent_id,
        normalized=False,
        parameters=param_dtos,
        model=PseudoVoigtPeakModel(),
        kind="peak",
    )


def _compute_param_rmse(
    optimized: dict[str, dict[str, float]],
    true_params: list[dict[str, float]],
    param_names: tuple[str, ...] = ("amp", "cen", "sig", "frac"),
) -> dict[str, float]:
    """Compute RMSE per parameter across peaks."""
    n_peaks = len(true_params)
    sq_errors: dict[str, list[float]] = {p: [] for p in param_names}
    for i, true in enumerate(true_params):
        comp_id = f"p{i + 1}"
        opt = optimized.get(comp_id, {})
        for p in param_names:
            if p in true and p in opt:
                sq_errors[p].append((opt[p] - true[p]) ** 2)
    rmse: dict[str, float] = {}
    for p in param_names:
        vals = sq_errors[p]
        rmse[p] = float(np.sqrt(np.mean(vals))) if vals else 0.0
    return rmse


def _build_contexts_from_peaks(
    x: np.ndarray,
    y: np.ndarray,
    peak_params: list[dict[str, float]],
    region_id: str = "r1",
    parent_id: str = "s1",
    amp_exprs: list[str | None] | None = None,
    id_prefix: str = "",
) -> tuple[OptimizationContext, ...]:
    """Build OptimizationContext(s) from peak parameters."""
    if amp_exprs is None:
        amp_exprs = [None] * len(peak_params)
    components = []
    for i, params in enumerate(peak_params):
        comp_id = f"{id_prefix}p{i + 1}"
        expr = amp_exprs[i] if i < len(amp_exprs) else None
        components.append(_make_component(comp_id, region_id, params, amp_expr=expr))
    ctx = OptimizationContext(
        id_=region_id,
        parent_id=parent_id,
        normalized=False,
        x=x,
        y=y,
        components=tuple(components),
    )
    return (ctx,)


# ---------------------------------------------------------------------------
# Quality benchmarks
# ---------------------------------------------------------------------------


class TestBenchmarkQuality:
    """Quality benchmarks: parameter recovery (RMSE)."""

    def test_benchmark_quality_1peak(self):
        """Single peak, ~10% perturbation, low noise."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        true = TRUE_PARAMS_1PEAK
        y = generate_pvoigt_spectrum(x, true, noise_scale=0.01, rng=RNG)
        init = perturb_params(true, 0.1, RNG)
        (ctx,) = _build_contexts_from_peaks(x, y, init)
        result = optimize([ctx], method="least_squares")
        opt_dict = {o.component_id: o.parameters for o in result}
        rmse = _compute_param_rmse(opt_dict, true)
        assert rmse["amp"] < 0.15 * true[0]["amp"]
        assert rmse["cen"] < 0.15 * max(abs(true[0]["cen"]), 1)
        assert rmse["sig"] < 0.15 * true[0]["sig"]

    def test_benchmark_quality_2peaks(self):
        """Two peaks, ~10% perturbation, low noise."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        true = TRUE_PARAMS_2PEAKS
        y = generate_pvoigt_spectrum(x, true, noise_scale=0.01, rng=RNG)
        init = perturb_params(true, 0.1, RNG)
        (ctx,) = _build_contexts_from_peaks(x, y, init)
        result = optimize([ctx], method="least_squares")
        opt_dict = {o.component_id: o.parameters for o in result}
        rmse = _compute_param_rmse(opt_dict, true)
        for p in ("amp", "cen", "sig"):
            max_true = max(t[p] for t in true) if p != "cen" else 1
            assert rmse[p] < 0.2 * max_true

    def test_benchmark_quality_3peaks(self):
        """Three peaks, ~10% perturbation, low noise."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        true = TRUE_PARAMS_3PEAKS
        y = generate_pvoigt_spectrum(x, true, noise_scale=0.01, rng=RNG)
        init = perturb_params(true, 0.1, RNG)
        (ctx,) = _build_contexts_from_peaks(x, y, init)
        result = optimize([ctx], method="least_squares")
        opt_dict = {o.component_id: o.parameters for o in result}
        rmse = _compute_param_rmse(opt_dict, true)
        for p in ("amp", "cen", "sig"):
            max_true = max(t[p] for t in true) if p != "cen" else 1
            assert rmse[p] < 0.25 * max_true

    def test_benchmark_quality_perturbed_20pct(self):
        """Single peak, ~20% perturbation, harder initial guess."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        true = TRUE_PARAMS_1PEAK
        y = generate_pvoigt_spectrum(x, true, noise_scale=0.01, rng=RNG)
        init = perturb_params(true, 0.2, RNG)
        (ctx,) = _build_contexts_from_peaks(x, y, init)
        result = optimize([ctx], method="least_squares")
        opt_dict = {o.component_id: o.parameters for o in result}
        rmse = _compute_param_rmse(opt_dict, true)
        assert rmse["amp"] < 0.2 * true[0]["amp"]
        assert rmse["cen"] < 0.2 * max(abs(true[0]["cen"]), 1)
        assert rmse["sig"] < 0.2 * true[0]["sig"]

    def test_benchmark_quality_noisy(self):
        """Single peak, ~10% perturbation, higher noise."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        true = TRUE_PARAMS_1PEAK
        y = generate_pvoigt_spectrum(x, true, noise_scale=0.05, rng=RNG)
        init = perturb_params(true, 0.1, RNG)
        (ctx,) = _build_contexts_from_peaks(x, y, init)
        result = optimize([ctx], method="least_squares")
        opt_dict = {o.component_id: o.parameters for o in result}
        rmse = _compute_param_rmse(opt_dict, true)
        assert rmse["amp"] < 0.25 * true[0]["amp"]
        assert rmse["cen"] < 0.25 * max(abs(true[0]["cen"]), 1)
        assert rmse["sig"] < 0.25 * true[0]["sig"]


# ---------------------------------------------------------------------------
# Expression-coupled benchmarks
# ---------------------------------------------------------------------------


class TestBenchmarkQualityExpr:
    """Quality benchmarks with expression-coupled components."""

    def test_benchmark_quality_expr_same_spectrum(self):
        """Two peaks in one region, p2.amp = 2 * p1."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        true_p1 = {"amp": 1.0, "cen": -2.0, "sig": 0.8, "frac": 0.3}
        true_p2 = {"amp": 2.0, "cen": 2.0, "sig": 1.0, "frac": 0.5}
        true = [true_p1, true_p2]
        y = generate_pvoigt_spectrum(x, true, noise_scale=0.01, rng=RNG)
        init_p1 = perturb_params([true_p1], 0.1, RNG)[0]
        init_p2 = perturb_params([true_p2], 0.1, RNG)[0]
        cmp1 = _make_component("p1", "r1", init_p1)
        cmp2 = _make_component("p2", "r1", init_p2, amp_expr="2 * p1")
        ctx = OptimizationContext("r1", "s1", False, x, y, (cmp1, cmp2))
        result = optimize([ctx], method="least_squares")
        opt_dict = {o.component_id: o.parameters for o in result}
        amp1, amp2 = opt_dict["p1"]["amp"], opt_dict["p2"]["amp"]
        assert np.isclose(amp2, 2 * amp1, rtol=0.1)
        assert np.isclose(opt_dict["p1"]["amp"], true_p1["amp"], rtol=0.2)

    def test_benchmark_quality_expr_different_spectra(self):
        """Two regions (different x/y), p2.amp = 2 * p1, grouped together."""
        x1 = np.linspace(-8, 0, 200)
        x2 = np.linspace(0, 8, 200)
        true_p1 = {"amp": 1.0, "cen": -2.0, "sig": 0.8, "frac": 0.3}
        true_p2 = {"amp": 2.0, "cen": 2.0, "sig": 1.0, "frac": 0.5}
        y1 = generate_pvoigt_spectrum(x1, [true_p1], noise_scale=0.01, rng=RNG)
        y2 = generate_pvoigt_spectrum(x2, [true_p2], noise_scale=0.01, rng=RNG)
        init_p1 = perturb_params([true_p1], 0.1, RNG)[0]
        init_p2 = perturb_params([true_p2], 0.1, RNG)[0]
        cmp1 = _make_component("p1", "r1", init_p1)
        cmp2 = _make_component("p2", "r2", init_p2, amp_expr="2 * p1")
        ctx1 = OptimizationContext("r1", "s1", False, x1, y1, (cmp1,))
        ctx2 = OptimizationContext("r2", "s1", False, x2, y2, (cmp2,))
        result = optimize([ctx1, ctx2], method="least_squares")
        opt_dict = {o.component_id: o.parameters for o in result}
        amp1, amp2 = opt_dict["p1"]["amp"], opt_dict["p2"]["amp"]
        assert np.isclose(amp2, 2 * amp1, rtol=0.15)
        assert np.isfinite(amp1) and np.isfinite(amp2)


# ---------------------------------------------------------------------------
# Latency benchmarks (pytest-benchmark)
# ---------------------------------------------------------------------------


class TestBenchmarkLatency:
    """Latency benchmarks using pytest-benchmark."""

    def test_benchmark_latency_1region_1peak(self, benchmark):
        """Baseline: 1 region, 1 peak."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        true = TRUE_PARAMS_1PEAK
        y = generate_pvoigt_spectrum(x, true, noise_scale=0.01, rng=RNG)
        init = perturb_params(true, 0.1, RNG)
        (ctx,) = _build_contexts_from_peaks(x, y, init)

        def run():
            return optimize([ctx], method="least_squares")

        result = benchmark(run)
        assert len(result) >= 1
        for opt in result:
            for v in opt.parameters.values():
                assert np.isfinite(v)

    def test_benchmark_latency_1region_3peaks(self, benchmark):
        """1 region, 3 peaks."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        true = TRUE_PARAMS_3PEAKS
        y = generate_pvoigt_spectrum(x, true, noise_scale=0.01, rng=RNG)
        init = perturb_params(true, 0.1, RNG)
        (ctx,) = _build_contexts_from_peaks(x, y, init)

        def run():
            return optimize([ctx], method="least_squares")

        result = benchmark(run)
        assert len(result) == 3

    def test_benchmark_latency_5regions(self, benchmark):
        """5 regions, 1-2 peaks each."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        contexts: list[OptimizationContext] = []
        for i in range(5):
            n_peaks = 2 if i % 2 == 0 else 1
            true = TRUE_PARAMS_2PEAKS[:n_peaks] if n_peaks == 2 else TRUE_PARAMS_1PEAK
            y = generate_pvoigt_spectrum(x, true, noise_scale=0.01, rng=RNG)
            init = perturb_params(true, 0.1, RNG)
            (ctx,) = _build_contexts_from_peaks(
                x, y, init, region_id=f"r{i}", parent_id="s1", id_prefix=f"r{i}_"
            )
            contexts.append(ctx)

        def run():
            return optimize(contexts, method="least_squares")

        result = benchmark(run)
        assert len(result) >= 5

    def test_benchmark_latency_20regions(self, benchmark):
        """20 regions, 1-2 peaks each."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        contexts: list[OptimizationContext] = []
        for i in range(20):
            n_peaks = 2 if i % 2 == 0 else 1
            true = TRUE_PARAMS_2PEAKS[:n_peaks] if n_peaks == 2 else TRUE_PARAMS_1PEAK
            y = generate_pvoigt_spectrum(x, true, noise_scale=0.01, rng=RNG)
            init = perturb_params(true, 0.1, RNG)
            (ctx,) = _build_contexts_from_peaks(
                x, y, init, region_id=f"r{i}", parent_id="s1", id_prefix=f"r{i}_"
            )
            contexts.append(ctx)

        def run():
            return optimize(contexts, method="least_squares")

        result = benchmark(run)
        assert len(result) >= 20

    def test_benchmark_latency_100regions(self, benchmark):
        """100 regions, 1-2 peaks each (stress)."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        contexts: list[OptimizationContext] = []
        for i in range(100):
            n_peaks = 2 if i % 2 == 0 else 1
            true = TRUE_PARAMS_2PEAKS[:n_peaks] if n_peaks == 2 else TRUE_PARAMS_1PEAK
            y = generate_pvoigt_spectrum(x, true, noise_scale=0.01, rng=RNG)
            init = perturb_params(true, 0.1, RNG)
            (ctx,) = _build_contexts_from_peaks(
                x, y, init, region_id=f"r{i}", parent_id="s1", id_prefix=f"r{i}_"
            )
            contexts.append(ctx)

        def run():
            return optimize(contexts, method="least_squares")

        result = benchmark(run)
        assert len(result) >= 100

    def test_benchmark_latency_expr_coupled(self, benchmark):
        """Latency with expression-coupled components (same spectrum)."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        true_p1 = {"amp": 1.0, "cen": -2.0, "sig": 0.8, "frac": 0.3}
        true_p2 = {"amp": 2.0, "cen": 2.0, "sig": 1.0, "frac": 0.5}
        true = [true_p1, true_p2]
        y = generate_pvoigt_spectrum(x, true, noise_scale=0.01, rng=RNG)
        init_p1 = perturb_params([true_p1], 0.1, RNG)[0]
        init_p2 = perturb_params([true_p2], 0.1, RNG)[0]
        cmp1 = _make_component("p1", "r1", init_p1)
        cmp2 = _make_component("p2", "r1", init_p2, amp_expr="2 * p1")
        ctx = OptimizationContext("r1", "s1", False, x, y, (cmp1, cmp2))

        def run():
            return optimize([ctx], method="least_squares")

        result = benchmark(run)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Stress tests (5 / 20 / 100 regions)
# ---------------------------------------------------------------------------


class TestBenchmarkStress:
    """Stress tests: many regions, verify finite params and measure latency."""

    def test_benchmark_stress_5_regions(self, benchmark):
        """5 regions, 1-2 peaks each."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        contexts: list[OptimizationContext] = []
        for i in range(5):
            n_peaks = 2 if i % 2 == 0 else 1
            true = TRUE_PARAMS_2PEAKS[:n_peaks] if n_peaks == 2 else TRUE_PARAMS_1PEAK
            y = generate_pvoigt_spectrum(x, true, noise_scale=0.01, rng=RNG)
            init = perturb_params(true, 0.1, RNG)
            (ctx,) = _build_contexts_from_peaks(
                x, y, init, region_id=f"r{i}", parent_id="s1", id_prefix=f"r{i}_"
            )
            contexts.append(ctx)

        def run():
            return optimize(contexts, method="least_squares")

        result = benchmark(run)
        for opt in result:
            for name, val in opt.parameters.items():
                assert np.isfinite(val), f"{opt.component_id}.{name} = {val}"

    def test_benchmark_stress_20_regions(self, benchmark):
        """20 regions, 1-2 peaks each."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        contexts: list[OptimizationContext] = []
        for i in range(20):
            n_peaks = 2 if i % 2 == 0 else 1
            true = TRUE_PARAMS_2PEAKS[:n_peaks] if n_peaks == 2 else TRUE_PARAMS_1PEAK
            y = generate_pvoigt_spectrum(x, true, noise_scale=0.01, rng=RNG)
            init = perturb_params(true, 0.1, RNG)
            (ctx,) = _build_contexts_from_peaks(
                x, y, init, region_id=f"r{i}", parent_id="s1", id_prefix=f"r{i}_"
            )
            contexts.append(ctx)

        def run():
            return optimize(contexts, method="least_squares")

        result = benchmark(run)
        for opt in result:
            for name, val in opt.parameters.items():
                assert np.isfinite(val), f"{opt.component_id}.{name} = {val}"

    def test_benchmark_stress_100_regions(self, benchmark):
        """100 regions, 1-2 peaks each."""
        x = np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)
        contexts: list[OptimizationContext] = []
        for i in range(100):
            n_peaks = 2 if i % 2 == 0 else 1
            true = TRUE_PARAMS_2PEAKS[:n_peaks] if n_peaks == 2 else TRUE_PARAMS_1PEAK
            y = generate_pvoigt_spectrum(x, true, noise_scale=0.01, rng=RNG)
            init = perturb_params(true, 0.1, RNG)
            (ctx,) = _build_contexts_from_peaks(
                x, y, init, region_id=f"r{i}", parent_id="s1", id_prefix=f"r{i}_"
            )
            contexts.append(ctx)

        def run():
            return optimize(contexts, method="least_squares")

        result = benchmark(run)
        for opt in result:
            for name, val in opt.parameters.items():
                assert np.isfinite(val), f"{opt.component_id}.{name} = {val}"
