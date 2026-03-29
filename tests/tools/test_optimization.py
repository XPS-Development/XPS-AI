"""
Tests for tools.optimization: build_contexts, OptimizationPlanner, LmfitOptimizer, optimize().
"""

from types import SimpleNamespace

import numpy as np
import pytest

import tools.optimization as optimization_module
from core.math_models import PseudoVoigtPeakModel
from tools.dto import ComponentDTO, ParameterDTO
from tools.optimization import (
    LmfitOptimizer,
    OptimizationContext,
    OptimizationPlanner,
    OptimizedComponent,
    build_contexts,
    optimize,
    resolve_component_reference,
)


@pytest.fixture
def optimization_context(dto_service, region_id):
    """Build OptimizationContext from simple_collection region via build_contexts."""
    reg_dto, comp_dtos = dto_service.get_region_repr(region_id, normalized=False)
    contexts = build_contexts([(reg_dto, comp_dtos)])
    return contexts[0]


class TestResolveComponentReference:
    """Tests for resolve_component_reference (short id → full id)."""

    def test_exact_match(self):
        ids = ("pfxAAA00xx", "pfxBBB00xx")
        assert resolve_component_reference("pfxAAA00xx", ids) == "pfxAAA00xx"

    def test_unique_prefix_under_19_chars(self):
        ids = ("pfxAAA00xx", "pfxBBB00xx")
        assert resolve_component_reference("pfxAA", ids) == "pfxAAA00xx"
        assert resolve_component_reference("pfxBB", ids) == "pfxBBB00xx"
        assert resolve_component_reference("pfx", ids) is None  # ambiguous: both start with pfx
        assert resolve_component_reference("pfx", ("pfxAAA00xx",)) == "pfxAAA00xx"

    def test_filter_wrong_tokens(self):
        full = "p" + "a" * 32
        ids = (full,)
        wrong = "q" + "a" * 32
        assert resolve_component_reference(wrong, ids) is None

    def test_ambiguous_prefix_returns_none(self):
        ids = ("ab12345xx", "ab12345yy")
        assert resolve_component_reference("ab12345", ids) is None


class TestBuildContexts:
    """Tests for build_contexts."""

    def test_one_region_yields_one_context(self, dto_service, region_id):
        """Single region repr yields one OptimizationContext."""
        region_reprs = [dto_service.get_region_repr(region_id, normalized=False)]
        contexts = build_contexts(region_reprs)
        assert len(contexts) == 1
        ctx = contexts[0]
        assert ctx.id_ == region_reprs[0][0].id_
        assert ctx.x is region_reprs[0][0].x
        assert len(ctx.components) >= 1

    def test_fully_fixed_component_subtracted_and_excluded(self, dto_service, region_id):
        """Components with all params inactive (vary=False) are subtracted from y and excluded."""
        reg_dto, comp_dtos = dto_service.get_region_repr(region_id, normalized=False)
        contexts = build_contexts([(reg_dto, comp_dtos)])
        ctx = contexts[0]
        # simple_collection: constant background has only const with vary=False → fixed
        assert len(ctx.components) == 1
        assert ctx.components[0].kind == "peak"

    def test_multiple_regions_yield_multiple_contexts(self, dto_service, region_id):
        """Multiple region reprs yield one context per region."""
        reg_dto, comp_dtos = dto_service.get_region_repr(region_id, normalized=False)
        region_reprs = [(reg_dto, comp_dtos), (reg_dto, comp_dtos)]
        contexts = build_contexts(region_reprs)
        assert len(contexts) == 2
        assert contexts[0].id_ == contexts[1].id_
        assert len(contexts[0].components) == len(contexts[1].components)

    def test_all_inactive_subtracted_even_with_expr_on_params(self):
        """All vary=False subtracts from y and excludes the component; expr does not block that."""
        x = np.linspace(0, 10, 30)
        y = np.zeros_like(x)
        leader = _make_component("lead123456", "r1", {"amp": 1.0, "cen": 5.0, "sig": 1.0, "frac": 0.0})
        slave_params = {
            "amp": ParameterDTO(
                name="amp",
                value=1.0,
                lower=0.0,
                upper=np.inf,
                vary=False,
                expr="lead123456 * 0.5",
            ),
            "cen": ParameterDTO(name="cen", value=5.0, lower=-np.inf, upper=np.inf, vary=False, expr=None),
            "sig": ParameterDTO(name="sig", value=1.0, lower=0.0, upper=np.inf, vary=False, expr=None),
            "frac": ParameterDTO(name="frac", value=0.0, lower=0.0, upper=1.0, vary=False, expr=None),
        }
        slave = ComponentDTO(
            id_="slave12345",
            parent_id="r1",
            normalized=False,
            parameters=slave_params,
            model=PseudoVoigtPeakModel(),
            kind="peak",
        )
        reg = SimpleNamespace(id_="r1", parent_id="s1", normalized=False, x=x, y=y)
        contexts = build_contexts([(reg, (leader, slave))])
        assert len(contexts[0].components) == 1
        assert contexts[0].components[0].id_ == "lead123456"


def _make_component(
    comp_id: str,
    parent_id: str,
    params: dict[str, float],
    *,
    amp_expr: str | None = None,
) -> ComponentDTO:
    """Create a minimal ComponentDTO for testing."""
    param_dtos = {}
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


class TestOptimizationPlanner:
    """Tests for OptimizationPlanner.get_groups."""

    def test_single_context_single_component_returns_one_group(self, optimization_context):
        """Single context with one component yields one group."""
        planner = OptimizationPlanner()
        groups = planner.get_groups((optimization_context,))
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert groups[0][0] is optimization_context

    def test_two_independent_contexts_returns_two_groups(self):
        """Two contexts with no parameter dependencies yield two groups."""
        x = np.linspace(0, 10, 50)
        y = np.zeros_like(x)

        cmp1 = _make_component("p1", "r1", {"amp": 1, "cen": 5, "sig": 1, "frac": 0})
        cmp2 = _make_component("p2", "r2", {"amp": 1, "cen": 5, "sig": 1, "frac": 0})

        ctx1 = OptimizationContext("r1", "s1", False, x, y, (cmp1,))
        ctx2 = OptimizationContext("r2", "s1", False, x, y, (cmp2,))

        planner = OptimizationPlanner()
        groups = planner.get_groups((ctx1, ctx2))
        assert len(groups) == 2
        assert (ctx1,) in groups
        assert (ctx2,) in groups

    def test_components_with_expr_dependency_grouped_together(self):
        """Contexts sharing components via expr are grouped together."""
        x = np.linspace(0, 10, 50)
        y = np.zeros_like(x)

        cmp1 = _make_component("p1", "r1", {"amp": 1, "cen": 5, "sig": 1, "frac": 0})
        cmp2 = _make_component("p2", "r1", {"amp": 1, "cen": 5, "sig": 1, "frac": 0}, amp_expr="2 * p1")

        ctx = OptimizationContext("r1", "s1", False, x, y, (cmp1, cmp2))

        planner = OptimizationPlanner()
        groups = planner.get_groups((ctx,))
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert groups[0][0] is ctx

    def test_planner_resolves_short_component_ref_in_expr(self):
        """Short prefix in expr links to the unique full component id for grouping."""
        x = np.linspace(0, 10, 50)
        y = np.zeros_like(x)

        cmp1 = _make_component("peakAA01xx", "r1", {"amp": 1, "cen": 5, "sig": 1, "frac": 0})
        cmp2 = _make_component(
            "peakBB01xx",
            "r1",
            {"amp": 1, "cen": 5, "sig": 1, "frac": 0},
            amp_expr="2 * peakAA",
        )

        ctx = OptimizationContext("r1", "s1", False, x, y, (cmp1, cmp2))

        planner = OptimizationPlanner()
        groups = planner.get_groups((ctx,))
        assert len(groups) == 1
        assert groups[0][0] is ctx


class TestLmfitOptimizer:
    """Tests for LmfitOptimizer."""

    def test_optimize_returns_optimized_component(self, optimization_context):
        """Optimizer returns OptimizedComponent with correct structure."""
        optimizer = LmfitOptimizer()
        result = optimizer.optimize((optimization_context,), method="least_squares")

        assert len(result) >= 1
        for opt in result:
            assert isinstance(opt, OptimizedComponent)
            assert isinstance(opt.component_id, str)
            assert isinstance(opt.parameters, dict)
            for k, v in opt.parameters.items():
                assert isinstance(k, str)
                assert isinstance(v, (int, float))
                assert np.isfinite(v)

    def test_optimize_produces_finite_parameters(self, optimization_context):
        """Optimization produces finite parameter values."""
        optimizer = LmfitOptimizer()
        result = optimizer.optimize((optimization_context,), method="least_squares")

        for opt in result:
            for name, val in opt.parameters.items():
                assert np.isfinite(val), f"{opt.component_id}.{name} = {val}"

    def test_optimize_with_short_component_ref_in_expr(self):
        """Expressions may use a short prefix (len < 19) of the referenced component id."""
        x = np.linspace(-5, 5, 100)
        y = 0.8 * np.exp(-(x**2) / 2)

        cmp1 = _make_component("peakAA01xx", "r1", {"amp": 0.8, "cen": 0.0, "sig": 1.0, "frac": 0.0})
        cmp2 = _make_component(
            "peakBB01xx",
            "r1",
            {"amp": 0.2, "cen": 0.0, "sig": 1.0, "frac": 0.0},
            amp_expr="peakAA * 0.25",
        )
        ctx = OptimizationContext("r1", "s1", False, x, y, (cmp1, cmp2))

        optimizer = LmfitOptimizer()
        result = optimizer.optimize((ctx,), method="least_squares")
        assert len(result) == 2
        by_id = {r.component_id: r for r in result}
        assert np.isclose(
            by_id["peakBB01xx"].parameters["amp"], by_id["peakAA01xx"].parameters["amp"] * 0.25
        )


class TestOptimize:
    """Tests for optimize() library entry point."""

    def test_optimize_returns_tuple_of_optimized_components(self, optimization_context):
        """Entry point returns tuple of OptimizedComponent."""
        result = optimize([optimization_context], method="least_squares")

        assert isinstance(result, tuple)
        assert len(result) >= 1
        for opt in result:
            assert isinstance(opt, OptimizedComponent)
            assert opt.component_id
            assert opt.parameters

    def test_optimize_accepts_multiple_contexts(self, optimization_context):
        """Entry point handles multiple contexts (multiple groups)."""
        result = optimize([optimization_context, optimization_context], method="least_squares")
        assert len(result) >= 1

    def test_optimize_builds_expression_plan_once(self, optimization_context, monkeypatch):
        """Library entry point resolves expressions in a single pass over contexts."""
        count = {"n": 0}
        real_build = optimization_module._build_expression_plan

        def counting_plan(
            ctx: tuple[OptimizationContext, ...],
        ) -> optimization_module.OptimizationExpressionPlan:
            count["n"] += 1
            return real_build(ctx)

        monkeypatch.setattr(optimization_module, "_build_expression_plan", counting_plan)
        optimize([optimization_context], method="least_squares")
        assert count["n"] == 1
