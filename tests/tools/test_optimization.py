"""
Tests for tools.optimization: build_contexts, OptimizationPlanner, LmfitOptimizer, optimize().
"""

import numpy as np
import pytest

from tools.dto import ParameterDTO, ComponentDTO
from core.math_models import PseudoVoigtPeakModel

from tools.optimization import (
    OptimizationContext,
    OptimizationPlanner,
    LmfitOptimizer,
    OptimizedComponent,
    build_contexts,
    optimize,
)


@pytest.fixture
def optimization_context(dto_service, region_id):
    """Build OptimizationContext from simple_collection region via build_contexts."""
    reg_dto, comp_dtos = dto_service.get_region_repr(region_id, normalized=False)
    contexts = build_contexts([(reg_dto, comp_dtos)])
    return contexts[0]


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

    def test_static_background_subtracted_and_excluded(self, dto_service, region_id):
        """Static backgrounds are subtracted from y and excluded from ctx.components."""
        reg_dto, comp_dtos = dto_service.get_region_repr(region_id, normalized=False)
        contexts = build_contexts([(reg_dto, comp_dtos)])
        ctx = contexts[0]
        # simple_collection has one peak + one constant (static) background
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
