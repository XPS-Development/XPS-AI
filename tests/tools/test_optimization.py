"""
Tests for tools.optimization: OptimizationPlanner, LmfitOptimizer, optimize().
"""

import numpy as np
import pytest

from tools.dto import ParameterDTO, ComponentDTO
from tools.evaluation import EvaluationService
from core.math_models import PseudoVoigtPeakModel

from tools.optimization import (
    OptimizationContext,
    OptimizationPlanner,
    LmfitOptimizer,
    OptimizedComponent,
    optimize,
)


@pytest.fixture
def optimization_context(dto_service, region_id):
    """Build OptimizationContext from simple_collection region (peak + constant bg)."""
    reg_dto, comp_dtos = dto_service.get_region_repr(region_id, normalized=False)
    eval_svc = EvaluationService()
    y = reg_dto.y.copy()

    cmps_to_opt = []
    for cmp in comp_dtos:
        if cmp.kind == "background" and cmp.model.static:
            y -= eval_svc.component_y(cmp, reg_dto.x, reg_dto.y)
        else:
            cmps_to_opt.append(cmp)

    return OptimizationContext(
        id_=reg_dto.id_,
        parent_id=reg_dto.parent_id,
        normalized=reg_dto.normalized,
        x=reg_dto.x,
        y=y,
        components=tuple(cmps_to_opt),
    )


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
