"""
Lmfit-based optimization for spectral fitting.

Provides OptimizationContext, OptimizationExpressionPlan, OptimizationPlanner,
LmfitOptimizer, and optimize()
for use as a standalone library or via the app layer. Uses core.dto projections;
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult

from core.dto import ComponentDTO, RegionDTO
from core.evaluation import component_y
from core.fitting.expressions import parse_parameter_expression, resolve_component_reference

__all__ = [
    "LmfitOptimizer",
    "OptimizationContext",
    "OptimizationExpressionPlan",
    "OptimizationPlanner",
    "OptimizedComponent",
    "build_contexts",
    "optimize",
    "resolve_component_reference",
]


def _component_fully_fixed(cmp: ComponentDTO) -> bool:
    """Return True if every parameter has ``vary=False``.

    Lmfit holds values fixed; expr is ignored for vary.
    Such components are subtracted from region ``y`` and omitted from the fit.
    """
    if not cmp.parameters:
        return False
    return not any(p.vary for p in cmp.parameters.values())


@dataclass(frozen=True)
class OptimizationContext:
    """
    Region-like context with components to optimize.

    Holds region arrays (x, y) plus components to optimize.
    """

    id_: str
    parent_id: str | None
    normalized: bool
    x: np.ndarray
    y: np.ndarray
    components: tuple[ComponentDTO, ...]


def build_contexts(
    region_reprs: Sequence[tuple[RegionDTO, Sequence[ComponentDTO]]],
) -> tuple[OptimizationContext, ...]:
    """
    Build optimization contexts from region and component DTOs.

    Subtracts contributions of fully fixed components (all parameters have
    ``vary=False``) from ``y`` and includes only components to optimize.
    Works with RegionDTO and ComponentDTO
    (e.g. DTOs from DTOService.get_region_repr).

    Parameters
    ----------
    region_reprs : Sequence[tuple[RegionDTO, Sequence[ComponentDTO]]]
        Per-region (region, components) pairs.

    Returns
    -------
    tuple[OptimizationContext, ...]
        Contexts for optimize().
    """
    contexts: list[OptimizationContext] = []

    for region, components in region_reprs:
        y = region.y.copy()
        cmps_to_opt: list[ComponentDTO] = []

        for cmp in components:
            if _component_fully_fixed(cmp):
                y -= component_y(cmp, region.x, region.y)
            else:
                cmps_to_opt.append(cmp)

        ctx = OptimizationContext(
            id_=region.id_,
            parent_id=region.parent_id,
            normalized=region.normalized,
            x=region.x,
            y=y,
            components=tuple(cmps_to_opt),
        )
        contexts.append(ctx)

    return tuple(contexts)


@dataclass(frozen=True)
class OptimizedComponent:
    """
    Minimal result of optimization: component ID and new parameter values.

    Used by library API and by app layer adapter to produce Change objects.
    """

    component_id: str
    parameters: dict[str, float]
    normalized: bool


@dataclass(frozen=True)
class OptimizationExpressionPlan:
    """
    Single pass over parameter expressions: dependency graph and lmfit translations.

    Built from all components in the optimization scope so grouping and fitting
    reuse the same token resolution without re-parsing expressions.

    Attributes
    ----------
    dependency_graph : dict[str, set[str]]
        Symmetric adjacency between component ids derived from expression references.
    lmfit_expr_by_component_param : dict[tuple[str, str], str | None]
        For each constrained parameter, the lmfit ``expr`` string or ``None`` if
        resolution failed.
    """

    dependency_graph: dict[str, set[str]]
    lmfit_expr_by_component_param: dict[tuple[str, str], str | None]


def _analyze_parameter_expression(
    expr: str,
    *,
    owner_component_id: str,
    param_name: str,
    known_ids: frozenset[str],
    parameter_names_by_component: dict[str, set[str]],
    graph: dict[str, set[str]],
) -> str | None:
    """Parse ``expr`` via :func:`parse_parameter_expression` and update ``graph``."""
    parsed = parse_parameter_expression(
        expr,
        owner_component_id=owner_component_id,
        parameter_name=param_name,
        known_component_ids=known_ids,
        parameter_names_by_component=parameter_names_by_component,
    )
    for ref in parsed.references:
        if ref.component_id != owner_component_id and ref.component_id in graph:
            graph[owner_component_id].add(ref.component_id)
            graph[ref.component_id].add(owner_component_id)
    return parsed.lmfit_expr


def _build_expression_plan(
    contexts: tuple[OptimizationContext, ...],
) -> OptimizationExpressionPlan:
    components = [cmp for ctx in contexts for cmp in ctx.components]
    known_ids = frozenset(cmp.id_ for cmp in components)
    parameter_names_by_component = {cmp.id_: set(cmp.parameters.keys()) for cmp in components}

    graph: dict[str, set[str]] = {}
    for cmp in components:
        graph.setdefault(cmp.id_, set())

    lmfit_expr_by_component_param: dict[tuple[str, str], str | None] = {}

    for cmp in components:
        for pname, p in cmp.parameters.items():
            if not p.expr:
                continue
            lmfit_e = _analyze_parameter_expression(
                p.expr,
                owner_component_id=cmp.id_,
                param_name=pname,
                known_ids=known_ids,
                parameter_names_by_component=parameter_names_by_component,
                graph=graph,
            )
            lmfit_expr_by_component_param[(cmp.id_, pname)] = lmfit_e

    return OptimizationExpressionPlan(
        dependency_graph=graph,
        lmfit_expr_by_component_param=lmfit_expr_by_component_param,
    )


class OptimizationPlanner:
    """Groups optimization contexts into independent tasks based on parameter dependencies."""

    @staticmethod
    def _connected_components(graph: dict[str, set[str]]) -> list[set[str]]:
        visited: set[str] = set()
        groups: list[set[str]] = []

        for node in graph:
            if node in visited:
                continue

            stack = [node]
            group: set[str] = set()

            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue

                visited.add(cur)
                group.add(cur)
                stack.extend(graph[cur] - visited)

            groups.append(group)

        return groups

    @staticmethod
    def _group_contexts(
        contexts: tuple[OptimizationContext, ...],
        component_groups: list[set[str]],
    ) -> list[tuple[OptimizationContext, ...]]:
        grouped: list[list[OptimizationContext]] = [[] for _ in component_groups]

        for ctx in contexts:
            ctx_cmp_ids = {cmp.id_ for cmp in ctx.components}

            for i, grp in enumerate(component_groups):
                if ctx_cmp_ids & grp:
                    grouped[i].append(ctx)
                    break

        return [tuple(g) for g in grouped if g]

    def get_groups(
        self,
        contexts: tuple[OptimizationContext, ...],
        *,
        expression_plan: OptimizationExpressionPlan | None = None,
    ) -> list[tuple[OptimizationContext, ...]]:
        """
        Split contexts into independent optimization groups.

        Parameters
        ----------
        contexts : tuple[OptimizationContext, ...]
            Contexts to partition.
        expression_plan : OptimizationExpressionPlan | None
            Pre-built expression analysis for ``contexts``. If ``None``, a plan
            is computed once from ``contexts`` (same graph as ``dependency_graph``).
        """
        if expression_plan is not None:
            graph = expression_plan.dependency_graph
        else:
            graph = _build_expression_plan(contexts).dependency_graph
        component_groups = self._connected_components(graph)
        return self._group_contexts(contexts, component_groups)


class LmfitOptimizer:
    """Maps ComponentDTO parameters to lmfit.Parameters and resolves component-scoped expressions."""

    def __init__(self) -> None:
        self._component_index: dict[str, ComponentDTO] = {}

    def _build_component_index(self, components: Sequence[ComponentDTO]) -> None:
        self._component_index = {cmp.id_: cmp for cmp in components}

    def _translate_expr_for_component(
        self,
        owner_component_id: str,
        expr: str,
        *,
        param_name: str,
    ) -> str | None:
        known_ids = frozenset(self._component_index.keys())
        parameter_names_by_component = {
            cid: set(cmp.parameters.keys()) for cid, cmp in self._component_index.items()
        }
        graph = {cid: set() for cid in self._component_index}
        return _analyze_parameter_expression(
            expr,
            owner_component_id=owner_component_id,
            param_name=param_name,
            known_ids=known_ids,
            parameter_names_by_component=parameter_names_by_component,
            graph=graph,
        )

    def _to_params(
        self,
        components: Sequence[ComponentDTO],
        *,
        expression_plan: OptimizationExpressionPlan | None = None,
    ) -> Parameters:
        params = Parameters()

        for cmp in components:
            for pname, param_obj in cmp.parameters.items():
                full_name = f"{cmp.id_}_{pname}"

                expr = None
                if param_obj.expr:
                    if expression_plan is not None:
                        expr = expression_plan.lmfit_expr_by_component_param.get((cmp.id_, pname))
                    else:
                        expr = self._translate_expr_for_component(
                            cmp.id_,
                            param_obj.expr,
                            param_name=pname,
                        )

                params.add(
                    full_name,
                    value=param_obj.value,
                    min=param_obj.lower,
                    max=param_obj.upper,
                    vary=param_obj.vary if expr is None else False,
                    expr=expr,
                )

        return params

    @staticmethod
    def residual(
        params: Parameters,
        contexts: tuple[OptimizationContext, ...],
    ) -> np.ndarray:
        """Return concatenated residuals for all optimization contexts."""
        residuals: list[np.ndarray] = []
        for ctx in contexts:
            y_model = np.zeros_like(ctx.y)
            for cmp in ctx.components:
                param_dict = {pname: params[f"{cmp.id_}_{pname}"].value for pname in cmp.parameters}
                y_model += cmp.model.evaluate(ctx.x, ctx.y, **param_dict)
            residuals.append(ctx.y - y_model)

        return np.concatenate(residuals)

    def _result_to_optimized(
        self,
        result: MinimizerResult,
    ) -> tuple[OptimizedComponent, ...]:
        output: list[OptimizedComponent] = []
        for cmp in self._component_index.values():
            params: dict[str, float] = {}
            params_obj = getattr(result, "params", None)
            if params_obj is None:
                raise RuntimeError("lmfit MinimizerResult missing `params` attribute")
            for pname in cmp.parameters:
                opt_pname = f"{cmp.id_}_{pname}"
                params[pname] = params_obj[opt_pname].value
            output.append(
                OptimizedComponent(
                    component_id=cmp.id_,
                    parameters=params,
                    normalized=cmp.normalized,
                )
            )
        return tuple(output)

    def optimize(
        self,
        contexts: tuple[OptimizationContext, ...],
        *,
        expression_plan: OptimizationExpressionPlan | None = None,
        **kwargs,
    ) -> tuple[OptimizedComponent, ...]:
        """
        Run lmfit minimization and return optimized parameter values per component.

        Parameters
        ----------
        contexts : tuple[OptimizationContext, ...]
            Contexts for this minimization subproblem.
        expression_plan : OptimizationExpressionPlan | None
            Pre-built lmfit expression strings for constrained parameters. If
            ``None``, expressions are resolved from ``contexts`` only.
        **kwargs
            Forwarded to ``lmfit.minimize``.
        """
        components = tuple(cmp for ctx in contexts for cmp in ctx.components)
        self._build_component_index(components)
        params = self._to_params(components, expression_plan=expression_plan)
        result = minimize(
            self.residual,
            params,
            args=(contexts,),
            **kwargs,
            nan_policy="omit",
        )
        return self._result_to_optimized(result)


def optimize(
    contexts: Sequence[OptimizationContext],
    **kwargs,
) -> tuple[OptimizedComponent, ...]:
    """
    Library API entry point: run optimization on contexts and return optimized components.

    Parameters
    ----------
    contexts : Sequence[OptimizationContext]
        Optimization contexts (region data + components to fit).
    **kwargs
        Passed to lmfit.minimize (e.g. method='least_squares', method='differential_evolution').

    Returns
    -------
    tuple[OptimizedComponent, ...]
        Optimized parameter values per component.
    """
    planner = OptimizationPlanner()
    optimizer = LmfitOptimizer()

    ctx_tuple = tuple(contexts)
    plan = _build_expression_plan(ctx_tuple)
    result: list[OptimizedComponent] = []

    for ctx_group in planner.get_groups(ctx_tuple, expression_plan=plan):
        result.extend(optimizer.optimize(ctx_group, expression_plan=plan, **kwargs))

    return tuple(result)
