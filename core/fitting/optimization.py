"""
Lmfit-based optimization for spectral fitting.

Provides OptimizationContext, OptimizationExpressionPlan, OptimizationPlanner,
LmfitOptimizer, and optimize()
for use as a standalone library or via the app layer. Uses core.dto projections;
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult

from core.dto import ComponentDTO, RegionDTO
from core.evaluation import chi_square_residual, component_y
from core.fitting.expressions import parse_parameter_expression, resolve_component_reference
from core.math_models.normalization import NormalizationContext

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

    Attributes
    ----------
    measured_y : np.ndarray or None
        Original intensity used as the Poisson weight. When ``None``, ``y``
        is used. Set this when fixed components have been subtracted from
        ``y`` so the variance still follows the measured counts.
    norm_offset, norm_scale : float or None
        Intensity normalization for solver variables. Intensity parameters are
        presented to the optimizer in this space and mapped back to counts
        before the model is evaluated.
    """

    id_: str
    parent_id: str | None
    normalized: bool
    x: np.ndarray
    y: np.ndarray
    components: tuple[ComponentDTO, ...]
    measured_y: np.ndarray | None = None
    norm_offset: float | None = None
    norm_scale: float | None = None


def build_contexts(
    region_reprs: Sequence[tuple[RegionDTO, Sequence[ComponentDTO]]],
) -> tuple[OptimizationContext, ...]:
    """
    Build optimization contexts from region and component DTOs.

    Subtracts contributions of fully fixed components (all parameters have
    ``vary=False``) from ``y`` and includes only components to optimize.
    Poisson weights stay on the original measured intensity. Works with
    RegionDTO and ComponentDTO (e.g. DTOs from DTOService.get_region_repr).

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
        measured = np.asarray(region.y, dtype=float).copy()
        y = measured.copy()
        cmps_to_opt: list[ComponentDTO] = []

        for cmp in components:
            if _component_fully_fixed(cmp):
                y -= component_y(cmp, region.x, region.y)
            else:
                cmps_to_opt.append(cmp)

        norm_offset: float | None = None
        norm_scale: float | None = None
        try:
            norm = NormalizationContext.from_array(measured)
        except ValueError:
            norm = None
        if norm is not None:
            norm_offset = norm.offset
            norm_scale = norm.scale

        ctx = OptimizationContext(
            id_=region.id_,
            parent_id=region.parent_id,
            normalized=region.normalized,
            x=region.x,
            y=y,
            components=tuple(cmps_to_opt),
            measured_y=measured,
            norm_offset=norm_offset,
            norm_scale=norm_scale,
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
    def _context_dependency_graph(
        contexts: tuple[OptimizationContext, ...],
        component_graph: dict[str, set[str]],
    ) -> dict[str, set[str]]:
        """
        Lift component expression edges onto the contexts (regions) that own them.

        A single region may hold peaks from more than one component-level clique.
        Grouping contexts by the first intersecting component clique orphans
        cross-clique expressions and makes lmfit raise ``NameError``. Context-level
        edges keep every region that participates in a dependency chain together.
        """
        context_ids = [ctx.id_ for ctx in contexts]
        graph: dict[str, set[str]] = {cid: set() for cid in context_ids}
        owner_context: dict[str, str] = {}
        for ctx in contexts:
            for cmp in ctx.components:
                owner_context[cmp.id_] = ctx.id_

        for component_id, neighbors in component_graph.items():
            ctx_a = owner_context.get(component_id)
            if ctx_a is None:
                continue
            for other_id in neighbors:
                ctx_b = owner_context.get(other_id)
                if ctx_b is None or ctx_b == ctx_a:
                    continue
                graph[ctx_a].add(ctx_b)
                graph[ctx_b].add(ctx_a)

        return graph

    def get_groups(
        self,
        contexts: tuple[OptimizationContext, ...],
        *,
        expression_plan: OptimizationExpressionPlan | None = None,
    ) -> list[tuple[OptimizationContext, ...]]:
        """
        Split contexts into independent optimization groups.

        Groups are connected components of **contexts** linked by expression
        dependencies between their components. Contexts with no cross-links
        remain separate so independent regions can still fit in parallel.

        Parameters
        ----------
        contexts : tuple[OptimizationContext, ...]
            Contexts to partition.
        expression_plan : OptimizationExpressionPlan | None
            Pre-built expression analysis for ``contexts``. If ``None``, a plan
            is computed once from ``contexts`` (same graph as ``dependency_graph``).
        """
        if not contexts:
            return []
        if expression_plan is not None:
            component_graph = expression_plan.dependency_graph
        else:
            component_graph = _build_expression_plan(contexts).dependency_graph
        context_graph = self._context_dependency_graph(contexts, component_graph)
        groups: list[tuple[OptimizationContext, ...]] = []
        for ctx_ids in self._connected_components(context_graph):
            # Preserve input order within each group.
            groups.append(tuple(ctx for ctx in contexts if ctx.id_ in ctx_ids))
        # Keep deterministic order by first context appearance in the input tuple.
        groups.sort(key=lambda group: contexts.index(group[0]))
        return groups


def _context_norm(ctx: OptimizationContext) -> NormalizationContext | None:
    """Return the solver normalization for ``ctx``, if intensity scaling is active."""
    if ctx.norm_offset is None or ctx.norm_scale is None or ctx.norm_scale <= 0.0:
        return None
    return NormalizationContext(offset=ctx.norm_offset, scale=ctx.norm_scale)


def _norms_by_component(
    contexts: Sequence[OptimizationContext],
) -> dict[str, NormalizationContext]:
    """Map each fitted component id to the normalization of its region."""
    norms: dict[str, NormalizationContext] = {}
    for ctx in contexts:
        norm = _context_norm(ctx)
        if norm is None:
            continue
        for cmp in ctx.components:
            norms[cmp.id_] = norm
    return norms


def _solver_token_to_raw(token: str, model: object, norm: NormalizationContext) -> str:
    """Embed a solver variable in an expression that yields its raw value."""
    expr = token
    if getattr(model, "use_scale", False):
        expr = f"(({expr})*({norm.scale!r}))"
    if getattr(model, "use_offset", False):
        expr = f"(({expr})+({norm.offset!r}))"
    return expr


def _raw_expr_to_solver(raw_expr: str, model: object, norm: NormalizationContext) -> str:
    """Map a raw-unit expression result into solver space."""
    expr = f"({raw_expr})"
    if getattr(model, "use_offset", False):
        expr = f"({expr}-({norm.offset!r}))"
    if getattr(model, "use_scale", False):
        expr = f"({expr}/({norm.scale!r}))"
    return expr


def _expression_in_solver_space(
    expr: str,
    *,
    owner: ComponentDTO,
    owner_param: str,
    components: Mapping[str, ComponentDTO],
    norms: Mapping[str, NormalizationContext],
) -> str:
    """Rewrite a raw-unit lmfit expression so it constrains solver variables.

    Component references that are intensity parameters are denormalized to
    counts inside the expression. When the constrained parameter itself is an
    intensity, the whole result is mapped back into solver space.
    """
    tokens: list[tuple[str, ComponentDTO, str]] = []
    for cmp in components.values():
        for pname in cmp.parameters:
            if pname not in cmp.model.normalization_target_parameters:
                continue
            if cmp.id_ not in norms:
                continue
            tokens.append((f"{cmp.id_}_{pname}", cmp, pname))
    tokens.sort(key=lambda item: len(item[0]), reverse=True)

    rewritten = expr
    placeholders: dict[str, str] = {}
    for index, (token, cmp, _pname) in enumerate(tokens):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"
        if re.search(pattern, rewritten) is None:
            continue
        placeholder = f"__solver_ref_{index}__"
        rewritten = re.sub(pattern, placeholder, rewritten)
        placeholders[placeholder] = _solver_token_to_raw(token, cmp.model, norms[cmp.id_])
    for placeholder, raw_token in placeholders.items():
        rewritten = rewritten.replace(placeholder, raw_token)

    owner_norm = norms.get(owner.id_)
    if owner_param in owner.model.normalization_target_parameters and owner_norm is not None:
        rewritten = _raw_expr_to_solver(rewritten, owner.model, owner_norm)
    return rewritten


def _solver_value_to_raw(
    value: float,
    component: ComponentDTO,
    name: str,
    norm: NormalizationContext | None,
) -> float:
    """Map one solver parameter back to counts when it is intensity-normalized."""
    if norm is None or name not in component.model.normalization_target_parameters:
        return value
    return float(component.model.denormalize_value(value, norm))


def _raw_value_to_solver(
    value: float,
    component: ComponentDTO,
    name: str,
    norm: NormalizationContext | None,
) -> float:
    """Map one raw parameter into solver space when it is intensity-normalized."""
    if norm is None or name not in component.model.normalization_target_parameters:
        return value
    return float(component.model.normalize_value(value, norm))


class LmfitOptimizer:
    """Maps ComponentDTO parameters to lmfit.Parameters and resolves component-scoped expressions."""

    def __init__(self) -> None:
        self._component_index: dict[str, ComponentDTO] = {}
        self._norm_by_component: dict[str, NormalizationContext] = {}

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

        # Add every parameter without expr first so cross-component references
        # resolve regardless of region/component iteration order.
        pending_expr: list[tuple[str, str, ComponentDTO, str]] = []
        for cmp in components:
            norm = self._norm_by_component.get(cmp.id_)
            for pname, param_obj in cmp.parameters.items():
                full_name = f"{cmp.id_}_{pname}"
                expr: str | None = None
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
                    value=_raw_value_to_solver(float(param_obj.value), cmp, pname, norm),
                    min=_raw_value_to_solver(float(param_obj.lower), cmp, pname, norm),
                    max=_raw_value_to_solver(float(param_obj.upper), cmp, pname, norm),
                    vary=param_obj.vary if expr is None else False,
                    expr=None,
                )
                if expr is not None:
                    pending_expr.append((full_name, expr, cmp, pname))

        for full_name, expr, cmp, pname in pending_expr:
            params[full_name].set(
                expr=_expression_in_solver_space(
                    expr,
                    owner=cmp,
                    owner_param=pname,
                    components=self._component_index,
                    norms=self._norm_by_component,
                )
            )

        return params

    @staticmethod
    def residual(
        params: Parameters,
        contexts: tuple[OptimizationContext, ...],
    ) -> np.ndarray:
        """Return concatenated Poisson chi-squared residuals.

        The sum of squares of the result is the chi-squared criterion
        ``Σ (target - model)² / max(|measured|, 1)``.
        """
        residuals: list[np.ndarray] = []
        for ctx in contexts:
            y_model = np.zeros_like(ctx.y)
            norm = _context_norm(ctx)
            for cmp in ctx.components:
                param_dict = {
                    pname: _solver_value_to_raw(
                        float(params[f"{cmp.id_}_{pname}"].value), cmp, pname, norm
                    )
                    for pname in cmp.parameters
                }
                y_model += cmp.model.evaluate(ctx.x, ctx.y, **param_dict)
            measured = ctx.y if ctx.measured_y is None else ctx.measured_y
            residuals.append(chi_square_residual(ctx.y - y_model, measured))

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
            norm = self._norm_by_component.get(cmp.id_)
            for pname in cmp.parameters:
                opt_pname = f"{cmp.id_}_{pname}"
                params[pname] = _solver_value_to_raw(
                    float(params_obj[opt_pname].value), cmp, pname, norm
                )
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
        self._norm_by_component = _norms_by_component(contexts)
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
    groups = planner.get_groups(ctx_tuple, expression_plan=plan)

    for ctx_group in groups:
        if len(groups) == 1 and len(ctx_group) == len(ctx_tuple):
            group_plan = plan
        else:
            # Per-group plan so exprs never reference params outside this minimize.
            group_plan = _build_expression_plan(ctx_group)
        result.extend(optimizer.optimize(ctx_group, expression_plan=group_plan, **kwargs))

    return tuple(result)
