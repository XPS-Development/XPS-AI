"""
Lmfit-based optimization for spectral fitting.

Provides OptimizationContext, OptimizationPlanner, LmfitOptimizer, and optimize()
for use as a standalone library or via the app layer. Uses core.types protocols only;
"""

import re
from dataclasses import dataclass

import numpy as np
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult

from core.types import ComponentLike

from typing import Sequence, Any

_COMPONENT_REF_RE = re.compile(r"\b([a-zA-Z0-9_]+)\b")


@dataclass(frozen=True)
class OptimizationContext:
    """
    Region-like context with components to optimize.

    Satisfies RegionLike requirements (x, y) plus components.
    Uses protocols for library API compatibility.
    """

    id_: str
    parent_id: str | None
    normalized: bool
    x: np.ndarray
    y: np.ndarray
    components: tuple[ComponentLike, ...]


@dataclass(frozen=True)
class OptimizedComponent:
    """
    Minimal result of optimization: component ID and new parameter values.

    Used by library API and by app layer adapter to produce Change objects.
    """

    component_id: str
    parameters: dict[str, float]


class OptimizationPlanner:
    """
    Groups optimization contexts into independent tasks based on parameter dependencies.
    """

    def _build_dependency_graph(
        self,
        contexts: tuple[OptimizationContext, ...],
    ) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = {}
        components = [cmp for ctx in contexts for cmp in ctx.components]

        for cmp in components:
            graph.setdefault(cmp.id_, set())

        for cmp in components:
            for p in cmp.parameters.values():
                if not p.expr:
                    continue

                tokens = _COMPONENT_REF_RE.findall(p.expr)
                for token in tokens:
                    if token == cmp.id_:
                        continue
                    if token in graph:
                        graph[cmp.id_].add(token)
                        graph[token].add(cmp.id_)

        return graph

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
    ) -> list[tuple[OptimizationContext, ...]]:
        """
        Split contexts into independent optimization groups.
        """
        graph = self._build_dependency_graph(contexts)
        component_groups = self._connected_components(graph)
        return self._group_contexts(contexts, component_groups)


class LmfitOptimizer:
    """
    Maps ComponentLike parameters to lmfit.Parameters and resolves component-scoped expressions.
    """

    def __init__(self) -> None:
        self._component_index: dict[str, ComponentLike] = {}

    def _build_component_index(self, components: Sequence[ComponentLike]) -> None:
        self._component_index = {cmp.id_: cmp for cmp in components}

    def _translate_expr(
        self,
        expr: str,
        *,
        param_name: str,
    ) -> str | None:
        """
        Translate core expr like "2 * p1234" into lmfit expr "2 * p1234_amp".
        """
        tokens = _COMPONENT_REF_RE.findall(expr)
        translated = expr

        for token in tokens:
            if token.replace(".", "", 1).isdigit():
                continue

            target = self._component_index.get(token)
            if target is None:
                return None

            if param_name not in target.parameters:
                return None

            translated = re.sub(
                rf"\b{token}\b",
                f"{token}_{param_name}",
                translated,
            )

        return translated

    def _to_params(self, components: Sequence[ComponentLike]) -> Parameters:
        params = Parameters()

        for cmp in components:
            for pname, param_obj in cmp.parameters.items():
                full_name = f"{cmp.id_}_{pname}"

                expr = None
                if param_obj.expr:
                    expr = self._translate_expr(param_obj.expr, param_name=pname)

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
            for pname in cmp.parameters:
                opt_pname = f"{cmp.id_}_{pname}"
                params[pname] = result.params[opt_pname].value
            output.append(OptimizedComponent(component_id=cmp.id_, parameters=params))
        return tuple(output)

    def optimize(
        self,
        contexts: tuple[OptimizationContext, ...],
        **kwargs: Any,
    ) -> tuple[OptimizedComponent, ...]:
        """
        Run lmfit minimization and return optimized parameter values per component.
        """
        components = tuple(cmp for ctx in contexts for cmp in ctx.components)
        self._build_component_index(components)
        params = self._to_params(components)
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
    **kwargs: Any,
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
    result: list[OptimizedComponent] = []

    for ctx_group in planner.get_groups(ctx_tuple):
        result.extend(optimizer.optimize(ctx_group, **kwargs))

    return tuple(result)
