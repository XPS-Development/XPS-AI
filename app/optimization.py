import re
from dataclasses import dataclass, asdict

import numpy as np
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult


from core.collection import CoreCollection
from .dto import DTOService, RegionDTO, ComponentDTO, ParameterDTO
from .evaluation import EvaluationService

from typing import Tuple, List, Dict, Sequence


_COMPONENT_REF_RE = re.compile(r"\b([a-zA-Z0-9_]+)\b")


@dataclass(frozen=True)
class OptimizationContext(RegionDTO):
    components: Tuple[ComponentDTO]


class OptimizationPlanner:
    """
    Responsible for grouping optimization contexts into independent
    optimization tasks based on parameter dependencies.
    """

    def _build_dependency_graph(
        self,
        contexts: Tuple[OptimizationContext, ...],
    ) -> Dict[str, set[str]]:
        graph: Dict[str, set[str]] = {}

        # collect all components
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
    def _connected_components(
        graph: Dict[str, set[str]],
    ) -> List[set[str]]:
        visited = set()
        groups: List[set[str]] = []

        for node in graph:
            if node in visited:
                continue

            stack = [node]
            group = set()

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
        contexts: Tuple[OptimizationContext, ...],
        component_groups: List[set[str]],
    ) -> List[Tuple[OptimizationContext, ...]]:

        grouped: List[List[OptimizationContext]] = [[] for _ in component_groups]

        for ctx in contexts:
            ctx_cmp_ids = {cmp.id_ for cmp in ctx.components}

            for i, grp in enumerate(component_groups):
                if ctx_cmp_ids & grp:
                    grouped[i].append(ctx)
                    break

        return [tuple(g) for g in grouped if g]

    def get_groups(
        self,
        contexts: Tuple[OptimizationContext, ...],
    ) -> List[Tuple[OptimizationContext, ...]]:
        """
        Split contexts into independent optimization groups.
        """
        graph = self._build_dependency_graph(contexts)
        component_groups = self._connected_components(graph)
        return self._group_contexts(contexts, component_groups)


class LmfitOptimizer:
    """
    Maps ComponentDTO parameters to lmfit.Parameters
    and resolves component-scoped expressions.
    """

    def __init__(self):
        self.component_index: Dict[str, ComponentDTO] = {}

    def _build_component_index(self, components: Sequence[ComponentDTO]):
        """
        Build index
        """
        self.component_index = {cmp.id_: cmp for cmp in components}

    def _translate_expr(
        self,
        expr: str,
        *,
        param_name: str,
    ) -> str | None:
        """
        Translate domain expr like:
            "2 * p1234"
        into lmfit expr:
            "2 * p1234_amp"

        If referenced component is not present or parameter missing,
        return None.
        """
        tokens = _COMPONENT_REF_RE.findall(expr)
        translated = expr

        for token in tokens:
            # numeric literals should be ignored
            if token.replace(".", "", 1).isdigit():
                continue

            # token is assumed to be component_id
            target = self.component_index.get(token)
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

    def _to_params(
        self,
        components: Sequence[ComponentDTO],
    ) -> Parameters:
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
    def residual(params: Parameters, contexts: Tuple[OptimizationContext]) -> np.ndarray:
        residuals = []
        for ctx in contexts:
            y_model = np.zeros_like(ctx.y)
            for cmp in ctx.components:
                param_dict = {pname: params[f"{cmp.id_}_{pname}"].value for pname in cmp.parameters}
                y_model += cmp.model.evaluate(ctx.x, ctx.y, **param_dict)
            residuals.append(ctx.y - y_model)

        return np.concatenate(residuals)

    def _result_to_dtos(self, result: MinimizerResult) -> Tuple[ComponentDTO, ...]:
        resulted_cmps: List[ComponentDTO] = []
        for cmp in self.component_index.values():
            new_cmp = asdict(cmp)
            new_params: dict[str, ParameterDTO] = {}
            for pname, param_dto in cmp.parameters.items():
                opt_pname = f"{cmp.id_}_{pname}"
                new_val = result.params[opt_pname].value
                param_dict = asdict(param_dto)
                param_dict["value"] = new_val
                new_params[pname] = ParameterDTO(**param_dict)
            new_cmp.update(parameters=new_params)
            new_cmp = ComponentDTO(**new_cmp)
            resulted_cmps.append(new_cmp)
        return tuple(resulted_cmps)

    def optimize(self, contexts: Tuple[OptimizationContext], **kwargs) -> Tuple[ComponentDTO, ...]:
        components = tuple(cmp for ctx in contexts for cmp in ctx.components)
        self._build_component_index(components)
        params = self._to_params(components)
        result = minimize(self.residual, params, args=(contexts,), **kwargs, nan_policy="omit")
        return self._result_to_dtos(result)


class OptimizationService:
    """
    Pipeline for multi-region optimization using lmfit.
    """

    def __init__(self, collection: CoreCollection):
        self.dto = DTOService(collection)
        self.eval = EvaluationService()
        self.planner = OptimizationPlanner()
        self.optimizer = LmfitOptimizer()

    def _build_context(self, region_id: str, normalize: bool = True) -> OptimizationContext:
        reg_dto, component_dtos = self.dto.get_region_repr(region_id, normalize=normalize)
        y = reg_dto.y.copy()

        cmps_to_opt = []
        for cmp in component_dtos:
            if cmp.kind == "background" and cmp.model.static:
                y -= self.eval.component_y(cmp, reg_dto.x, reg_dto.y)
            else:
                cmps_to_opt.append(cmp)

        reg_dto_dict = asdict(reg_dto)
        reg_dto_dict["y"] = y
        return OptimizationContext(**reg_dto_dict, components=tuple(cmps_to_opt))

    def build_contexts(self, region_ids: List[str], normalize: bool = True) -> Tuple[OptimizationContext]:
        return tuple(self._build_context(rid, normalize) for rid in region_ids)

    def apply(self, components: Tuple[ComponentDTO, ...]):
        for cmp in components:
            self.dto.apply_component(cmp, values_only=True)

    def optimize_regions(self, *region_ids: str, **kwargs):
        contexts = self.build_contexts(region_ids)
        for ctx_group in self.planner.get_groups(contexts):
            optimized_components = self.optimizer.optimize(ctx_group, **kwargs)
            self.apply(optimized_components)
