import re
from dataclasses import dataclass
import numpy as np

from lmfit import Parameters, minimize

from typing import List, Dict, Optional

from .parametrics import RuntimeParameter, NormalizationContext, ParametricModelLike
from .domain import SpectrumCollection, Background, Component
from .domain_services import CollectionQueryService, DataQueryService, ComponentService


@dataclass
class OptimizableComponent:
    id_: str
    model: ParametricModelLike
    parameters: Dict[str, RuntimeParameter]
    is_background: bool
    active: bool

    def evaluate(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.model.evaluate(x, y, **{k: p.value for k, p in self.parameters.items()})

    @classmethod
    def from_component(
        cls, cmp: Component, norm_ctx: Optional[NormalizationContext] = None
    ) -> "OptimizableComponent":
        if norm_ctx is not None:
            parameters = cmp.model.normalize(cmp.parameters, norm_ctx)
        else:
            parameters = cmp.parameters

        return cls(
            id_=cmp.id_,
            model=cmp.model,
            parameters=parameters,
            is_background=isinstance(cmp, Background),
            active=getattr(cmp.model, "is_active", True),
        )

    def get_parameters(self, norm_ctx: Optional[NormalizationContext] = None) -> Dict[str, float]:
        denorm_params = self.model.denormalize(self.parameters, norm_ctx)
        return {k: p.value for k, p in denorm_params.items()}


@dataclass
class OptimizationContext:
    region_id: str
    x: np.ndarray
    y: np.ndarray
    components: List[OptimizableComponent]
    norm_ctx: Optional[NormalizationContext] = None


class DTOBuilder:
    """
    Converts Region + Components into OptimizationContext for optimization.
    """

    def __init__(self, query: CollectionQueryService, data: DataQueryService):
        self.query = query
        self.data = data

    def build_context(self, region_id: str) -> OptimizationContext:
        print(region_id)
        norm_ctx = self.data.get_norm_ctx(region_id=region_id)
        x, y = self.data.get_region_data(region_id, normalized=True)
        y_adj = y.copy()

        components = []
        for cmp in self.query.get_components(region_id):
            opt_cmp = OptimizableComponent.from_component(cmp, norm_ctx)
            if opt_cmp.is_background and not opt_cmp.active:
                y_adj -= opt_cmp.evaluate(x, y)
            else:
                components.append(opt_cmp)

        return OptimizationContext(
            region_id=region_id, x=x, y=y_adj, components=components, norm_ctx=norm_ctx
        )


_COMPONENT_REF_RE = re.compile(r"\b([a-zA-Z0-9_]+)\b")


class ParameterMapper:
    """
    Maps OptimizableComponent parameters to lmfit.Parameters
    and resolves component-scoped expressions.
    """

    @staticmethod
    def _build_component_index(
        contexts: List[OptimizationContext],
    ) -> Dict[str, OptimizableComponent]:
        """
        Build index: component_id -> OptimizableComponent
        """
        index: Dict[str, OptimizableComponent] = {}
        for ctx in contexts:
            for cmp in ctx.components:
                index[cmp.id_] = cmp
        return index

    @staticmethod
    def _translate_expr(
        expr: str,
        *,
        param_name: str,
        component_index: Dict[str, OptimizableComponent],
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
            target = component_index.get(token)
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

    @classmethod
    def to_lmfit_params(
        cls,
        contexts: List[OptimizationContext],
    ) -> Parameters:
        params = Parameters()
        component_index = cls._build_component_index(contexts)

        for ctx in contexts:
            for cmp in ctx.components:
                for pname, rp in cmp.parameters.items():
                    full_name = f"{cmp.id_}_{pname}"

                    expr = None
                    if rp.expr:
                        expr = cls._translate_expr(
                            rp.expr,
                            param_name=pname,
                            component_index=component_index,
                        )

                    params.add(
                        full_name,
                        value=rp.value,
                        min=rp.lower,
                        max=rp.upper,
                        vary=rp.vary if expr is None else False,
                        expr=expr,
                    )

        return params

    @staticmethod
    def update_components_from_params(
        params: Parameters,
        contexts: List[OptimizationContext],
    ) -> None:
        for ctx in contexts:
            for cmp in ctx.components:
                for pname, rp in cmp.parameters.items():
                    full_name = f"{cmp.id_}_{pname}"
                    if full_name in params:
                        rp.set(value=params[full_name].value)


class OptimizationService:
    """
    Pipeline for multi-region optimization using lmfit.
    """

    def __init__(self, collection: SpectrumCollection):
        self.collection = collection
        self.query = CollectionQueryService(collection)
        self.data = DataQueryService(collection)
        self.cmp_serv = ComponentService(collection)
        self.dto_builder = DTOBuilder(self.query, self.data)

    def build_contexts(self, region_ids: List[str]) -> List[OptimizationContext]:
        return [self.dto_builder.build_context(rid) for rid in region_ids]

    @staticmethod
    def residual(params: Parameters, contexts: List[OptimizationContext]) -> np.ndarray:
        residuals = []
        for ctx in contexts:
            y_model = np.zeros_like(ctx.y)
            for cmp in ctx.components:
                param_dict = {pname: params[f"{cmp.id_}_{pname}"].value for pname in cmp.parameters}
                y_model += cmp.model.evaluate(ctx.x, ctx.y, **param_dict)
            residuals.append(ctx.y - y_model)

        return np.concatenate(residuals)

    def apply_to_domain(self, contexts: List[OptimizationContext]):
        for ctx in contexts:
            for cmp in ctx.components:
                denorm_params = cmp.get_parameters(ctx.norm_ctx)
                self.cmp_serv.set_parameters(cmp.id_, denorm_params)

    def optimize_regions(self, region_ids: List[str], **kwargs) -> Parameters:
        print(region_ids)
        contexts = self.build_contexts(region_ids)
        lmfit_params = ParameterMapper.to_lmfit_params(contexts)
        result = minimize(self.residual, lmfit_params, args=(contexts,), **kwargs)
        ParameterMapper.update_components_from_params(result.params, contexts)
        self.apply_to_domain(contexts)
        return result.params
