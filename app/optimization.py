"""
App-layer optimization service: DTO-only interface returning Change objects.

Uses tools.optimization for lmfit-based fitting. Caller provides DTOs;
service returns BaseChange instances for CommandExecutor.
"""

from .dto import RegionDTO, ComponentDTO
from .evaluation import EvaluationService
from .command.changes import UpdateMultipleParameterValues, CompositeChange

from tools.optimization import (
    OptimizationContext,
    OptimizedComponent,
    optimize as run_optimize,
)

from typing import Sequence, Any


def components_to_changes(
    components: Sequence[OptimizedComponent],
) -> CompositeChange:
    """
    Convert optimized components to UpdateMultipleParameterValues changes.

    Parameters
    ----------
    components : Sequence[OptimizedComponent]
        Optimization results from tools.optimization.optimize.

    Returns
    -------
    CompositeChange
        Changes suitable for CommandExecutor.
    """
    return CompositeChange(
        changes=[
            UpdateMultipleParameterValues(
                component_id=c.component_id,
                parameters=c.parameters,
                normalized=c.normalized,
            )
            for c in components
        ]
    )


class OptimizationService:
    """
    Pipeline for multi-region optimization using lmfit.

    Works only with DTOs. Caller fetches region representations via
    DTOService.get_region_repr; service returns Change objects for execution.
    """

    def __init__(self) -> None:
        self.eval = EvaluationService()

    def build_contexts(
        self,
        region_reprs: Sequence[tuple[RegionDTO, tuple[ComponentDTO, ...]]],
    ) -> tuple[OptimizationContext, ...]:
        """
        Build optimization contexts from region DTOs.

        Subtracts static backgrounds from y. Caller must provide DTOs
        (e.g. via DTOService.get_region_repr).

        Parameters
        ----------
        region_reprs : Sequence[tuple[RegionDTO, tuple[ComponentDTO, ...]]]
            Region and component DTOs per region.

        Returns
        -------
        tuple[OptimizationContext, ...]
            Contexts for tools.optimization.optimize.
        """
        contexts: list[OptimizationContext] = []

        for reg_dto, component_dtos in region_reprs:
            y = reg_dto.y.copy()

            cmps_to_opt: list[ComponentDTO] = []
            for cmp in component_dtos:
                if cmp.kind == "background" and cmp.model.static:
                    y -= self.eval.component_y(cmp, reg_dto.x, reg_dto.y)
                else:
                    cmps_to_opt.append(cmp)

            ctx = OptimizationContext(
                id_=reg_dto.id_,
                parent_id=reg_dto.parent_id,
                normalized=reg_dto.normalized,
                x=reg_dto.x,
                y=y,
                components=tuple(cmps_to_opt),
            )
            contexts.append(ctx)

        return tuple(contexts)

    def optimize_regions(
        self,
        region_reprs: Sequence[tuple[RegionDTO, tuple[ComponentDTO, ...]]],
        **kwargs: Any,
    ) -> CompositeChange:
        """
        Run optimization and return Change objects for parameter updates.

        Parameters
        ----------
        region_reprs : Sequence[tuple[RegionDTO, tuple[ComponentDTO, ...]]]
            Region and component DTOs (e.g. from DTOService.get_region_repr).
        **kwargs
            Passed to lmfit.minimize.

        Returns
        -------
        CompositeChange
            CompositeChange containing UpdateMultipleParameterValues changes.
        """
        contexts = self.build_contexts(region_reprs)
        optimized = run_optimize(contexts, **kwargs)
        return components_to_changes(optimized)
