"""
App-layer optimization service: DTO-only interface returning Change objects.

Uses tools.optimization for lmfit-based fitting. Caller provides DTOs;
service returns BaseChange instances for CommandExecutor.
"""

from typing import Any, Sequence

from tools.dto import ComponentDTO, RegionDTO
from tools.optimization import (
    OptimizedComponent,
    build_contexts,
    optimize as run_optimize,
)

from .command.changes import CompositeChange, UpdateMultipleParameterValues


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
        contexts = build_contexts(region_reprs)
        optimized = run_optimize(contexts, **kwargs)
        return components_to_changes(optimized)
