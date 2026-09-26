"""
Analysis use-cases: NN segmentation and region optimization.

Builds Change objects from query state and app services. Does not execute
commands. ``auto_fit`` remains two orchestrator executions so optimization
sees regions created by the segmenter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.command.changes import CompositeChange
from core.fitting.fit_scope import (
    ExpressionProblem,
    collect_expression_problems,
    expand_fit_region_ids,
    format_expression_problems,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.nn_service import NNService
    from app.optimization import OptimizationService
    from app.parameters import AppParameters
    from app.query_service import QueryService
    from core.dto import ComponentDTO, RegionDTO


@dataclass(frozen=True)
class FitScopePreview:
    """
    Selected vs expression-closed region set for an upcoming optimize.

    Attributes
    ----------
    selected_region_ids : tuple of str
        Regions resolved from the caller's ``region_ids`` / ``spectrum_ids``.
    expanded_region_ids : tuple of str
        Selected regions plus transitive expression dependents.
    extra_region_count : int
        ``len(expanded) - len(selected)`` (regions pulled in by exprs).
    extra_spectrum_count : int
        Distinct parent spectra of those extra regions.
    expression_problems : tuple of ExpressionProblem
        Invalid exprs on components in the expanded region set.
    selected_expression_problems : tuple of ExpressionProblem
        Invalid exprs on components in the selected region set only.
    """

    selected_region_ids: tuple[str, ...]
    expanded_region_ids: tuple[str, ...]
    extra_region_count: int
    extra_spectrum_count: int
    expression_problems: tuple[ExpressionProblem, ...] = ()
    selected_expression_problems: tuple[ExpressionProblem, ...] = ()

    @property
    def needs_confirmation(self) -> bool:
        """Return True when optimization would touch regions outside the selection."""
        return self.extra_region_count > 0

    @property
    def has_expression_errors(self) -> bool:
        """Return True when the expanded scope has unresolved parameter expressions."""
        return bool(self.expression_problems)

    @property
    def has_selected_expression_errors(self) -> bool:
        """Return True when the selection alone has unresolved parameter expressions."""
        return bool(self.selected_expression_problems)


class AnalysisUseCases:
    """
    Build Change objects for segmentation and optimization workflows.

    Parameters
    ----------
    query
        Read-only query façade for collection and DTO access.
    nn
        App-level NN service that returns segmenter Change objects.
    optimization
        App-level optimization service that returns parameter-update Changes.
    params
        Application parameters; ``optimization_kwargs`` is read live.
    """

    def __init__(
        self,
        query: QueryService,
        nn: NNService,
        optimization: OptimizationService,
        params: AppParameters,
    ) -> None:
        """
        Initialize analysis use-cases.

        Parameters
        ----------
        query
            Read-only query façade.
        nn
            Segmenter service.
        optimization
            Region optimization service.
        params
            Application parameters.
        """
        self._query = query
        self._nn = nn
        self._optimization = optimization
        self._params = params

    def set_nn(self, nn: NNService) -> None:
        """
        Replace the NN service after parameter reconfiguration.

        Parameters
        ----------
        nn
            Newly constructed NN service.
        """
        self._nn = nn

    def run_segmenter(self, spectrum_ids: Sequence[str]) -> CompositeChange | None:
        """
        Run the segmenter on spectra that do not yet have regions.

        Parameters
        ----------
        spectrum_ids
            Parent spectrum identifiers.

        Returns
        -------
        CompositeChange or None
            Combined CreateRegion/CreateBackground/CreatePeak changes, or None
            if every spectrum already has regions or ``spectrum_ids`` is empty.
        """
        changes: list[CompositeChange] = []
        for spectrum_id in spectrum_ids:
            if self._query.get_regions_ids(spectrum_id):
                continue
            normalized_spectrum = self._query.get_spectrum_dto(spectrum_id, normalized=True)
            original_spectrum = self._query.get_spectrum_dto(spectrum_id, normalized=False)
            changes.append(
                self._nn.run_segmenter(spectrum_id, normalized_spectrum, original_spectrum)
            )

        if not changes:
            return None

        return CompositeChange(changes=changes)

    def _resolve_region_ids(
        self,
        *,
        region_ids: Sequence[str] | None,
        spectrum_ids: Sequence[str] | None,
    ) -> tuple[str, ...]:
        """Resolve ``region_ids`` or flatten regions under ``spectrum_ids``."""
        if region_ids is not None:
            return tuple(region_ids)
        if spectrum_ids is None:
            raise ValueError("region_ids or spectrum_ids must be provided")
        resolved: list[str] = []
        for spectrum_id in spectrum_ids:
            resolved.extend(self._query.get_regions_ids(spectrum_id))
        return tuple(resolved)

    def preview_fit_scope(
        self,
        *,
        region_ids: Sequence[str] | None = None,
        spectrum_ids: Sequence[str] | None = None,
    ) -> FitScopePreview:
        """
        Compute the expression closure for an upcoming optimization.

        Does not run the fitter. Uses all components in the document so
        cross-region / cross-spectrum references resolve.

        Parameters
        ----------
        region_ids
            Regions the user selected to optimize.
        spectrum_ids
            Spectra whose regions to treat as the selection (when
            ``region_ids`` is omitted).

        Returns
        -------
        FitScopePreview
            Selected vs expanded region sets and extra counts for UI confirm.

        Raises
        ------
        ValueError
            If neither ``region_ids`` nor ``spectrum_ids`` is provided.
        """
        selected = self._resolve_region_ids(region_ids=region_ids, spectrum_ids=spectrum_ids)
        components: list[ComponentDTO] = []
        for region_id in self._query.get_all_regions_ids():
            _region, region_components = self._query.get_region_dto_repr(region_id, normalized=True)
            components.extend(region_components)

        expanded = expand_fit_region_ids(selected, components)
        selected_set = set(selected)
        extra_regions = tuple(rid for rid in expanded if rid not in selected_set)
        extra_spectra = {
            self._query.get_parent_id(rid)
            for rid in extra_regions
            if self._query.check_object_exists(rid)
        }
        return FitScopePreview(
            selected_region_ids=selected,
            expanded_region_ids=expanded,
            extra_region_count=len(extra_regions),
            extra_spectrum_count=len(extra_spectra),
            expression_problems=collect_expression_problems(components, region_ids=expanded),
            selected_expression_problems=collect_expression_problems(
                components, region_ids=selected
            ),
        )

    def optimize_regions(
        self,
        *,
        region_ids: Sequence[str] | None = None,
        spectrum_ids: Sequence[str] | None = None,
        expand_linked: bool = True,
        **kwargs,
    ) -> CompositeChange:
        """
        Run optimization and return parameter-update changes.

        When ``expand_linked`` is True (default), expands the requested regions to
        the expression-dependency closure. When False, fits only the resolved
        selection; cross-scope exprs become inactive for that run. The fit
        minimizes the same Poisson chi-squared shown on the error plot,
        using raw counts. Intensity parameters are scaled to order 1 for the
        solver and mapped back to counts before the model is evaluated.
        Default optimization kwargs from AppParameters are merged with explicit
        kwargs; caller values override defaults on conflict.

        Parameters
        ----------
        region_ids
            Identifiers of the regions to optimize.
        spectrum_ids
            Identifiers of the spectra whose regions to optimize.
        expand_linked
            If True, include expression-linked regions outside the selection.
        **kwargs
            Passed to lmfit.minimize; overrides AppParameters.optimization_kwargs.

        Returns
        -------
        CompositeChange
            ``UpdateMultipleParameterValues`` changes for optimized components.

        Raises
        ------
        ValueError
            If neither ``region_ids`` nor ``spectrum_ids`` is provided, or if any
            parameter expression in the effective scope fails to resolve.
        """
        merged = {**self._params.optimization_kwargs, **kwargs}
        preview = self.preview_fit_scope(region_ids=region_ids, spectrum_ids=spectrum_ids)
        if expand_linked:
            target_ids = preview.expanded_region_ids
            problems = preview.expression_problems
        else:
            target_ids = preview.selected_region_ids
            problems = preview.selected_expression_problems
        if problems:
            raise ValueError(format_expression_problems(problems))

        region_reprs: list[tuple[RegionDTO, tuple[ComponentDTO, ...]]] = []
        for region_id in target_ids:
            # Raw counts: the least-squares objective is then the plotted χ².
            region_reprs.append(self._query.get_region_dto_repr(region_id, normalized=False))

        return CompositeChange(
            changes=[self._optimization.optimize_regions(region_reprs, **merged)]
        )
