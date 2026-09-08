"""
Analysis use-cases: NN segmentation and region optimization.

Builds Change objects from query state and app services. Does not execute
commands. ``auto_fit`` remains two orchestrator executions so optimization
sees regions created by the segmenter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.command.changes import CompositeChange

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.nn_service import NNService
    from app.optimization import OptimizationService
    from app.parameters import AppParameters
    from app.query_service import QueryService
    from core.dto import ComponentDTO, RegionDTO


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

    def optimize_regions(
        self,
        *,
        region_ids: Sequence[str] | None = None,
        spectrum_ids: Sequence[str] | None = None,
        **kwargs,
    ) -> CompositeChange:
        """
        Run optimization and return parameter-update changes.

        Default optimization kwargs from AppParameters are merged with explicit
        kwargs; caller values override defaults on conflict.

        Parameters
        ----------
        region_ids
            Identifiers of the regions to optimize.
        spectrum_ids
            Identifiers of the spectra whose regions to optimize.
        **kwargs
            Passed to lmfit.minimize; overrides AppParameters.optimization_kwargs.

        Returns
        -------
        CompositeChange
            ``UpdateMultipleParameterValues`` changes for optimized components.

        Raises
        ------
        ValueError
            If neither ``region_ids`` nor ``spectrum_ids`` is provided.
        """
        merged = {**self._params.optimization_kwargs, **kwargs}

        if region_ids is None:
            if spectrum_ids is None:
                raise ValueError("region_ids or spectrum_ids must be provided")
            resolved: list[str] = []
            for spectrum_id in spectrum_ids:
                resolved.extend(self._query.get_regions_ids(spectrum_id))
            region_ids = resolved

        region_reprs: list[tuple[RegionDTO, tuple[ComponentDTO, ...]]] = []
        for region_id in region_ids:
            region_reprs.append(self._query.get_region_dto_repr(region_id, normalized=True))

        return CompositeChange(
            changes=[self._optimization.optimize_regions(region_reprs, **merged)]
        )
