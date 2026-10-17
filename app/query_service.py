"""Read-only query façade over core context and DTO projections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from core.dto import AnalyticalTableDTO
from core.evaluation import (
    SpectrumPlotData,
    peak_area,
    plot_data_from_evaluation,
    region_area,
    spectrum_bundle,
)
from core.math_models import ModelRegistry
from core.post_processing.analysis import build_analysis, dependency_stamp
from core.post_processing.fingerprint import input_fingerprint
from core.post_processing.tables import TableView, build_table, build_trend_tables

from .dto_service import DTOService
from .post_processing import METRIC_OPERATIONS, MetricOperationOption, project_series

if TYPE_CHECKING:
    from core.dto import (
        AnalysisTable,
        ComponentDTO,
        PostProcessingSeriesDTO,
        RegionDTO,
        SpectrumDTO,
    )
    from core.math_models.base_models import ParameterSpec
    from core.metadata import Metadata
    from core.post_processing.state import PostProcessingSeries
    from core.services import CoreContext


class QueryService:
    """Thin wrapper for querying the collection, metadata, DTOs, and model catalog."""

    def __init__(self, ctx: CoreContext) -> None:
        """
        Initialize query service with access to core services and DTOs.

        Parameters
        ----------
        ctx : CoreContext
            Core services context providing query, metadata and data access.
        """
        self._ctx = ctx
        self._dto = DTOService(ctx)

    @property
    def document_generation(self) -> str:
        """Return the lifecycle token used to reject jobs from another document."""
        return self._ctx.query.collection.generation

    def get_post_processing_fingerprint(self, series: PostProcessingSeries) -> str:
        """Return the current version of all decomposition inputs for a series."""
        return input_fingerprint(self._ctx.query.collection, series)

    def get_post_processing_series(self, series_id: str) -> PostProcessingSeriesDTO:
        """Project a saved series without exposing mutable document stores."""
        return project_series(self, self._ctx.post_processing[series_id])

    def get_post_processing_series_ids(self) -> tuple[str, ...]:
        """Return saved series IDs in document order."""
        return tuple(self._ctx.post_processing)

    def get_analytical_table_ids(self) -> tuple[str, ...]:
        """Return independent analytical tables in document order."""
        return tuple(self._ctx.analytical_tables)

    def get_matching_settings(self, series_id: str) -> tuple[tuple[str, float], ...]:
        """Return active editable matching parameters without exposing model classes."""
        from dataclasses import asdict

        return tuple(asdict(self._ctx.post_processing[series_id].settings).items())

    def get_trend_tables(
        self, series_id: str, track_id: str
    ) -> tuple[AnalysisTable, AnalysisTable]:
        """Return selected-trend components and its numerical preview."""
        series = self._ctx.post_processing[series_id]
        names = {
            sid: getattr(self.get_metadata(sid), "name", None) or sid for sid in series.spectrum_ids
        }
        return build_trend_tables(series, names, track_id)

    def get_analytical_source_labels(self, table_id: str) -> tuple[str, ...]:
        """Describe qualified sources and their independently counted peaks and steps."""
        labels = []
        for source in self._ctx.analytical_tables[table_id].sources:
            series = self._ctx.post_processing.get(source.series_id)
            if series is None:
                labels.append(f"Unavailable series: {source.series_id}")
                continue
            track = (
                None
                if series.result is None
                else next((t for t in series.result.tracks if t.id_ == source.track_id), None)
            )
            labels.append(
                f"{series.name} / {dict(series.track_names).get(source.anchor_id, source.track_id)}: "
                f"{0 if track is None else len(track.component_ids)} peaks / {len(series.spectrum_ids)} steps"
            )
        return tuple(labels)

    def get_used_trend_anchors(self, series_id: str) -> set[str]:
        """Return anchors whose identity must survive a series recalculation."""
        return {
            source.anchor_id
            for table in self._ctx.analytical_tables.values()
            for source in table.sources
            if source.series_id == series_id
        }

    def get_trend_color_keys(self, series_id: str) -> dict[str, str]:
        """Return common colour identities for a series and source-spectrum navigation."""
        result = self._ctx.post_processing[series_id].result
        return (
            {}
            if result is None
            else {cid: track.anchor_id for track in result.tracks for cid in track.component_ids}
        )

    def get_analytical_table(self, table_id: str) -> AnalyticalTableDTO:
        """Calculate aligned cross-series values and expose stale dependency conflicts."""
        state = self._ctx.analytical_tables[table_id]
        saved = self._ctx.post_processing
        stamps = dict(state.dependencies)
        stale = False
        conflicts = []
        counts = []
        for source in state.sources:
            series = saved.get(source.series_id)
            if series is None:
                conflicts.append(f"unavailable_series:{source.series_id}")
                counts.append((source.id_, 0, 0))
                continue
            dto = self.get_post_processing_series(source.series_id)
            stale |= dto.stale or stamps.get(source.series_id) != dependency_stamp(series)
            track = (
                None
                if series.result is None
                else next(
                    (
                        t
                        for t in series.result.tracks
                        if t.id_ == source.track_id and t.anchor_id == source.anchor_id
                    ),
                    None,
                )
            )
            if track is None:
                conflicts.append(f"operand_selection_required:{source.id_}")
            counts.append(
                (
                    f"{series.name} / {dict(series.track_names).get(source.anchor_id, source.track_id)}",
                    len(series.spectrum_ids),
                    0 if track is None else len(track.component_ids),
                )
            )
        row_ids = {row.id_ for row in state.rows}
        source_ids = {source.id_ for source in state.sources}
        for metric in state.metrics:
            if not set((*metric.tracks, *metric.denominator)).issubset(source_ids):
                conflicts.append(f"operand_selection_required:{metric.id_}")
            if metric.reference_row_id is not None and metric.reference_row_id not in row_ids:
                conflicts.append(f"reference_selection_required:{metric.id_}")
        warnings = ()
        if len({(length, peaks) for _, length, peaks in counts}) > 1:
            warnings = (
                "Unequal source counts (steps / peaks): "
                + "; ".join(
                    f"{source_id}: {length} / {peaks}" for source_id, length, peaks in counts
                ),
            )
        names = {
            sid: getattr(self.get_metadata(sid), "name", None) or sid
            for row in state.rows
            for point in row.points
            if (sid := point.spectrum_id)
        }
        projection = build_analysis(state, saved, names, set(self.get_all_spectra_ids()))
        return AnalyticalTableDTO(state, projection, stale, tuple(conflicts), warnings)

    def get_post_processing_table(
        self, series_id: str, view: TableView = "components"
    ) -> AnalysisTable:
        """Return shared full precision values for a table, trends or export."""
        series = self._ctx.post_processing[series_id]
        snapshot_ids = (
            ()
            if series.result is None
            else tuple(s.spectrum_id for s in series.result.series.steps)
        )
        names = {sid: getattr(self.get_metadata(sid), "name", None) or sid for sid in snapshot_ids}
        return build_table(series, names, view)

    def get_post_processing_operations(self) -> tuple[MetricOperationOption, ...]:
        """Return supported operation labels and operand requirements."""
        return METRIC_OPERATIONS

    def get_post_processing_source(
        self,
        series_id: str,
        spectrum_id: str,
        track_id: str | None = None,
        *,
        component_id: str | None = None,
    ) -> tuple[str, str | None, str | None] | None:
        """Resolve a table/trend point to its live spectrum and optional peak."""
        if spectrum_id not in self.get_all_spectra_ids():
            return None
        result = self._ctx.post_processing[series_id].result
        if result is not None and track_id is not None:
            track = next((track for track in result.tracks if track.id_ == track_id), None)
            step = next(
                (step for step in result.series.steps if step.spectrum_id == spectrum_id), None
            )
            if track is not None and step is not None:
                peak = next(
                    (peak for peak in step.peaks if peak.component_id in track.component_ids), None
                )
                if peak is not None:
                    component_id = peak.component_id
        if component_id in self.get_all_peaks_ids():
            region_id = self.get_parent_id(component_id)
            if self.get_parent_id(region_id) == spectrum_id:
                return spectrum_id, region_id, component_id
        return spectrum_id, None, None

    # ---- Collection queries (read-only) ----

    def check_object_exists(self, obj_id: str) -> bool:
        """
        Return True if an object with the given ID exists in the collection.

        Parameters
        ----------
        obj_id : str
            Identifier of the core object.

        Returns
        -------
        bool
            True if the object exists, False otherwise.
        """
        return self._ctx.query.check_object_exists(obj_id)

    def get_parent_id(self, obj_id: str) -> str:
        """
        Return the identifier of the first parent of a core object.

        Parameters
        ----------
        obj_id : str
            Identifier of the core object.

        Returns
        -------
        str
            Identifier of the object's parent.
        """
        return self._ctx.query.get_parent(obj_id)

    def get_subtree_ids(self, obj_id: str) -> tuple[str, ...]:
        """
        Return the identifier of the object and all its descendants.

        Parameters
        ----------
        obj_id : str
            Root object identifier.

        Returns
        -------
        tuple[str, ...]
            The object ID and all descendant IDs.
        """
        return self._ctx.query.get_subtree(obj_id)

    def get_regions_ids(self, spectrum_id: str) -> tuple[str, ...]:
        """
        Return identifiers of all regions that belong to a spectrum.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the parent spectrum.

        Returns
        -------
        tuple[str, ...]
            Region identifiers whose parent is the given spectrum.
        """
        return self._ctx.query.get_regions(spectrum_id)

    def get_components_ids(self, region_id: str) -> tuple[str, ...]:
        """
        Return identifiers of all components (peaks and background) in a region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.

        Returns
        -------
        tuple[str, ...]
            Component identifiers belonging to the region.
        """
        return self._ctx.query.get_components(region_id)

    def get_peaks_ids(self, region_id: str) -> tuple[str, ...]:
        """
        Return identifiers of all peak components in a region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.

        Returns
        -------
        tuple[str, ...]
            Peak identifiers belonging to the region.
        """
        return self._ctx.query.get_peaks(region_id)

    def get_spectrum_structure_status(
        self,
        spectrum_id: str,
    ) -> Literal["empty", "regions", "peaks"]:
        """
        Return a coarse structural status for status-dot UI.

        Parameters
        ----------
        spectrum_id : str
            Spectrum identifier.

        Returns
        -------
        {"empty", "regions", "peaks"}
            ``empty`` — no regions; ``regions`` — regions but no peaks;
            ``peaks`` — at least one peak exists.
        """
        region_ids = self.get_regions_ids(spectrum_id)
        if not region_ids:
            return "empty"
        for region_id in region_ids:
            if self.get_peaks_ids(region_id):
                return "peaks"
        return "regions"

    def get_background_id(self, region_id: str) -> str | None:
        """
        Return the identifier of the unique background component in a region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.

        Returns
        -------
        str or None
            Background identifier, or None if no background exists.
        """
        return self._ctx.query.get_background(region_id)

    def get_all_peaks_ids(self) -> tuple[str, ...]:
        """
        Return identifiers of all peaks in the collection.

        Returns
        -------
        tuple[str, ...]
            All peak identifiers registered in the collection.
        """
        return self._ctx.query.get_all_peaks()

    def get_all_spectra_ids(self) -> tuple[str, ...]:
        """
        Return identifiers of all spectra in the collection.

        Returns
        -------
        tuple[str, ...]
            All spectrum identifiers registered in the collection.
        """
        return self._ctx.query.get_all_spectra()

    def get_all_regions_ids(self) -> tuple[str, ...]:
        """
        Return identifiers of all regions in the collection.

        Returns
        -------
        tuple[str, ...]
            All region identifiers registered in the collection.
        """
        return self._ctx.query.get_all_regions()

    # ---- Metadata queries (read-only) ----

    def get_metadata(self, obj_id: str) -> Metadata | None:
        """
        Retrieve metadata for a core object.

        Parameters
        ----------
        obj_id : str
            Identifier of the object.

        Returns
        -------
        Metadata or None
            Stored metadata, if any.
        """
        return self._ctx.metadata.get_metadata(obj_id)

    def find_objects(
        self,
        md_field: str,
        md_value: str,
        *,
        match_exact: bool = False,
        tp: type[Metadata] | None = None,
    ) -> tuple[str, ...]:
        """
        Find object identifiers whose metadata matches the given field and value.

        Parameters
        ----------
        md_field : str
            Metadata field to match.
        md_value : str
            Metadata value to match.
        match_exact : bool, default=False
            If True, match exact value, otherwise perform a substring match.
        tp : type[Metadata] or None, optional
            Metadata type to filter by.

        Returns
        -------
        tuple[str, ...]
            Object identifiers with matching metadata.
        """
        return self._ctx.metadata.find_objects(
            md_field=md_field,
            md_value=md_value,
            match_exact=match_exact,
            tp=tp,
        )

    # ---- Region queries ----

    def get_region_slice(
        self, region_id: str, mode: Literal["value", "index"] = "index"
    ) -> tuple[int | float, int | float]:
        """
        Return the start and stop values or indices of a region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        mode : Literal["value", "index"], default="index"
            Mode of the slice retrieval.

        Returns
        -------
        tuple[int | float, int | float]
            Start and stop values or indices of the region.
        """
        return self._ctx.region.get_slice(region_id, mode=mode)

    # ---- Model catalog (read-only) ----

    def get_peak_model_names(self) -> list[str]:
        """Return registered peak model names for UI selection."""
        return ModelRegistry.get_peak_model_names()

    def get_background_model_names(self) -> list[str]:
        """Return registered background model names for UI selection."""
        return ModelRegistry.get_background_model_names()

    def get_model_parameter_schema(self, model_name: str) -> tuple[ParameterSpec, ...]:
        """
        Return parameter schema entries for a registered model.

        Parameters
        ----------
        model_name : str
            Registered model name.

        Returns
        -------
        tuple[ParameterSpec, ...]
            Parameter specifications for the model.
        """
        return tuple(ModelRegistry.get(model_name).parameter_schema)

    # ---- DTO projections ----

    def get_spectrum_dto(self, spectrum_id: str, *, normalized: bool = False) -> SpectrumDTO:
        """
        Return an immutable DTO projection of a spectrum.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum.
        normalized : bool, default=False
            If True, return normalized spectrum data.

        Returns
        -------
        SpectrumDTO
            Spectrum data transfer object.
        """
        return self._dto.get_spectrum(spectrum_id, normalized=normalized)

    def get_region_dto(self, region_id: str, *, normalized: bool = False) -> RegionDTO:
        """
        Return an immutable DTO projection of a region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        normalized : bool, default=False
            If True, return normalized region data.

        Returns
        -------
        RegionDTO
            Region data transfer object.
        """
        return self._dto.get_region(region_id, normalized=normalized)

    def get_component_dto(self, component_id: str, *, normalized: bool = False) -> ComponentDTO:
        """
        Return an immutable DTO projection of a component.

        Parameters
        ----------
        component_id : str
            Identifier of the component.
        normalized : bool, default=False
            If True, return normalized component parameters.

        Returns
        -------
        ComponentDTO
            Component data transfer object.
        """
        return self._dto.get_component(component_id, normalized=normalized)

    def get_peak_area(self, component_id: str) -> float | None:
        """
        Return the fitted area of a peak, or None for a background.

        Parameters
        ----------
        component_id : str
            Identifier of the component.

        Returns
        -------
        float or None
            Fitted peak area from the component model, or None when the
            component does not define one.
        """
        return peak_area(self.get_component_dto(component_id, normalized=False))

    def get_region_area(self, region_id: str) -> float:
        """
        Return the sum of fitted peak areas in a region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.

        Returns
        -------
        float
            Sum of peak areas. Background is excluded; an empty region is 0.
        """
        _region, components = self.get_region_dto_repr(region_id, normalized=False)
        return region_area(components)

    def get_spectrum_dto_repr(
        self,
        spectrum_id: str,
        *,
        normalized: bool = False,
    ) -> tuple[SpectrumDTO, tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...]]:
        """
        Return a complete immutable representation of a spectrum.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum.
        normalized : bool, default=False
            If True, return normalized data and parameters.

        Returns
        -------
        tuple[
            SpectrumDTO,
            tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...],
        ]
            Spectrum DTO and all regions with their component DTOs.
        """
        return self._dto.get_spectrum_repr(spectrum_id, normalized=normalized)

    def get_region_dto_repr(
        self,
        region_id: str,
        *,
        normalized: bool = False,
    ) -> tuple[RegionDTO, tuple[ComponentDTO, ...]]:
        """
        Return a complete immutable representation of a region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        normalized : bool, default=False
            If True, return normalized data and parameters.

        Returns
        -------
        tuple[RegionDTO, tuple[ComponentDTO, ...]]
            Region DTO and its component DTOs.
        """
        return self._dto.get_region_repr(region_id, normalized=normalized)

    def get_spectrum_plot_data(
        self,
        spectrum_id: str,
        *,
        normalized: bool = False,
    ) -> SpectrumPlotData:
        """
        Evaluate a spectrum and build display-ready plot curves.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum.
        normalized : bool, default=False
            If True, evaluate using normalized data and parameters.

        Returns
        -------
        SpectrumPlotData
            Precomputed curves and optional residuals y-range for plotting.
        """
        spectrum, regions = self.get_spectrum_dto_repr(spectrum_id, normalized=normalized)
        result = spectrum_bundle(spectrum, regions, include_background=True)
        return plot_data_from_evaluation(result)
