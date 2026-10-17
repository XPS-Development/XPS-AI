"""Export use-cases: CSV projections of spectra and peak parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from app.csv_export import CSVExportService
    from app.query_service import QueryService
    from core.post_processing.tables import TableView


class ExportUseCases:
    """Build CSV exports from query DTOs via :class:`CSVExportService`."""

    def __init__(self, query: QueryService, csv_export: CSVExportService) -> None:
        """
        Initialize export use-cases.

        Parameters
        ----------
        query
            Read-only query façade for spectrum and component DTOs.
        csv_export
            App-level CSV export adapter.
        """
        self._query = query
        self._csv_export = csv_export

    def export_post_processing(
        self,
        series_id: str,
        path: str | Path,
        *,
        view: TableView = "components",
        separator: str = ",",
        precision: int | None = None,
    ) -> None:
        """Export a current conflict-free snapshot using the same query projection as the UI."""
        dto = self._query.get_post_processing_series(series_id)
        if not dto.exportable:
            raise ValueError("Recalculate the series and resolve conflicts before exporting")
        table = self._query.get_post_processing_table(series_id, view)
        self._csv_export.export_post_processing(
            path, table, dto.series, separator=separator, precision=precision
        )

    def export_analytical_table(
        self, table_id: str, path: str | Path, *, precision: int | None = None
    ) -> None:
        """Reject stale dependencies, then export values and complete provenance."""
        dto = self._query.get_analytical_table(table_id)
        if not dto.exportable:
            raise ValueError("Update alignment and resolve source conflicts before exporting")
        sources = {
            s.series_id: self._query.get_post_processing_series(s.series_id).series
            for s in dto.state.sources
        }
        self._csv_export.export_analytical_table(
            path, dto.table, dto.state, sources, precision=precision
        )

    def export_peak_parameters(
        self,
        spectrum_id: str,
        path: str | Path,
        *,
        normalized: bool = False,
        separator: str = ",",
        use_xps_peak_names: bool = False,
        precision: int | None = None,
    ) -> None:
        """
        Export peak parameters for all peaks in a spectrum to a CSV-like file.

        Parameters
        ----------
        spectrum_id
            Identifier of the spectrum whose peaks are exported.
        path
            Output file path.
        normalized
            If True, export normalized parameters.
        separator
            Column separator character.
        use_xps_peak_names
            If True, apply pseudo-voigt XPS aliases.
        precision
            Optional float formatting precision.
        """
        components = []
        for region_id in self._query.get_regions_ids(spectrum_id):
            for peak_id in self._query.get_peaks_ids(region_id):
                components.append(self._query.get_component_dto(peak_id, normalized=normalized))
        self._csv_export.export_spectrum_peak_parameters(
            path,
            tuple(components),
            separator=separator,
            use_xps_peak_names=use_xps_peak_names,
            precision=precision,
        )

    def export_spectrum(
        self,
        spectrum_id: str,
        path: str | Path,
        *,
        normalized: bool = False,
        separator: str = ",",
        include_evaluated_components: bool = False,
        include_background: bool = True,
        include_difference: bool = True,
        precision: int | None = None,
    ) -> None:
        """
        Export a full spectrum DTO representation to a CSV-like file.

        Parameters
        ----------
        spectrum_id
            Identifier of the spectrum to export.
        path
            Output file path.
        normalized
            If True, export normalized spectrum data.
        separator
            Column separator character.
        include_evaluated_components
            If True, include evaluated model columns.
        include_background
            If True, include background model column.
        include_difference
            If True, include residual/difference column.
        precision
            Optional float formatting precision.
        """
        spectrum_repr = self._query.get_spectrum_dto_repr(spectrum_id, normalized=normalized)
        self._csv_export.export_spectrum(
            path,
            spectrum_repr,
            separator=separator,
            include_evaluated_components=include_evaluated_components,
            include_background=include_background,
            include_difference=include_difference,
            precision=precision,
        )
