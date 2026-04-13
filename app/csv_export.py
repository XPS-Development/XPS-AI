"""
App-level CSV export service for DTO projections.
"""

from pathlib import Path

from core.types import ComponentLike
from tools.csv_export import SpectrumRepr, export_spectrum_csv, export_spectrum_peak_parameters_csv


class CSVExportService:
    """
    Service that exports DTO projections into CSV-like files.
    """

    def export_spectrum_peak_parameters(
        self,
        path: str | Path,
        components: tuple[ComponentLike, ...],
        *,
        separator: str = ",",
        use_xps_peak_names: bool = False,
        precision: int | None = None,
    ) -> None:
        """
        Export a peak component's parameters to a CSV-like file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        components : tuple[ComponentLike, ...]
            Components to export as peak-parameter table.
        separator : str, optional
            Column separator character.
        use_xps_peak_names : bool, optional
            If True, apply pseudo-voigt XPS aliases.
        """
        export_spectrum_peak_parameters_csv(
            path=path,
            components=components,
            separator=separator,
            use_xps_peak_names=use_xps_peak_names,
            precision=precision,
        )

    def export_spectrum(
        self,
        path: str | Path,
        spectrum_repr: SpectrumRepr,
        *,
        separator: str = ",",
        include_evaluated_components: bool = False,
        include_background: bool = True,
        include_difference: bool = True,
        precision: int | None = None,
    ) -> None:
        """
        Export a spectrum representation to a CSV-like file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        spectrum_repr : tuple[SpectrumDTO, tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...]]
            Spectrum representation from the query service.
        separator : str, optional
            Column separator character.
        include_evaluated_components : bool, optional
            If True, include evaluated model columns.
        include_background : bool, optional
            If True, include background model column.
        include_difference : bool, optional
            If True, include residual/difference column.
        """
        export_spectrum_csv(
            path=path,
            spectrum_repr=spectrum_repr,
            separator=separator,
            include_evaluated_components=include_evaluated_components,
            include_background=include_background,
            include_difference=include_difference,
            precision=precision,
        )
