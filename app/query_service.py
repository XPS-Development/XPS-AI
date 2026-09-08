"""Read-only query façade over core context and DTO projections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from core.evaluation import SpectrumPlotData, plot_data_from_evaluation, spectrum_bundle
from core.math_models import ModelRegistry

from .dto_service import DTOService

if TYPE_CHECKING:
    from core.dto import ComponentDTO, RegionDTO, SpectrumDTO
    from core.math_models.base_models import ParameterSpec
    from core.metadata import Metadata
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
