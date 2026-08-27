"""Construct immutable DTO projections from mutable core domain state."""

from typing import cast

from core.dto import ComponentDTO, ParameterDTO, RegionDTO, SpectrumDTO
from core.math_models import BaseBackgroundModel
from core.services import CoreContext


class DTOService:
    """
    Service responsible for constructing immutable DTOs.

    Acts as a boundary between mutable domain objects and
    read-only representations used by evaluation, optimization,
    and UI layers.
    """

    def __init__(self, ctx: CoreContext) -> None:
        """
        Initialize DTO service with access to core domain services.

        Parameters
        ----------
        ctx : CoreContext
            Active core context used as the data source.
        """
        self.query_srv = ctx.query
        self.comp_srv = ctx.component
        self.data_srv = ctx.data

    def get_component(self, component_id: str, *, normalized: bool = False) -> ComponentDTO:
        """
        Construct an immutable DTO projection of a component.

        Parameters
        ----------
        component_id : str
            Identifier of the component.
        normalized : bool, optional
            If True, parameters are returned in normalized form.

        Returns
        -------
        ComponentDTO
            Immutable component projection with parameters and model metadata.
        """
        core_params = self.comp_srv.get_parameters(component_id, normalized=normalized)
        model = self.comp_srv.get_model(component_id)

        # `core_params` comes from `asdict(RuntimeParameter)` and `ty` sees it as a
        # wide union. Construct DTO fields explicitly with numeric casts.
        params: dict[str, ParameterDTO] = {}
        for pname, raw in core_params.items():
            value = float(raw["value"])
            lower = float(raw["lower"])
            upper = float(raw["upper"])
            vary = bool(cast(bool, raw["vary"]))
            expr = cast(str | None, raw.get("expr"))

            params[pname] = ParameterDTO(
                name=pname,
                value=value,
                lower=lower,
                upper=upper,
                vary=vary,
                expr=expr,
            )

        return ComponentDTO(
            id_=component_id,
            parent_id=self.query_srv.get_parent(component_id),
            normalized=normalized,
            parameters=params,
            model=model,
            kind="background" if isinstance(model, BaseBackgroundModel) else "peak",
        )

    def get_region(self, region_id: str, *, normalized: bool = False) -> RegionDTO:
        """
        Construct an immutable DTO projection of a region's numerical data.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        normalized : bool, optional
            If True, spectrum data is returned in normalized form.

        Returns
        -------
        RegionDTO
            Immutable region data projection.
        """
        # DataQueryService return views of the original arrays
        # set arr.flags.writeable = False
        # to prevent modifying
        x, y = self.data_srv.get_region_data(region_id, normalized=normalized)
        x.flags.writeable = False
        y.flags.writeable = False
        parent_id = self.query_srv.get_parent(region_id)

        return RegionDTO(
            id_=region_id,
            parent_id=parent_id,
            normalized=normalized,
            x=x,
            y=y,
        )

    def get_region_repr(
        self, region_id: str, *, normalized: bool = False
    ) -> tuple[RegionDTO, tuple[ComponentDTO, ...]]:
        """
        Construct a complete immutable representation of a region.

        Includes the region DTO and all associated component DTOs.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        normalized : bool, optional
            If True, data and parameters are normalized.

        Returns
        -------
        tuple[RegionDTO, tuple[ComponentDTO, ...]]
            Region DTO and its component DTOs.
        """
        reg_dto = self.get_region(region_id, normalized=normalized)
        cmp_dtos = tuple(
            self.get_component(cid, normalized=normalized)
            for cid in self.query_srv.get_components(region_id)
        )
        return reg_dto, cmp_dtos

    def get_spectrum(self, spectrum_id: str, *, normalized: bool = False) -> SpectrumDTO:
        """
        Construct an immutable DTO projection of a spectrum's numerical data.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum.
        normalized : bool, optional
            If True, spectrum data is returned in normalized form.

        Returns
        -------
        SpectrumDTO
            Immutable spectrum data projection.
        """
        x, y = self.data_srv.get_spectrum_data(spectrum_id, normalized=normalized)
        x.flags.writeable = False
        y.flags.writeable = False

        return SpectrumDTO(
            id_=spectrum_id,
            parent_id=None,
            normalized=normalized,
            x=x,
            y=y,
        )

    def get_spectrum_repr(
        self, spectrum_id: str, *, normalized: bool = False
    ) -> tuple[SpectrumDTO, tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...]]:
        """
        Construct a complete immutable representation of a spectrum.

        Includes the spectrum DTO, all regions, and their components.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum.
        normalized : bool, optional
            If True, all numerical data and parameters are normalized.

        Returns
        -------
        tuple[
            SpectrumDTO,
            tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...]
        ]
            Full immutable spectrum representation.
        """
        spectrum_dto = self.get_spectrum(spectrum_id, normalized=normalized)
        reg_reprs = tuple(
            self.get_region_repr(rid, normalized=normalized)
            for rid in self.query_srv.get_regions(spectrum_id)
        )
        return spectrum_dto, reg_reprs
