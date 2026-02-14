from dataclasses import dataclass, asdict

from core.collection import CoreCollection
from core.services import CollectionQueryService, DataQueryService, ComponentService
from core.math_models import BaseBackgroundModel, ParametricModelLike, NormalizationContext
from core.types import ParameterLike

from typing import Literal
from core.math_models.base_models import NormalizationLikeFn
from numpy.typing import NDArray


@dataclass(frozen=True)
class ParameterDTO:
    """
    Immutable data transfer object representing a single model parameter.

    Used for normalized and denormalized parameter exchange between
    services without mutating domain state.
    """

    name: str
    value: float
    lower: float
    upper: float
    vary: bool
    expr: str | None


@dataclass(frozen=True)
class BaseDTO:
    """
    Base immutable projection of a core domain object.

    Contains common identity and normalization metadata shared
    by all DTO projections.
    """

    id_: str
    parent_id: str
    normalized: bool


@dataclass(frozen=True)
class ComponentDTO(BaseDTO):
    """
    Immutable projection of a spectral component.

    Encapsulates model metadata and a snapshot of component
    parameters in either normalized or denormalized form.
    """

    parameters: dict[str, ParameterDTO]
    model: ParametricModelLike
    kind: Literal["peak", "background"]


@dataclass(frozen=True)
class RegionDTO(BaseDTO):
    """
    Immutable projection of region numerical data.

    References sliced views of the parent spectrum arrays and
    does not own data independently.
    """

    x: NDArray
    y: NDArray


@dataclass(frozen=True)
class SpectrumDTO(BaseDTO):
    """
    Immutable projection of spectrum numerical data.

    Provides access to raw or normalized spectrum arrays without
    exposing mutable domain objects.
    """

    x: NDArray
    y: NDArray
    parent_id = None


class DTOService:
    """
    Service responsible for constructing and applying immutable DTOs.

    Acts as a boundary between mutable domain objects and
    read-only representations used by evaluation, optimization,
    and UI layers.
    """

    def __init__(self, collection: CoreCollection):
        """
        Initialize DTO service with access to core domain services.

        Parameters
        ----------
        collection : CoreCollection
            Active spectrum collection used as the data source.
        """
        self.query_srv = CollectionQueryService(collection)
        self.comp_srv = ComponentService(collection)
        self.data_srv = DataQueryService(collection)

    @staticmethod
    def _transform(
        params: dict[str, dict[str, float | str | bool]],
        targets: tuple[str, ...],
        fn: NormalizationLikeFn,
        norm_ctx: NormalizationContext,
    ) -> dict[str, dict[str, float | str | bool]]:
        """
        Apply a normalization or denormalization transform to raw parameters.

        Parameters
        ----------
        params : dict[str, dict[str, float | str | bool]]
            Raw parameter dictionaries keyed by parameter name.
        targets : tuple[str, ...]
            Names of parameters subject to transformation.
        fn : NormalizationLikeFn
            Scalar transformation function applied to values and bounds.
        norm_ctx : NormalizationContext
            Normalization context providing offset and scale.

        Returns
        -------
        dict[str, dict[str, float | str | bool]]
            Transformed raw parameter dictionaries.
        """
        for name, pdict in params.items():
            if name in targets:
                pdict["value"] = fn(pdict["value"], norm_ctx)
                pdict["lower"] = fn(pdict["lower"], norm_ctx)
                pdict["upper"] = fn(pdict["upper"], norm_ctx)
        return params

    @staticmethod
    def _params_to_raw(params: dict[str, ParameterLike]) -> dict[str, dict[str, float | str | bool]]:
        """
        Convert parameter-like objects into raw dictionary representations.

        Parameters
        ----------
        params : dict[str, ParameterLike]
            Mapping of parameter names to parameter-like objects.

        Returns
        -------
        dict[str, dict[str, float | str | bool]]
            Raw serializable parameter dictionaries.
        """

        return {k: asdict(v) for k, v in params.items()}

    @staticmethod
    def _raw_params_to_dtos(params: dict[str, dict[str, float | str | bool]]) -> dict[str, ParameterDTO]:
        """
        Convert raw parameter dictionaries into immutable ParameterDTO objects.

        Parameters
        ----------
        params : dict[str, dict[str, float | str | bool]]
            Raw parameter dictionaries.

        Returns
        -------
        dict[str, ParameterDTO]
            Immutable parameter DTOs keyed by parameter name.
        """

        return {k: ParameterDTO(**v) for k, v in params.items()}

    def get_component(self, component_id: str, *, normalize: bool = False):
        """
        Construct an immutable DTO projection of a component.

        Parameters
        ----------
        component_id : str
            Identifier of the component.
        normalize : bool, optional
            If True, parameters are returned in normalized form.

        Returns
        -------
        ComponentDTO
            Immutable component projection with parameters and model metadata.
        """

        core_params = self.comp_srv.get_parameters(component_id)
        parent_id = self.query_srv.get_parent(component_id)
        norm_ctx = self.data_srv.get_norm_ctx(region_id=parent_id)
        model = self.comp_srv.get_model(component_id)

        # get mutable inner projection
        raw_params = self._params_to_raw(core_params)

        if normalize:
            # from denorm to norm values
            raw_params = self._transform(
                raw_params,
                model.normalization_target_parameters,
                model.normalize_value,  # from raw values to norm
                norm_ctx,
            )

        params = self._raw_params_to_dtos(raw_params)

        return ComponentDTO(
            id_=component_id,
            parent_id=parent_id,
            normalized=normalize,
            parameters=params,
            model=model,
            kind="background" if isinstance(model, BaseBackgroundModel) else "peak",
        )

    def get_region(self, region_id: str, *, normalize: bool = False) -> RegionDTO:
        """
        Construct an immutable DTO projection of a region's numerical data.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        normalize : bool, optional
            If True, spectrum data is returned in normalized form.

        Returns
        -------
        RegionDTO
            Immutable region data projection.
        """
        # DataQueryService return views of the original arrays
        # set arr.flags.writeable = False
        # to prevent modifying
        x, y = self.data_srv.get_region_data(region_id, normalized=normalize)
        x.flags.writeable = False
        y.flags.writeable = False
        parent_id = self.query_srv.get_parent(region_id)

        return RegionDTO(
            id_=region_id,
            parent_id=parent_id,
            normalized=normalize,
            x=x,
            y=y,
        )

    def get_region_repr(
        self, region_id: str, *, normalize: bool = False
    ) -> tuple[RegionDTO, tuple[ComponentDTO, ...]]:
        """
        Construct a complete immutable representation of a region.

        Includes the region DTO and all associated component DTOs.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        normalize : bool, optional
            If True, data and parameters are normalized.

        Returns
        -------
        tuple[RegionDTO, tuple[ComponentDTO, ...]]
            Region DTO and its component DTOs.
        """

        reg_dto = self.get_region(region_id, normalize=normalize)
        cmp_dtos = tuple(
            self.get_component(cid, normalize=normalize) for cid in self.query_srv.get_components(region_id)
        )
        return reg_dto, cmp_dtos

    def get_spectrum(self, spectrum_id: str, *, normalize: bool = False) -> SpectrumDTO:
        """
        Construct an immutable DTO projection of a spectrum's numerical data.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum.
        normalize : bool, optional
            If True, spectrum data is returned in normalized form.

        Returns
        -------
        SpectrumDTO
            Immutable spectrum data projection.
        """

        x, y = self.data_srv.get_spectrum_data(spectrum_id, normalized=normalize)
        x.flags.writeable = False
        y.flags.writeable = False
        parent_id = self.query_srv.get_parent(spectrum_id)

        return SpectrumDTO(
            id_=spectrum_id,
            parent_id=parent_id,
            normalized=normalize,
            x=x,
            y=y,
        )

    def get_spectrum_repr(
        self, spectrum_id: str, *, normalize: bool = False
    ) -> tuple[SpectrumDTO, tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...]]:
        """
        Construct a complete immutable representation of a spectrum.

        Includes the spectrum DTO, all regions, and their components.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum.
        normalize : bool, optional
            If True, all numerical data and parameters are normalized.

        Returns
        -------
        tuple[
            SpectrumDTO,
            tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...]
        ]
            Full immutable spectrum representation.
        """

        spectrum_dto = self.get_spectrum(spectrum_id, normalize=normalize)
        reg_reprs = tuple(
            self.get_region_repr(rid, normalize=normalize) for rid in self.query_srv.get_regions(spectrum_id)
        )
        return spectrum_dto, reg_reprs
