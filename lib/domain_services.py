from .parametrics import (
    NormalizationContext,
    ModelRegistry,
    BaseBackgroundModel,
    BasePeakModel,
    RuntimeParameter,
)
from .domain import SpectrumCollection, Spectrum, Region, Peak, Background, Component

from typing import Optional, TypeVar
from numpy.typing import NDArray

T = TypeVar("T")


class DomainService:
    """
    Base class for all domain-level services.

    Encapsulates access to a shared :class:`SpectrumCollection` instance
    and provides common protected helpers for retrieving domain objects
    by ID, with or without type checking.

    This class is not intended to be used directly.
    """

    def __init__(self, collection: SpectrumCollection):
        """
        Initialize the service with a spectrum collection.

        Parameters
        ----------
        collection : SpectrumCollection
            Central registry containing all domain objects.
        """
        self.collection = collection

    def _get(self, obj_id: str):
        """
        Retrieve a domain object by ID.

        Parameters
        ----------
        obj_id : str
            Identifier of the domain object.

        Returns
        -------
        DomainObject
            The object registered under the given ID.

        Raises
        ------
        KeyError
            If no object with this ID exists.
        """
        return self.collection.get(obj_id)

    def _get_typed(self, obj_id: str, tp: type[T]) -> T:
        """
        Retrieve a domain object by ID and ensure its type.

        Parameters
        ----------
        obj_id : str
            Identifier of the domain object.
        tp : type
            Expected type of the object.

        Returns
        -------
        T
            The requested object cast to the expected type.

        Raises
        ------
        KeyError
            If no object with the ID exists.
        TypeError
            If the object is not an instance of the requested type.
        """
        return self.collection.get_typed(obj_id, tp)


class CollectionQueryService(DomainService):
    """
    Query-oriented service for navigating a SpectrumCollection.

    This service provides read-oriented access to spectra, regions,
    and components without modifying the collection structure.
    It does not enforce immutability: returned objects are live
    domain instances.
    """

    def get(self, obj_id: str):
        """
        Retrieve any domain object by ID.

        Parameters
        ----------
        obj_id : str
            Identifier of the object.

        Returns
        -------
        DomainObject
            The requested domain object.
        """
        return self._get(obj_id)

    def get_regions(self, spectrum_id: str) -> tuple[Region, ...]:
        """
        Retrieve all regions belonging to a spectrum.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the parent spectrum.

        Returns
        -------
        tuple[Region, ...]
            Regions whose parent is the given spectrum.
        """
        return tuple(obj for obj in self.collection.get_children(spectrum_id) if isinstance(obj, Region))

    def get_components(self, region_id: str) -> tuple[Component, ...]:
        """
        Retrieve all components (peaks and background) in a region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.

        Returns
        -------
        tuple[Component, ...]
            All components belonging to the region.
        """
        return tuple(obj for obj in self.collection.get_children(region_id) if isinstance(obj, Component))

    def get_peaks(self, region_id: str) -> tuple[Peak, ...]:
        """
        Retrieve all peak components in a region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.

        Returns
        -------
        tuple[Peak, ...]
            All peak components in the region.
        """
        return tuple(obj for obj in self.collection.get_children(region_id) if isinstance(obj, Peak))

    def get_background(self, region_id: str) -> Background:
        """
        Retrieve the unique background component of a region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.

        Returns
        -------
        Background
            The background component of the region.

        Raises
        ------
        RuntimeError
            If the region has no background or more than one background.
        """
        bgs = [obj for obj in self.collection.get_children(region_id) if isinstance(obj, Background)]

        if not bgs:
            raise RuntimeError(f"Region {region_id} has no Background")
        if len(bgs) > 1:
            raise RuntimeError(f"Region {region_id} has multiple Backgrounds")

        return bgs[0]

    def get_all_peaks(self) -> tuple[Peak, ...]:
        """
        Retrieve all peak components in the collection.

        Returns
        -------
        tuple[Peak, ...]
            All peaks registered in the collection.
        """
        return tuple(obj for obj in self.collection.objects_index.values() if isinstance(obj, Peak))

    def get_all_spectra(self) -> tuple[Spectrum, ...]:
        """
        Retrieve all spectra in the collection.

        Returns
        -------
        tuple[Spectrum, ...]
            All registered spectra.
        """
        return tuple(obj for obj in self.collection.objects_index.values() if isinstance(obj, Spectrum))

    def get_all_regions(self) -> tuple[Region, ...]:
        """
        Retrieve all regions in the collection.

        Returns
        -------
        tuple[Region, ...]
            All registered regions.
        """
        return tuple(obj for obj in self.collection.objects_index.values() if isinstance(obj, Region))


class SpectrumService(DomainService):
    """
    Service responsible for spectrum lifecycle management.

    This includes creation, data replacement, and removal of spectra.
    """

    def create_spectrum(
        self,
        x: NDArray,
        y: NDArray,
        *,
        spectrum_id: Optional[str] = None,
    ) -> str:
        """
        Create and register a new spectrum.

        Parameters
        ----------
        x : NDArray
            X-axis values.
        y : NDArray
            Y-axis intensity values.
        spectrum_id : str, optional
            Explicit spectrum ID.

        Returns
        -------
        str
            ID of the newly created spectrum.
        """
        spectrum = Spectrum(x=x, y=y, id_=spectrum_id)
        self.collection.add(spectrum)
        return spectrum.id_

    def replace_data(self, spectrum_id: str, x: NDArray, y: NDArray) -> None:
        """
        Replace the numerical data of an existing spectrum.

        This operation updates x/y arrays and recomputes the
        normalization context.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum.
        x : NDArray
            New x-axis data.
        y : NDArray
            New y-axis data.
        """
        spectrum = self._get_typed(spectrum_id, Spectrum)
        spectrum.x = x
        spectrum.y = y
        spectrum.norm_ctx = NormalizationContext.from_array(y)

    def remove_spectrum(self, spectrum_id: str):
        """
        Remove a spectrum and all its dependent objects.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum to remove.
        """
        self.collection.remove(spectrum_id)


class DataQueryService(DomainService):
    """
    Service for accessing numerical spectral data.

    Provides normalized and raw views of spectrum and region data
    without modifying domain state.
    """

    def get_norm_ctx(self, spectrum_id: str) -> NormalizationContext:
        """
        Retrieve the normalization context of a spectrum.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum.

        Returns
        -------
        NormalizationContext
            Normalization parameters derived from spectrum data.
        """
        spectrum = self._get(spectrum_id)
        return spectrum.norm_ctx

    def get_spectrum_data(self, spectrum_id: str, normalized: bool = False) -> tuple[NDArray, NDArray]:
        """
        Retrieve spectrum x/y data.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum.
        normalized : bool, default=False
            Whether to return normalized y-values.

        Returns
        -------
        tuple[NDArray, NDArray]
            X and Y arrays.
        """
        spectrum = self._get_typed(spectrum_id, Spectrum)

        if normalized:
            norm_ctx = spectrum.norm_ctx
            y = (spectrum.y - norm_ctx.offset) / norm_ctx.scale
        else:
            y = spectrum.y

        return spectrum.x, y

    def get_region_data(
        self,
        region_id: str,
        normalized: bool = False,
    ) -> tuple[NDArray, NDArray]:
        """
        Retrieve x/y data corresponding to a region slice.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        normalized : bool, default=False
            Whether to return normalized y-values.

        Returns
        -------
        tuple[NDArray, NDArray]
            Sliced x and y arrays for the region.
        """
        region = self._get_typed(region_id, Region)
        x, y = self.get_spectrum_data(region.parent_id, normalized=normalized)

        sl = region.slice_
        x = x[sl]
        y = y[sl]

        return x, y


class RegionService(DomainService):
    """
    Service responsible for region lifecycle management.

    Regions define index-based subranges of spectra.
    """

    def create_region(
        self, spectrum_id: str, start: int, stop: int, region_id: Optional[str] = None
    ) -> Region:
        """
        Create and register a region bound to a spectrum.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the parent spectrum.
        start : int
            Start index (inclusive).
        stop : int
            Stop index (exclusive).
        region_id : str, optional
            Explicit region identifier.

        Returns
        -------
        str
            ID of the newly created region.

        Raises
        ------
        IndexError
            If indices are outside spectrum bounds.
        ValueError
            If start >= stop.
        """
        spectrum = self._get_typed(spectrum_id, Spectrum)

        if start < 0 or stop > len(spectrum.x):
            raise IndexError("Region indices out of spectrum bounds")
        if start >= stop:
            raise ValueError("start_idx must be < end_idx")

        region_slice = slice(start, stop)

        region = Region(
            slice_=region_slice,
            parent_id=spectrum_id,
            id_=region_id,
        )
        self.collection.add(region)

        return region.id_

    def remove_region(self, region_id: str) -> None:
        """
        Remove a region and all its components.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        """
        self.collection.remove(region_id)

    def update_slice(self, region_id: str, start: int, stop: int) -> None:
        """
        Update the index slice of an existing region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        start : int
            New start index.
        stop : int
            New stop index.

        Raises
        ------
        ValueError
            If the slice definition is invalid.
        """
        region = self._get_typed(region_id, Region)

        if start < 0 or stop <= start:
            raise ValueError("Invalid region slice")

        region.slice_ = slice(start, stop)


class ComponentService(DomainService):
    """
    Service for managing parametric components.

    Handles creation, replacement, removal, and parameter manipulation
    of peak and background components.
    """

    @staticmethod
    def _create_component_obj(
        region_id: str,
        model_name: str,
        parameters: Optional[dict[str, float]] = None,
        component_id: Optional[str] = None,
    ) -> Peak | Background:
        """
        Instantiate a component using a registered parametric model.

        Parameters
        ----------
        region_id : str
            Identifier of the parent region.
        model_name : str
            Name of the registered model.
        parameters : dict[str, float], optional
            Initial parameter values.
        component_id : str, optional
            Explicit component identifier.

        Returns
        -------
        Peak or Background
            Newly created component instance.

        Raises
        ------
        TypeError
            If the model type is unsupported.
        """

        if parameters is None:
            parameters = {}

        model = ModelRegistry.get(model_name)

        init_params = dict(
            model=model,
            region_id=region_id,
            component_id=component_id,
            **parameters,
        )

        if isinstance(model, BaseBackgroundModel):
            obj = Background(**init_params)
        elif isinstance(model, BasePeakModel):
            obj = Peak(**init_params)
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

        return obj

    def create_peak(
        self,
        region_id: str,
        model_name: str,
        parameters: Optional[dict[str, float]] = None,
        peak_id: Optional[str] = None,
    ) -> str:
        """
        Create and register a peak component.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        model_name : str
            Name of the peak model.
        parameters : dict[str, float], optional
            Initial parameter values.
        peak_id : str, optional
            Explicit peak identifier.

        Returns
        -------
        str
            ID of the created peak.
        """
        peak = self._create_component_obj(region_id, model_name, parameters, peak_id)

        if not isinstance(peak, Peak):
            raise ValueError(f"Model {model_name} is not a peak model")

        self.collection.add(peak)

        return peak.id_

    def replace_background(
        self,
        region_id: str,
        model_name: str,
        parameters: Optional[dict[str, float]] = None,
        background_id: Optional[str] = None,
    ) -> str:
        """
        Replace or create the background component of a region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        model_name : str
            Name of the background model.
        parameters : dict[str, float], optional
            Initial parameter values.
        background_id : str, optional
            Explicit background identifier.

        Returns
        -------
        str
            ID of the new background component.

        Raises
        ------
        RuntimeError
            If the region has more than one background.
        ValueError
            If the model is not a background model.
        """

        backgrounds = [obj for obj in self.collection.get_children(region_id) if isinstance(obj, Background)]

        if len(backgrounds) > 1:
            raise RuntimeError(
                f"Region {region_id} must have exactly one Background, " f"found {len(backgrounds)}"
            )

        new_bg = self._create_component_obj(
            region_id=region_id,
            model_name=model_name,
            parameters=parameters,
            component_id=background_id,
        )

        if not isinstance(new_bg, Background):
            raise ValueError(f"Model {model_name} is not a background model")

        if len(backgrounds) == 1:
            old_bg = backgrounds[0]
            self.collection.remove(old_bg.id_)

        self.collection.add(new_bg)

        return new_bg.id_

    def remove_component(self, component_id: str):
        """
        Remove a component from the collection.

        Parameters
        ----------
        component_id : str
            Identifier of the component.
        """
        self.collection.remove(component_id)

    def get_parameter(self, component_id: str, param: str) -> RuntimeParameter:
        """
        Retrieve a runtime parameter of a component.

        Parameters
        ----------
        component_id : str
            Identifier of the component.
        param : str
            Parameter name.

        Returns
        -------
        RuntimeParameter
            Requested parameter object.
        """
        component = self._get_typed(component_id, Component)
        return component.get_param(param)

    def get_parameters(self, component_id: str) -> dict[str, RuntimeParameter]:
        """
        Retrieve all parameters of a component.

        Parameters
        ----------
        component_id : str
            Identifier of the component.

        Returns
        -------
        dict[str, RuntimeParameter]
            Mapping of parameter names to runtime parameters.
        """
        component = self._get_typed(component_id, Component)
        return component.parameters

    def set_parameter(self, component_id: str, param: str, **kwargs: float | str) -> None:
        """
        Update attributes of a single component parameter.

        Parameters
        ----------
        component_id : str
            Identifier of the component.
        param : str
            Parameter name.
        **kwargs
            Parameter attributes to update.
        """
        component = self._get_typed(component_id, Component)
        component.set_param(param, **kwargs)

    def set_parameters(self, component_id: str, parameters: dict[str, float]) -> None:
        """
        Update values of multiple parameters at once.

        Parameters
        ----------
        component_id : str
            Identifier of the component.
        parameters : dict[str, float]
            Mapping of parameter names to new values.
        """
        component = self._get_typed(component_id, Component)
        for name, val in parameters.items():
            component.set_param(name, value=val)

    def copy_parameters(
        self,
        src_cmp_id: str,
        dst_cmp_id: str,
    ) -> None:
        """
        Copy parameter values from one component to another.

        Components must share the same model.

        Parameters
        ----------
        src_cmp_id : str
            Source component ID.
        dst_cmp_id : str
            Destination component ID.

        Raises
        ------
        ValueError
            If component models do not match.
        KeyError
            If a destination parameter is missing.
        """
        src = self._get_typed(src_cmp_id, Component)
        dst = self._get_typed(dst_cmp_id, Component)

        if src.model.name != dst.model.name:
            raise ValueError(
                f"Cannot copy parameters: source model '{src.model.name}' "
                f"does not match destination model '{dst.model.name}'"
            )

        for name, src_par in src.parameters.items():
            if name not in dst.parameters:
                raise KeyError(f"No parameter {name} in component {dst.id_}")
            src_par.copy_with(dst.parameters[name])
