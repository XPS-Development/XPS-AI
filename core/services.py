from dataclasses import dataclass
from core.math_models import (
    NormalizationContext,
    ModelRegistry,
    BaseBackgroundModel,
    BasePeakModel,
    ParametricModelLike,
)

from .objects import Spectrum, Region, Peak, Background, Component, RuntimeParameter, CoreObject
from .collection import CoreCollection
from .metadata import Metadata

from typing import Optional, TypeVar
from numpy.typing import NDArray

T = TypeVar("T")


class BaseCoreService:
    """
    Base class for all core-level services.

    Encapsulates access to a shared :class:`SpectrumCollection` instance
    and provides common protected helpers for retrieving core objects
    by ID, with or without type checking.

    This class is not intended to be used directly.
    """

    def __init__(self, collection: CoreCollection):
        """
        Initialize the service with a spectrum collection.

        Parameters
        ----------
        collection : SpectrumCollection
            Central registry containing all core objects.
        """
        self.collection = collection

    def _get(self, obj_id: str):
        """
        Retrieve a core object by ID.

        Parameters
        ----------
        obj_id : str
            Identifier of the core object.

        Returns
        -------
        CoreObject
            The object registered under the given ID.

        Raises
        ------
        KeyError
            If no object with this ID exists.
        """
        return self.collection.get(obj_id)

    def _get_typed(self, obj_id: str, tp: type[T]) -> T:
        """
        Retrieve a core object by ID and ensure its type.

        Parameters
        ----------
        obj_id : str
            Identifier of the core object.
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

    def attach(self, obj: CoreObject) -> None:
        """
        Attach an object to the collection.

        Parameters
        ----------
        obj : CoreObject
            The object to attach.
        """
        self.collection.add(obj)

    def detach(self, obj: CoreObject | str) -> list[CoreObject]:
        """
        Detach an object from the collection and return the detached objects.

        Parameters
        ----------
        obj : CoreObject or str
            The object to detach.

        Returns
        -------
        list[CoreObject]
            The detached objects.
        """
        return self.collection.remove(obj)


class CollectionQueryService(BaseCoreService):
    """
    Query-oriented service for navigating a SpectrumCollection.

    This service provides read-oriented access to spectra, regions,
    and components without modifying the collection structure.
    It does not enforce immutability: returned objects are live
    core instances.
    """

    def check_object_exists(self, obj_id: str) -> bool:
        """
        Check if an object exists in the collection.
        """
        return obj_id in self.collection.objects_index

    def get_parent(self, obj_id: str) -> str:
        """
        Retrieve the first parent ID of core object.

        Parameters
        ----------
        obj_id : str
            Identifier of the object.

        Returns
        -------
        str
            Identifier of the core object's first parent.
        """
        return self._get(obj_id).parent_id

    def get_subtree(self, obj_id: str) -> tuple[str, ...]:
        """
        Return the object and all descendants without removing.

        Parameters
        ----------
        obj_id : str
            Root object ID.

        Returns
        -------
        tuple[str, ...]
            The object and all its descendants IDs.
        """
        return tuple(obj.id_ for obj in self.collection.get_subtree(obj_id))

    def get_regions(self, spectrum_id: str) -> tuple[str, ...]:
        """
        Retrieve all region identifiers belonging to a spectrum.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the parent spectrum.

        Returns
        -------
        tuple[str, ...]
            Regions identifiers whose parent is the given spectrum.
        """
        return tuple(obj.id_ for obj in self.collection.get_children(spectrum_id) if isinstance(obj, Region))

    def get_components(self, region_id: str) -> tuple[str, ...]:
        """
        Retrieve all components identifiers (peaks and background) in a region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.

        Returns
        -------
        tuple[str, ...]
            All components identifiers belonging to the region.
        """
        return tuple(
            obj.id_ for obj in self.collection.get_children(region_id) if isinstance(obj, Component)
        )

    def get_peaks(self, region_id: str) -> tuple[str, ...]:
        """
        Retrieve all peak identifiers in a region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.

        Returns
        -------
        tuple[str, ...]
            All peak identifiers belonging to the region.
        """
        return tuple(obj.id_ for obj in self.collection.get_children(region_id) if isinstance(obj, Peak))

    def get_background(self, region_id: str) -> Optional[str]:
        """
        Retrieve the unique background component of a region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.

        Returns
        -------
        Optional[str]
            The background identifier of the region.

        Raises
        ------
        RuntimeError
            If the region has no background or more than one background.
        """
        bgs = [obj for obj in self.collection.get_children(region_id) if isinstance(obj, Background)]

        if len(bgs) > 1:
            raise RuntimeError(f"Region {region_id} has multiple Backgrounds")

        return bgs[0].id_ if bgs else None

    def get_all_peaks(self) -> tuple[str, ...]:
        """
        Retrieve all peak identifiers in the collection.

        Returns
        -------
        tuple[str, ...]
            All peak identifiers registered in the collection.
        """
        return tuple(obj.id_ for obj in self.collection.objects_index.values() if isinstance(obj, Peak))

    def get_all_spectra(self) -> tuple[str, ...]:
        """
        Retrieve all spectrum identifiers in the collection.

        Returns
        -------
        tuple[str, ...]
            All registered spectra identifiers.
        """
        return tuple(obj.id_ for obj in self.collection.objects_index.values() if isinstance(obj, Spectrum))

    def get_all_regions(self) -> tuple[str, ...]:
        """
        Retrieve all regions identifiers in the collection.

        Returns
        -------
        tuple[str, ...]
            All registered regions identifiers.
        """
        return tuple(obj.id_ for obj in self.collection.objects_index.values() if isinstance(obj, Region))


class SpectrumService(BaseCoreService):
    """
    Service responsible for spectrum lifecycle management.

    This includes creation, data replacement, and removal of spectra.
    """

    @staticmethod
    def _create_spectrum_obj(x: NDArray, y: NDArray, spectrum_id: Optional[str] = None) -> Spectrum:
        """
        Create a new spectrum object.
        """
        return Spectrum(x=x, y=y, id_=spectrum_id)

    def create_spectrum(
        self,
        x: NDArray,
        y: NDArray,
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
        spectrum = self._create_spectrum_obj(x, y, spectrum_id)
        self.attach(spectrum)
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


class RegionService(BaseCoreService):
    """
    Service responsible for region lifecycle management.

    Regions define index-based subranges of spectra.
    """

    @staticmethod
    def _create_region_obj(
        spectrum_id: str,
        start: int,
        stop: int,
        region_id: Optional[str] = None,
    ) -> Region:
        """
        Create a new region object.
        """
        return Region(slice_=slice(start, stop), parent_id=spectrum_id, id_=region_id)

    def _check_slice(self, spectrum_id: str, start: int, stop: int) -> bool:
        """
        Check if a slice is valid.
        """
        spectrum = self._get_typed(spectrum_id, Spectrum)
        if start < 0 or stop > len(spectrum.x):
            return False
        if start >= stop:
            return False
        return True

    def create_region(self, spectrum_id: str, start: int, stop: int, region_id: Optional[str] = None) -> str:
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
        ValueError
            If indices are outside spectrum bounds or start >= stop.
        """
        if not self._check_slice(spectrum_id, start, stop):
            raise ValueError("Invalid region slice")
        region = self._create_region_obj(spectrum_id, start, stop, region_id)
        self.attach(region)
        return region.id_

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
            If indices are outside spectrum bounds or start >= stop.
        """
        region = self._get_typed(region_id, Region)
        if not self._check_slice(region.parent_id, start, stop):
            raise ValueError("Invalid region slice")
        region.slice_ = slice(start, stop)

    def get_slice(self, region_id: str) -> slice:
        """
        Retrieve the index slice of an existing region.

        Parameters
        ----------
        region_id : str
            Identifier of the region.

        Returns
        -------
        slice
            Index slice of the region.
        """
        return self._get_typed(region_id, Region).slice_


class DataQueryService(BaseCoreService):
    """
    Service for accessing numerical spectral data.

    Provides normalized and raw views of spectrum and region data
    without modifying core state.
    """

    def get_norm_ctx(
        self, *, spectrum_id: Optional[str] = None, region_id: Optional[str] = None
    ) -> NormalizationContext:
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
        if spectrum_id is None and region_id is None:
            raise ValueError("spectrum_id or region_id should be provided")
        elif spectrum_id is not None and region_id is not None:
            raise ValueError("Only one of spectrum_id or region_id should be provided")
        elif spectrum_id is not None:
            spectrum = self._get(spectrum_id)
            return spectrum.norm_ctx
        elif region_id is not None:
            region = self._get_typed(region_id, Region)
            return self._get_typed(region.parent_id, Spectrum).norm_ctx

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

        x = spectrum.x.view()
        y = y.view()

        return x, y

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


class ComponentService(BaseCoreService):
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
        expected_type: type[Peak] | type[Background] = Component,
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

        if not isinstance(obj, expected_type):
            raise TypeError(f"Model {model_name} is not a {expected_type.__name__} model")

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
        peak = self._create_component_obj(region_id, model_name, parameters, peak_id, expected_type=Peak)
        self.attach(peak)
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
            expected_type=Background,
        )

        if len(backgrounds) == 1:
            old_bg = backgrounds[0]
            self.detach(old_bg.id_)

        self.attach(new_bg)

        return new_bg.id_

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

    def set_values(self, component_id: str, parameters: dict[str, float]) -> None:
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

    def get_model(self, component_id: str) -> ParametricModelLike:
        component = self._get_typed(component_id, Component)
        return component.model


class MetadataService(BaseCoreService):
    """
    Service for storing and retrieving metadata for core objects.

    Metadata is stored separately from core objects in dicts keyed by object IDs.
    All methods validate that the object exists in the collection before
    reading or writing metadata.

    Metadata schemas are strongly typed per object type:
    - Spectrum: name, group, file
    - Region: (empty, extensible later)
    - Peak: element_type
    """

    def __init__(self, collection: CoreCollection):
        super().__init__(collection)
        self._metadata: dict[str, Metadata] = {}

    def get_metadata(self, obj_id: str) -> Metadata | None:
        """
        Retrieve metadata for an object.

        Parameters
        ----------
        obj_id : str
            Identifier of the object.

        Returns
        -------
        Metadata or None
            Stored metadata or None if none exists.

        Raises
        ------
        KeyError
            If no object with this ID exists in the collection.
        """
        return self._metadata.get(obj_id)

    def set_metadata(self, obj_id: str, metadata: Metadata) -> None:
        """
        Store metadata for an object.

        Parameters
        ----------
        obj_id : str
            Identifier of the object.
        metadata : Metadata
            Metadata to store.

        Raises
        ------
        KeyError
            If no object with this ID exists in the collection.
        """
        if obj_id not in self.collection.objects_index:
            raise KeyError(f"Object with ID {obj_id} does not exist in the collection")
        self._metadata[obj_id] = metadata

    def remove_metadata(self, obj_id: str) -> None:
        """
        Remove metadata for an object by ID.

        Idempotent if no metadata exists.

        Parameters
        ----------
        obj_id : str
            Identifier of the object whose metadata to remove.
        """
        self._metadata.pop(obj_id, None)

    def find_objects(
        self, md_field: str, md_value: str, match_exact: bool = False, tp: type[Metadata] | None = None
    ) -> tuple[str, ...]:
        """
        Return object IDs whose metadata matches the given metadata field and value.

        Parameters
        ----------
        md_field : str
            Metadata field to match.
        md_value : str
            Metadata value to match.
        match_exact : bool, optional
            If True, match exact value. Otherwise, match substring.
        tp : type[Metadata], optional
            Type of the metadata to filter by (e.g., SpectrumMetadata, PeakMetadata).
            If None, matches all metadata types.

        Returns
        -------
        tuple[str, ...]
            Object IDs with matching metadata.
        """
        results = []
        for obj_id, obj_metadata in self._metadata.items():
            # Filter by metadata type if specified
            if tp is not None and not isinstance(obj_metadata, tp):
                continue

            # Get field value using getattr
            try:
                field_value = getattr(obj_metadata, md_field)
            except AttributeError:
                continue

            # Convert to string for comparison
            field_value_str = str(field_value)

            # Match based on exact or fuzzy
            if match_exact:
                if field_value_str == md_value:
                    results.append(obj_id)
            else:
                if md_value.lower() in field_value_str.lower():
                    results.append(obj_id)

        return tuple(results)


@dataclass(frozen=True)
class CoreContext:
    """Read and write access to core services."""

    collection: CollectionQueryService
    spectrum: SpectrumService
    region: RegionService
    data: DataQueryService
    component: ComponentService
    metadata: MetadataService

    @classmethod
    def from_collection(cls, collection: CoreCollection) -> "CoreContext":
        """Build context from a core collection."""
        return cls(
            collection=CollectionQueryService(collection),
            spectrum=SpectrumService(collection),
            region=RegionService(collection),
            data=DataQueryService(collection),
            component=ComponentService(collection),
            metadata=MetadataService(collection),
        )
