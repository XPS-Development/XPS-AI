"""
Thin orchestration layer for spectra analysis.

Aggregates app services and the command/change pipeline into a single entry point
for running services, applying changes (create/update/metadata/remove), and undo/redo.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from core.collection import CoreCollection
from core.metadata import Metadata
from core.services import CoreContext
from tools.dto import ComponentDTO, DTOService, RegionDTO, SpectrumDTO

from .automatization import AutomatizationAdapter
from .command.changes import (
    BaseChange,
    CompositeChange,
    CreateBackground,
    CreatePeak,
    CreateRegion,
    CreateSpectrum,
    FullRemoveObject,
    ParameterField,
    RemoveMetadata,
    RemoveObject,
    ReplaceBackgroundModel,
    ReplacePeakModel,
    SetMetadata,
    UpdateMultipleParameterValues,
    UpdateParameter,
    UpdateRegionSlice,
)
from .command.core import CommandExecutor, UndoRedoStack, create_default_registry
from .import_service import import_spectra as import_spectra_changes
from .nn_service import NNService
from .optimization import OptimizationService
from .serialization import SerializationService


@dataclass
class AppParameters:
    """
    Parameters for the app orchestrator.
    """

    # ---- Core collection parameters ----
    automatic_methods: bool = True
    default_background_model_for_auto_methods: str = "shirley"

    # ---- Import service parameters ----
    import_use_binding_energy: bool = True
    import_use_cps: bool = True

    # ---- NN service parameters ----
    nn_model_path: str | None = None
    nn_pred_threshold: float = 0.5
    nn_smooth: bool = True
    nn_interp_num: int = 256

    # ---- Optimization service parameters ----
    optimization_kwargs: dict[str, Any] = field(default_factory=dict)

    # ---- Serialization service parameters ----
    default_serialization_mode: Literal["append", "replace", "new"] = "replace"
    default_serialization_path: str | Path | None = None
    default_serialization_indent: int | None = None


class QueryService:
    """
    Thin wrapper for querying the collection, metadata and DTO.
    """

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


class AppOrchestrator:
    """
    Single entry point for app services and command execution.

    Holds core context, undo/redo stack, command executor, QueryService,
    NN service, and optimization service. All mutating operations go through
    the command executor for undo/redo support.
    """

    def __init__(
        self,
        collection: CoreCollection,
        params: AppParameters,
    ) -> None:
        """
        Initialize the orchestrator with a core collection and parameters.

        Parameters
        ----------
        collection : CoreCollection
            The core spectrum collection (mutable state).
        params : AppParameters
            Parameters governing import, NN, optimization, and serialization behavior.
        """
        self._core_collection = collection
        self._params = params
        self.__ctx = CoreContext.from_collection(collection)
        self._query = QueryService(self.__ctx)

        self.__stack = UndoRedoStack()
        self._executor = CommandExecutor(self.__ctx, self.__stack, create_default_registry())

        self._nn = NNService(
            model_path=params.nn_model_path,
            pred_threshold=params.nn_pred_threshold,
            smooth=params.nn_smooth,
            interp_num=params.nn_interp_num,
        )
        self._optimization = OptimizationService()
        self._automatization = AutomatizationAdapter()
        self._serialization = SerializationService()

    @property
    def core_collection(self) -> CoreCollection:
        """The core spectrum collection (mutable state)."""
        return self._core_collection

    @property
    def ctx(self) -> CoreContext:
        """Core services context (query, metadata, component, region, etc.)."""
        return self.__ctx

    @property
    def query(self) -> QueryService:
        """
        Read-only query service exposing collection, metadata and DTO queries.

        Returns
        -------
        QueryService
            Query façade bound to the current core context.
        """
        return self._query

    @property
    def can_undo(self) -> bool:
        """True if there is at least one command to undo."""
        return self.__stack.can_undo

    @property
    def can_redo(self) -> bool:
        """True if there is at least one command to redo."""
        return self.__stack.can_redo

    @property
    def is_dirty(self) -> bool:
        """True if there are unsaved changes."""
        return self._serialization.is_dirty

    def execute(self, change: BaseChange) -> None:
        """
        Execute a change (build command, apply, push to undo stack).

        Parameters
        ----------
        change : BaseChange
            Any change (single or CompositeChange).
        """
        self._executor.execute(change)
        self._serialization.mark_dirty()

    def undo(self) -> None:
        """Undo the last executed command."""
        self._executor.undo()

    def redo(self) -> None:
        """Redo the last undone command."""
        self._executor.redo()

    # ---- App services ----

    def import_spectra(self, path: str | Path) -> None:
        """
        Parse a spectrum file and execute changes to create spectra with metadata.

        Import behavior (use_binding_energy, use_cps) is governed by AppParameters.

        Parameters
        ----------
        path : str or Path
            Path to the spectrum file (.txt, .dat, .vms, .vamas).
        """
        change = import_spectra_changes(
            path,
            use_binding_energy=self._params.import_use_binding_energy,
            use_cps=self._params.import_use_cps,
        )
        self.execute(change)

    def load_nn_model(self, model_path: str | Path) -> None:
        """
        Load the NN model into the segmenter pipeline.

        Parameters
        ----------
        model_path : str or Path
            Path to the NN model file.
        """
        self._nn.load_model(model_path)

    def run_segmenter(
        self,
        spectrum_ids: Sequence[str],
    ) -> None:
        """
        Run the segmenter pipeline and execute CompositeChange containing CreateRegion/CreateBackground/CreatePeak changes.

        Parameters
        ----------
        spectrum_ids : Sequence of str
            Identifiers of the parent spectra for CreateRegion.
        """
        changes: list[CompositeChange] = []
        for spectrum_id in spectrum_ids:
            normalized_spectrum = self._query.get_spectrum_dto(spectrum_id, normalized=True)
            original_spectrum = self._query.get_spectrum_dto(spectrum_id, normalized=False)
            changes.append(self._nn.run_segmenter(spectrum_id, normalized_spectrum, original_spectrum))

        self.execute(CompositeChange(changes=changes))

    def optimize_regions(
        self,
        *,
        region_ids: Sequence[str] | None = None,
        spectrum_ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Run optimization and execute UpdateMultipleParameterValues changes.

        Default optimization kwargs from AppParameters are merged with explicit
        kwargs; caller values override defaults on conflict.

        Parameters
        ----------
        region_ids : Sequence of str
            Identifiers of the regions to optimize.
        spectrum_ids : Sequence of str
            Identifiers of the spectra to optimize.
        **kwargs
            Passed to lmfit.minimize; overrides AppParameters.optimization_kwargs.
        """
        merged = {**self._params.optimization_kwargs, **kwargs}
        changes: list[CompositeChange] = []

        if region_ids is None and spectrum_ids is None:
            raise ValueError("region_ids or spectrum_ids must be provided")

        if region_ids is None:
            region_ids = []
            for spectrum_id in spectrum_ids:
                region_ids.extend(self._query.get_regions_ids(spectrum_id))

        region_reprs: list[tuple[RegionDTO, tuple[ComponentDTO, ...]]] = []
        for region_id in region_ids:
            region_repr = self._query.get_region_dto_repr(region_id, normalized=True)
            region_reprs.append(region_repr)

        changes.append(self._optimization.optimize_regions(region_reprs, **merged))
        self.execute(CompositeChange(changes=changes))

    # ---- Parameters and models ----

    def update_parameter(
        self,
        component_id: str,
        name: str,
        parameter_field: ParameterField,
        new_value: str | bool | float,
        *,
        normalized: bool = False,
    ) -> None:
        """Update a single parameter attribute; executed as a command."""
        self.execute(
            UpdateParameter(
                component_id=component_id,
                name=name,
                parameter_field=parameter_field,
                new_value=new_value,
                normalized=normalized,
            )
        )

    def update_parameters(
        self,
        component_id: str,
        parameters: dict[str, str | bool | float],
        *,
        normalized: bool = False,
    ) -> None:
        """Update multiple parameter values at once; executed as a command."""
        self.execute(
            UpdateMultipleParameterValues(
                component_id=component_id,
                parameters=parameters,
                normalized=normalized,
            )
        )

    def update_region_slice(
        self,
        region_id: str,
        start: int | float,
        stop: int | float,
        mode: Literal["value", "index"] = "index",
    ) -> None:
        """Update the index slice of an existing region; executed as a command."""
        if self._params.automatic_methods:
            background_id = self._query.get_background_id(region_id)
            if background_id is not None:
                background_dto = self._query.get_component_dto(background_id)
                background_model_name = background_dto.model.name
                spectrum_id = self._query.get_parent_id(region_id)
                spectrum = self._query.get_spectrum_dto(spectrum_id, normalized=False)
                change = self._automatization.update_slice_with_intensities(
                    region_id=region_id,
                    background_id=background_id,
                    background_model_name=background_model_name,
                    spectrum_x=spectrum.x,
                    spectrum_y=spectrum.y,
                    start=start,
                    stop=stop,
                    mode=mode,
                )
                self.execute(change)
                return

        self.execute(UpdateRegionSlice(region_id=region_id, start=start, stop=stop, mode=mode))

    def replace_peak_model(
        self,
        peak_id: str,
        new_model_name: str,
        parameters: dict[str, float] | None = None,
    ) -> None:
        """Replace a peak's model (and optionally parameters); executed as a command."""
        self.execute(
            ReplacePeakModel(
                peak_id=peak_id,
                new_model_name=new_model_name,
                parameters=parameters,
            )
        )

    def replace_background_model(
        self,
        region_id: str,
        new_model_name: str,
        parameters: dict[str, float] | None = None,
        background_id: str | None = None,
    ) -> None:
        """Replace a background's model; executed as a command."""
        self.execute(
            ReplaceBackgroundModel(
                region_id=region_id,
                new_model_name=new_model_name,
                parameters=parameters,
                background_id=background_id,
            )
        )

    # ---- Create ----

    def create_spectrum(
        self,
        x: Any,
        y: Any,
        spectrum_id: str | None = None,
    ) -> None:
        """Create a new spectrum; executed as a command."""
        self.execute(CreateSpectrum(x=x, y=y, spectrum_id=spectrum_id))

    def create_region(
        self,
        spectrum_id: str,
        start: int | float,
        stop: int | float,
        region_id: str | None = None,
        mode: Literal["value", "index"] = "index",
    ) -> None:
        """Create a new region; executed as a command."""
        self.execute(
            CreateRegion(
                spectrum_id=spectrum_id,
                start=start,
                stop=stop,
                region_id=region_id,
                mode=mode,
            )
        )

    def create_peak(
        self,
        region_id: str,
        model_name: str,
        parameters: dict[str, float] | None = None,
        peak_id: str | None = None,
    ) -> None:
        """Create a new peak component; executed as a command."""
        if self._params.automatic_methods and model_name == "pseudo-voigt" and parameters is None:
            region_repr = self._query.get_region_dto_repr(region_id, normalized=False)
            change = self._automatization.create_pseudo_voigt_peak(region_repr[0], region_repr[1])
            self.execute(change)
        else:
            self.execute(
                CreatePeak(
                    region_id=region_id,
                    model_name=model_name,
                    parameters=parameters,
                    peak_id=peak_id,
                )
            )

    def create_background(
        self,
        region_id: str,
        model_name: str,
        parameters: dict[str, float] | None = None,
        background_id: str | None = None,
    ) -> None:
        """Create or replace a background component; executed as a command."""
        if self._params.automatic_methods and parameters is None:
            spectrum_id = self._query.get_parent_id(region_id)
            spectrum = self._query.get_spectrum_dto(spectrum_id, normalized=False)
            start, stop = self._query.get_region_slice(region_id, mode="index")
            change = self._automatization.create_background(
                region_id=region_id,
                spectrum_x=spectrum.x,
                spectrum_y=spectrum.y,
                start=start,
                stop=stop,
                mode="index",
                model_name=model_name,
            )
            if background_id is not None:
                change = CreateBackground(
                    region_id=change.region_id,
                    model_name=change.model_name,
                    parameters=change.parameters,
                    background_id=background_id,
                )
            self.execute(change)
        else:
            self.execute(
                CreateBackground(
                    region_id=region_id,
                    model_name=model_name,
                    parameters=parameters,
                    background_id=background_id,
                )
            )

    # ---- Metadata ----

    def set_metadata(self, obj_id: str, metadata: Metadata) -> None:
        """Set metadata for an object; executed as a command."""
        self.execute(SetMetadata(obj_id=obj_id, metadata=metadata))

    # ---- Remove ----

    def remove_object(self, obj_id: str) -> None:
        """Remove an object from the collection (cascades to children); executed as a command."""
        self.execute(RemoveObject(obj_id=obj_id))

    def remove_metadata(self, obj_id: str) -> None:
        """Remove metadata for an object; executed as a command."""
        self.execute(RemoveMetadata(obj_id=obj_id))

    def full_remove_object(self, obj_id: str) -> None:
        """Remove an object and its metadata (and all descendants' metadata); executed as a command."""
        self.execute(FullRemoveObject(obj_id=obj_id))

    # ---- Serialization ----

    def dump_collection(
        self,
        path: str | Path | None = None,
        *,
        indent: int | None = None,
    ) -> None:
        """
        Save collection and metadata to a JSON file.

        If path or indent are omitted, AppParameters defaults are used.

        Parameters
        ----------
        path : str or Path or None, optional
            File path. If None, uses AppParameters.default_serialization_path.
        indent : int or None, optional
            JSON indentation level. If None, uses AppParameters.default_serialization_indent.

        Raises
        ------
        ValueError
            If path is None and no default path is set.
        """
        resolved_path = path if path is not None else self._params.default_serialization_path
        if resolved_path is None:
            raise ValueError("path is required when AppParameters.default_serialization_path is not set")
        resolved_indent = indent if indent is not None else self._params.default_serialization_indent
        self._serialization.dump(
            path=resolved_path,
            collection=self._core_collection,
            metadata_service=self.__ctx.metadata,
            indent=resolved_indent,
        )

    def load_collection(
        self,
        path: str | Path,
        *,
        mode: Literal["append", "replace"] | None = None,
    ) -> None:
        """
        Load collection and metadata from a JSON file.

        If mode is omitted, AppParameters.default_serialization_mode is used.
        For replace mode, the undo/redo stack is cleared.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file.
        mode : {"append", "replace"} or None, optional
            - append: add loaded objects to the current collection/metadata.
            - replace: clear current collection/metadata in-place, then load.
            If None, uses AppParameters.default_serialization_mode (must be append or replace).
        """
        resolved_mode = mode if mode is not None else self._params.default_serialization_mode
        if resolved_mode not in ("append", "replace"):
            raise ValueError(
                f"mode must be 'append' or 'replace', got {resolved_mode!r}; "
                "AppParameters.default_serialization_mode='new' is not supported"
            )
        self._serialization.load(
            path=path,
            collection=self._core_collection,
            metadata_service=self.__ctx.metadata,
            mode=resolved_mode,
        )
        if resolved_mode == "replace":
            self._executor.clear()

    def set_default_save_path(self, path: str | Path | None) -> None:
        """Set the default save path (stored in AppParameters)."""
        self._params.default_serialization_path = path

    def get_default_save_path(self) -> Path | None:
        """Return the default save path from AppParameters."""
        p = self._params.default_serialization_path
        return Path(p) if p is not None else None
