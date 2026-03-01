"""
Thin orchestration layer for spectra analysis.

Aggregates app services and the command/change pipeline into a single entry point
for running services, applying changes (create/update/metadata/remove), and undo/redo.
"""

from pathlib import Path
from typing import Any, Literal, Sequence

from core.collection import CoreCollection
from core.metadata import Metadata
from core.services import CoreContext
from tools.dto import DTOService

from .command.changes import (
    BaseChange,
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
from .command.core import (
    CommandExecutor,
    CommandRegistry,
    UndoRedoStack,
    create_default_registry,
)
from .import_service import import_spectra as import_spectra_changes
from .nn_service import NNService
from .optimization import OptimizationService
from .serialization import SerializationService


class AppOrchestrator:
    """
    Single entry point for app services and command execution.

    Holds core context, undo/redo stack, command executor, DTO service,
    NN service, and optimization service. All mutating operations go through
    the command executor for undo/redo support.
    """

    def __init__(
        self,
        collection: CoreCollection,
        registry: CommandRegistry | None = None,
        *,
        nn_model_path: str | None = None,
    ) -> None:
        """
        Initialize the orchestrator with a core collection.

        Parameters
        ----------
        collection : CoreCollection
            The core spectrum collection (mutable state).
        registry : CommandRegistry or None, optional
            Custom command registry; if None, the default registry is used.
        nn_model_path : str or None, optional
            Path to segmenter ONNX model for NNService; if None, load_model
            must be called before run_segmenter.
        """
        self._core_collection = collection
        self._ctx = CoreContext.from_collection(collection)
        self._stack = UndoRedoStack()
        self._executor = CommandExecutor(self._ctx, self._stack, registry or create_default_registry())
        self._dto = DTOService(self._ctx)
        self._nn = NNService(model_path=nn_model_path)
        self._optimization = OptimizationService()
        self._serialization = SerializationService(collection, self._ctx.metadata)

    @property
    def core_collection(self) -> CoreCollection:
        """The core spectrum collection (mutable state)."""
        return self._core_collection

    @property
    def ctx(self) -> CoreContext:
        """Read/write access to core services."""
        return self._ctx

    @property
    def dto_service(self) -> DTOService:
        """DTO service for get_spectrum, get_region_repr, etc."""
        return self._dto

    @property
    def can_undo(self) -> bool:
        """True if there is at least one command to undo."""
        return self._stack.can_undo

    @property
    def can_redo(self) -> bool:
        """True if there is at least one command to redo."""
        return self._stack.can_redo

    @property
    def serialization(self) -> SerializationService:
        """Serialization service for save/load and dirty state."""
        return self._serialization

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

    # ---- Run app services ----

    def import_spectra(self, path: str | Path) -> None:
        """
        Parse a spectrum file and execute changes to create spectra with metadata.

        Parameters
        ----------
        path : str or Path
            Path to the spectrum file (.txt, .dat, .vms, .vamas).
        """
        change = import_spectra_changes(path)
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
        spectrum_id: str,
        normalized_spectrum: Any,
        original_spectrum: Any,
    ) -> None:
        """
        Run the segmenter pipeline and execute CreateRegion/CreateBackground/CreatePeak changes.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the parent spectrum for CreateRegion.
        normalized_spectrum : SpectrumLike
            Spectrum with normalized y (e.g. from dto_service.get_spectrum(..., normalized=True)).
        original_spectrum : SpectrumLike
            Original spectrum with raw x/y (e.g. from dto_service.get_spectrum(..., normalized=False)).
        """
        change = self._nn.run_segmenter(spectrum_id, normalized_spectrum, original_spectrum)
        self.execute(change)

    def optimize_regions(
        self,
        region_reprs: Sequence[tuple[Any, tuple[Any, ...]]],
        **kwargs: Any,
    ) -> None:
        """
        Run optimization and execute UpdateMultipleParameterValues changes.

        Parameters
        ----------
        region_reprs : Sequence of (RegionDTO, tuple of ComponentDTO)
            Region and component DTOs (e.g. from dto_service.get_region_repr).
        **kwargs
            Passed to lmfit.minimize.
        """
        change = self._optimization.optimize_regions(region_reprs, **kwargs)
        self.execute(change)

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

    def update_region_slice(self, region_id: str, start: int, stop: int) -> None:
        """Update the index slice of an existing region; executed as a command."""
        self.execute(UpdateRegionSlice(region_id=region_id, start=start, stop=stop))

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
        start: int,
        stop: int,
        region_id: str | None = None,
    ) -> None:
        """Create a new region; executed as a command."""
        self.execute(
            CreateRegion(
                spectrum_id=spectrum_id,
                start=start,
                stop=stop,
                region_id=region_id,
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

        Parameters
        ----------
        path : str or Path or None, optional
            File path. If None, uses the serialization service's default path.
        indent : int or None, optional
            JSON indentation level.

        Raises
        ------
        ValueError
            If path is None and no default path is set.
        """
        self._serialization.dump(path=path, indent=indent)

    def load_collection(
        self,
        path: str | Path,
        *,
        mode: Literal["append", "replace", "new"] = "replace",
    ) -> None:
        """
        Load collection and metadata from a JSON file.

        For replace and new, the undo/redo stack is cleared. For new, the
        orchestrator replaces its collection and context with the loaded state
        and wires a new serialization service.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file.
        mode : {"append", "replace", "new"}, optional
            - append: add loaded objects to the current collection/metadata.
            - replace: clear current collection/metadata in-place, then load.
            - new: load into new collection and metadata; replace orchestrator state.
        """
        result = self._serialization.load(path, mode=mode)
        if mode in ("replace", "new"):
            self._executor.clear()
        if mode == "new" and result is not None:
            new_collection, new_metadata_service = result
            self._core_collection = new_collection
            self._ctx = CoreContext.from_collection_and_metadata(new_collection, new_metadata_service)
            self._executor.ctx = self._ctx
            self._dto = DTOService(self._ctx)
            self._serialization = SerializationService(new_collection, new_metadata_service)

    def set_default_save_path(self, path: str | Path) -> None:
        """Set the default path used by dump_collection when path is omitted."""
        self._serialization.set_default_path(path)

    def get_default_save_path(self) -> Path | None:
        """Return the current default save path, or None."""
        return self._serialization.get_default_path()
