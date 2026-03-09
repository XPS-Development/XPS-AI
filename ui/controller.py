from pathlib import Path
from typing import Any, Sequence

from PySide6.QtCore import QObject, Signal

from app.command.changes import BaseChange, ParameterField
from app.orchestration import AppOrchestrator, AppParameters, QueryService
from core.collection import CoreCollection
from core.metadata import Metadata


class ControllerWrapper(QObject):
    """
    Qt-aware wrapper around :class:`AppOrchestrator`.

    This wrapper owns a core collection and orchestrator instance, exposes a
    small, UI-friendly API surface, and emits Qt signals whenever the
    underlying model or undo/redo state changes.

    Parameters
    ----------
    collection : CoreCollection or None, optional
        Existing collection to wrap. If None and ``orchestrator`` is also
        None, a new empty :class:`CoreCollection` is created.
    orchestrator : AppOrchestrator or None, optional
        Existing orchestrator instance. If provided, ``collection`` is
        ignored and the orchestrator's internal collection/context are used.
    nn_model_path : str or None, optional
        Optional path to an NN model passed through to
        :class:`AppOrchestrator` when it is constructed internally.
    parent : QObject or None, optional
        QObject parent used by Qt for lifetime management.
    """

    collectionChanged: Signal = Signal()
    undoRedoStateChanged: Signal = Signal(bool, bool)
    selectionChanged: Signal = Signal(object, object)

    def __init__(
        self,
        collection: CoreCollection | None = None,
        orchestrator: AppOrchestrator | None = None,
        *,
        nn_model_path: str | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        if orchestrator is not None:
            self._orchestrator = orchestrator
            self._collection = orchestrator.core_collection
        else:
            self._collection = collection or CoreCollection()
            params = AppParameters(nn_model_path=nn_model_path)
            self._orchestrator = AppOrchestrator(self._collection, params)

        self._selected_spectrum_id: str | None = None
        self._selected_region_id: str | None = None

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    @property
    def collection(self) -> CoreCollection:
        """
        Underlying core collection.

        Returns
        -------
        CoreCollection
            The wrapped collection instance.
        """
        return self._collection

    @property
    def orchestrator(self) -> AppOrchestrator:
        """
        Underlying application orchestrator.

        Returns
        -------
        AppOrchestrator
            The wrapped orchestrator instance.
        """
        return self._orchestrator

    @property
    def query(self) -> QueryService:
        """
        Query service for collection, metadata, and DTO projections.

        Returns
        -------
        QueryService
            Read-only query façade bound to the orchestrator.
        """
        return self._orchestrator.query

    # ------------------------------------------------------------------
    # Selection handling
    # ------------------------------------------------------------------

    @property
    def selected_spectrum_id(self) -> str | None:
        """
        Identifier of the currently selected spectrum.

        Returns
        -------
        str or None
            Selected spectrum identifier or None if nothing is selected.
        """
        return self._selected_spectrum_id

    @property
    def selected_region_id(self) -> str | None:
        """
        Identifier of the currently selected region.

        Returns
        -------
        str or None
            Selected region identifier or None if no region is selected.
        """
        return self._selected_region_id

    def set_selection(self, spectrum_id: str | None, region_id: str | None = None) -> None:
        """
        Update the current spectrum/region selection.

        Parameters
        ----------
        spectrum_id : str or None
            New spectrum identifier or None to clear selection.
        region_id : str or None, optional
            New region identifier or None to clear region selection.
        """
        if spectrum_id == self._selected_spectrum_id and region_id == self._selected_region_id:
            return

        self._selected_spectrum_id = spectrum_id
        self._selected_region_id = region_id
        self.selectionChanged.emit(spectrum_id, region_id)

    # ------------------------------------------------------------------
    # Read-only helpers for views
    # ------------------------------------------------------------------

    def get_all_spectra(self) -> tuple[str, ...]:
        """
        Return identifiers of all spectra in the collection.

        Returns
        -------
        tuple[str, ...]
            All spectrum identifiers.
        """
        return self._orchestrator.query.get_all_spectra_ids()

    def get_metadata(self, obj_id: str) -> Metadata | None:
        """
        Retrieve metadata for a core object.

        Parameters
        ----------
        obj_id : str
            Object identifier.

        Returns
        -------
        Metadata or None
            Stored metadata, if any.
        """
        return self._orchestrator.query.get_metadata(obj_id)

    # ------------------------------------------------------------------
    # ViewerDataProvider (for plot area; delegates to query)
    # ------------------------------------------------------------------

    def get_spectrum(self, spectrum_id: str, *, normalized: bool = False) -> Any:
        """
        Return a spectrum-like projection with .x and .y arrays.

        Implements ViewerDataProvider for the plot area.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum.
        normalized : bool, optional
            If True, return normalized data.

        Returns
        -------
        SpectrumLike
            Spectrum projection with x and y arrays.
        """
        return self._orchestrator.query.get_spectrum_dto(spectrum_id, normalized=normalized)

    def get_region(self, region_id: str, *, normalized: bool = False) -> Any:
        """
        Return a region-like projection with .x and .y arrays.

        Implements ViewerDataProvider for the plot area.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        normalized : bool, optional
            If True, return normalized data.

        Returns
        -------
        RegionLike
            Region projection with x and y arrays.
        """
        return self._orchestrator.query.get_region_dto(region_id, normalized=normalized)

    def get_spectrum_repr(
        self, spectrum_id: str, *, normalized: bool = False
    ) -> Any:
        """
        Return spectrum-like and its region-like and component-like objects.

        Implements ViewerDataProvider for the plot area.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum.
        normalized : bool, optional
            If True, return normalized data and parameters.

        Returns
        -------
        tuple[SpectrumLike, tuple[tuple[RegionLike, tuple[ComponentLike, ...]], ...]]
            Spectrum and its regions with components for evaluation.
        """
        return self._orchestrator.query.get_spectrum_dto_repr(spectrum_id, normalized=normalized)

    def get_region_repr(
        self, region_id: str, *, normalized: bool = False
    ) -> tuple[Any, tuple[Any, ...]]:
        """
        Return region-like and its component-like objects.

        Implements ViewerDataProvider for the plot area.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        normalized : bool, optional
            If True, return normalized data and parameters.

        Returns
        -------
        tuple[RegionLike, tuple[ComponentLike, ...]]
            Region and its components for evaluation.
        """
        return self._orchestrator.query.get_region_dto_repr(region_id, normalized=normalized)

    # ------------------------------------------------------------------
    # Mutation API: thin forwarding to orchestrator with signals
    # ------------------------------------------------------------------

    def execute(self, change: BaseChange) -> None:
        """
        Execute a change via the orchestrator and emit signals.

        Parameters
        ----------
        change : BaseChange
            Change instance to execute.
        """
        self._orchestrator.execute(change)
        self._emit_collection_and_undo_redo()

    def undo(self) -> None:
        """
        Undo the last executed command and emit signals.
        """
        self._orchestrator.undo()
        self._emit_collection_and_undo_redo()

    def redo(self) -> None:
        """
        Redo the last undone command and emit signals.
        """
        self._orchestrator.redo()
        self._emit_collection_and_undo_redo()

    def import_spectra(self, path: str | Path) -> None:
        """
        Import spectra from a file and emit signals.

        Parameters
        ----------
        path : str or Path
            Path to the spectra file.
        """
        self._orchestrator.import_spectra(path)
        self._emit_collection_and_undo_redo()

    def run_segmenter(self, spectrum_ids: Sequence[str]) -> None:
        """
        Run the segmenter pipeline and emit signals.

        Parameters
        ----------
        spectrum_ids : Sequence[str]
            Identifiers of the parent spectra for segmentation.
        """
        self._orchestrator.run_segmenter(spectrum_ids)
        self._emit_collection_and_undo_redo()

    def optimize_regions(
        self,
        *,
        region_ids: Sequence[str] | None = None,
        spectrum_ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Run optimization for regions and emit signals.

        Parameters
        ----------
        region_ids : Sequence[str] or None, optional
            Identifiers of the regions to optimize.
        spectrum_ids : Sequence[str] or None, optional
            Identifiers of the spectra whose regions to optimize.
        **kwargs
            Extra keyword arguments forwarded to the optimization service.
        """
        self._orchestrator.optimize_regions(
            region_ids=region_ids,
            spectrum_ids=spectrum_ids,
            **kwargs,
        )
        self._emit_collection_and_undo_redo()

    def update_parameter(
        self,
        component_id: str,
        name: str,
        parameter_field: ParameterField,
        new_value: str | bool | float,
        *,
        normalized: bool = False,
    ) -> None:
        """
        Update a single parameter attribute and emit signals.

        Parameters
        ----------
        component_id : str
            Identifier of the component.
        name : str
            Parameter name.
        parameter_field : ParameterField
            Field of the parameter to modify.
        new_value : str or bool or float
            New parameter value.
        normalized : bool, default=False
            Whether the value is expressed in normalized units.
        """
        self._orchestrator.update_parameter(
            component_id=component_id,
            name=name,
            parameter_field=parameter_field,
            new_value=new_value,
            normalized=normalized,
        )
        self._emit_collection_and_undo_redo()

    def update_parameters(
        self,
        component_id: str,
        parameters: dict[str, str | bool | float],
        *,
        normalized: bool = False,
    ) -> None:
        """
        Update multiple parameter values at once and emit signals.

        Parameters
        ----------
        component_id : str
            Identifier of the component.
        parameters : dict[str, str or bool or float]
            Mapping from parameter names to new values.
        normalized : bool, default=False
            Whether values are expressed in normalized units.
        """
        self._orchestrator.update_parameters(
            component_id=component_id,
            parameters=parameters,
            normalized=normalized,
        )
        self._emit_collection_and_undo_redo()

    def update_region_slice(self, region_id: str, start: int, stop: int) -> None:
        """
        Update the index slice of a region and emit signals.

        Parameters
        ----------
        region_id : str
            Region identifier.
        start : int
            Start index of the slice (inclusive).
        stop : int
            Stop index of the slice (exclusive).
        """
        self._orchestrator.update_region_slice(region_id, start, stop)
        self._emit_collection_and_undo_redo()

    def create_spectrum(
        self,
        x: Any,
        y: Any,
        spectrum_id: str | None = None,
    ) -> None:
        """
        Create a new spectrum and emit signals.

        Parameters
        ----------
        x : Any
            X-axis values.
        y : Any
            Y-axis intensity values.
        spectrum_id : str or None, optional
            Explicit spectrum identifier.
        """
        self._orchestrator.create_spectrum(x=x, y=y, spectrum_id=spectrum_id)
        self._emit_collection_and_undo_redo()

    def create_region(
        self,
        spectrum_id: str,
        start: int,
        stop: int,
        region_id: str | None = None,
    ) -> None:
        """
        Create a new region and emit signals.

        Parameters
        ----------
        spectrum_id : str
            Parent spectrum identifier.
        start : int
            Start index (inclusive).
        stop : int
            Stop index (exclusive).
        region_id : str or None, optional
            Explicit region identifier.
        """
        self._orchestrator.create_region(
            spectrum_id=spectrum_id,
            start=start,
            stop=stop,
            region_id=region_id,
        )
        self._emit_collection_and_undo_redo()

    def create_peak(
        self,
        region_id: str,
        model_name: str,
        parameters: dict[str, float] | None = None,
        peak_id: str | None = None,
    ) -> None:
        """
        Create a new peak component and emit signals.

        Parameters
        ----------
        region_id : str
            Parent region identifier.
        model_name : str
            Name of the peak model.
        parameters : dict[str, float] or None, optional
            Initial parameter values.
        peak_id : str or None, optional
            Explicit peak identifier.
        """
        self._orchestrator.create_peak(
            region_id=region_id,
            model_name=model_name,
            parameters=parameters,
            peak_id=peak_id,
        )
        self._emit_collection_and_undo_redo()

    def create_background(
        self,
        region_id: str,
        model_name: str,
        parameters: dict[str, float] | None = None,
        background_id: str | None = None,
    ) -> None:
        """
        Create or replace a background component and emit signals.

        Parameters
        ----------
        region_id : str
            Parent region identifier.
        model_name : str
            Background model name.
        parameters : dict[str, float] or None, optional
            Initial parameter values.
        background_id : str or None, optional
            Explicit background identifier.
        """
        self._orchestrator.create_background(
            region_id=region_id,
            model_name=model_name,
            parameters=parameters,
            background_id=background_id,
        )
        self._emit_collection_and_undo_redo()

    def set_metadata(self, obj_id: str, metadata: Metadata) -> None:
        """
        Set metadata for an object and emit signals.

        Parameters
        ----------
        obj_id : str
            Object identifier.
        metadata : Metadata
            Metadata instance to store.
        """
        self._orchestrator.set_metadata(obj_id, metadata)
        self._emit_collection_and_undo_redo()

    def remove_object(self, obj_id: str) -> None:
        """
        Remove an object and its descendants and emit signals.

        Parameters
        ----------
        obj_id : str
            Identifier of the object to remove.
        """
        self._orchestrator.remove_object(obj_id)
        self._emit_collection_and_undo_redo()

    def remove_metadata(self, obj_id: str) -> None:
        """
        Remove metadata for an object and emit signals.

        Parameters
        ----------
        obj_id : str
            Identifier of the object whose metadata to remove.
        """
        self._orchestrator.remove_metadata(obj_id)
        self._emit_collection_and_undo_redo()

    def full_remove_object(self, obj_id: str) -> None:
        """
        Remove an object, all descendants, and their metadata and emit signals.

        Parameters
        ----------
        obj_id : str
            Identifier of the root object to remove.
        """
        self._orchestrator.full_remove_object(obj_id)
        self._emit_collection_and_undo_redo()

    def dump_collection(
        self,
        path: str | Path | None = None,
        *,
        indent: int | None = None,
    ) -> None:
        """
        Persist the collection and metadata to disk and emit signals.

        Parameters
        ----------
        path : str or Path or None, optional
            Path to the JSON file. If None, uses the default path from
            the serialization service.
        indent : int or None, optional
            JSON indentation level.
        """
        self._orchestrator.dump_collection(path=path, indent=indent)
        self.collectionChanged.emit()
        self._emit_undo_redo_state()

    def load_collection(
        self,
        path: str | Path,
        *,
        mode: str = "replace",
    ) -> None:
        """
        Load collection and metadata from disk and emit signals.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file.
        mode : {\"append\", \"replace\", \"new\"}, default=\"replace\"
            Loading mode passed to the orchestrator.
        """
        self._orchestrator.load_collection(path, mode=mode)  # type: ignore[arg-type]
        self.collectionChanged.emit()
        self._emit_undo_redo_state()

    def set_default_save_path(self, path: str | Path) -> None:
        """
        Set the default save path in the serialization service.

        Parameters
        ----------
        path : str or Path
            Default file path to use when saving.
        """
        self._orchestrator.set_default_save_path(path)

    def get_default_save_path(self) -> Path | None:
        """
        Return the current default save path.

        Returns
        -------
        Path or None
            Default save path or None if not set.
        """
        return self._orchestrator.get_default_save_path()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_collection_and_undo_redo(self) -> None:
        """Emit collectionChanged and updated undo/redo state."""
        self.collectionChanged.emit()
        self._emit_undo_redo_state()

    def _emit_undo_redo_state(self) -> None:
        """Emit the current undo/redo capability state."""
        self.undoRedoStateChanged.emit(
            self._orchestrator.can_undo,
            self._orchestrator.can_redo,
        )
