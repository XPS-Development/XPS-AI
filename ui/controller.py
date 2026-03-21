from pathlib import Path
from typing import Any, Literal, Sequence

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
    spectrumTreeChanged: Signal = Signal()

    def __init__(
        self,
        collection: CoreCollection | None = None,
        orchestrator: AppOrchestrator | None = None,
        *,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        if orchestrator is not None:
            self._orchestrator = orchestrator
            self._collection = orchestrator.core_collection
        else:
            self._collection = collection or CoreCollection()
            params = AppParameters()
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

    def get_spectrum_repr(self, spectrum_id: str, *, normalized: bool = False) -> Any:
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

    def get_region_repr(self, region_id: str, *, normalized: bool = False) -> tuple[Any, tuple[Any, ...]]:
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

    def undo(self) -> None:
        """
        Undo the last executed command and emit signals.
        """
        self._orchestrator.undo()
        self._emit_spectrum_tree_changed()
        self._emit_collection_and_undo_redo()

    def redo(self) -> None:
        """
        Redo the last undone command and emit signals.
        """
        self._orchestrator.redo()
        self._emit_spectrum_tree_changed()
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
        self._emit_spectrum_tree_changed()
        self._emit_undo_redo_state()

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
        new_value: str | bool | float | None,
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

    def update_region_slice(
        self,
        region_id: str,
        start: int | float | None = None,
        stop: int | float | None = None,
        mode: Literal["value", "index"] = "index",
    ) -> None:
        """
        Update the slice of a region and emit signals.

        Parameters
        ----------
        region_id : str
            Region identifier.
        start : int or float or None, default=None
            Start of the slice (index or x value depending on mode).
        stop : int or float or None, default=None
            Stop of the slice (index or x value depending on mode).
        mode : Literal["value", "index"], default="index"
            Whether start/stop are indices or x-axis values.
        """
        self._orchestrator.update_region_slice(region_id, start, stop, mode=mode)
        self._emit_collection_and_undo_redo()

    def replace_peak_model(
        self,
        peak_id: str,
        new_model_name: str,
        parameters: dict[str, float] | None = None,
    ) -> None:
        """
        Replace a peak's model and emit signals.

        Parameters
        ----------
        peak_id : str
            Identifier of the peak component.
        new_model_name : str
            Name of the new registered peak model.
        parameters : dict[str, float] or None, optional
            Optional initial parameter values for the new model.
        """
        self._orchestrator.replace_peak_model(peak_id, new_model_name, parameters=parameters)
        self._emit_collection_and_undo_redo()

    def replace_background_model(
        self,
        region_id: str,
        new_model_name: str,
        parameters: dict[str, float] | None = None,
        background_id: str | None = None,
    ) -> None:
        """
        Replace the background model for a region and emit signals.

        Parameters
        ----------
        region_id : str
            Identifier of the region.
        new_model_name : str
            Name of the new registered background model.
        parameters : dict[str, float] or None, optional
            Optional initial parameter values for the new model.
        background_id : str or None, optional
            Identifier of the background to replace; if None, the region's current background is used.
        """
        self._orchestrator.replace_background_model(
            region_id, new_model_name, parameters=parameters, background_id=background_id
        )
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
        self._emit_spectrum_tree_changed()
        self._emit_undo_redo_state()

    def create_region(
        self,
        spectrum_id: str,
        start: int | float | None = None,
        stop: int | float | None = None,
        region_id: str | None = None,
        mode: Literal["value", "index"] = "index",
    ) -> None:
        """
        Create a new region and emit signals.

        Parameters
        ----------
        spectrum_id : str
            Parent spectrum identifier.
        start : int or float or None, default=None
            Start index (inclusive) or x-axis value depending on mode.
        stop : int or float or None, default=None
            Stop index (exclusive) or x-axis value depending on mode.
        region_id : str or None, optional
            Explicit region identifier.
        mode : Literal["value", "index"], default="index"
            Whether start/stop are indices or x-axis values.
        """
        self._orchestrator.create_region(
            spectrum_id=spectrum_id,
            start=start,
            stop=stop,
            region_id=region_id,
            mode=mode,
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
        self._emit_spectrum_tree_changed()
        self._emit_undo_redo_state()

    def rename_spectrum(self, spectrum_id: str, new_name: str) -> None:
        """
        Rename a single spectrum and emit collection/undo-redo signals.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum to rename.
        new_name : str
            New display name.
        """
        self._orchestrator.rename_spectrum(spectrum_id, new_name)
        self._emit_spectrum_tree_changed()
        self._emit_undo_redo_state()

    def rename_group(self, file_label: str, old_group_label: str, new_group_label: str) -> None:
        """
        Rename a group within a file and emit collection/undo-redo signals.

        Parameters
        ----------
        file_label : str
            File label whose group should be renamed.
        old_group_label : str
            Existing group label.
        new_group_label : str
            New group label.
        """
        self._orchestrator.rename_group(file_label, old_group_label, new_group_label)
        self._emit_spectrum_tree_changed()
        self._emit_undo_redo_state()

    def rename_file(self, old_file_label: str, new_file_label: str) -> None:
        """
        Rename a file bucket and emit collection/undo-redo signals.

        Parameters
        ----------
        old_file_label : str
            Existing file label.
        new_file_label : str
            New file label.
        """
        self._orchestrator.rename_file(old_file_label, new_file_label)
        self._emit_spectrum_tree_changed()
        self._emit_undo_redo_state()

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
        self._emit_spectrum_tree_changed()
        self._emit_undo_redo_state()

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

    def remove_spectrum(self, spectrum_id: str) -> None:
        """
        Remove a spectrum and emit signals.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum to remove.
        """
        self._orchestrator.full_remove_object(spectrum_id)
        self._emit_spectrum_tree_changed()
        self._emit_undo_redo_state()

    def remove_group(self, file_label: str, group_label: str) -> None:
        """
        Remove all spectra belonging to a given file/group combination.

        Parameters
        ----------
        file_label : str
            File label whose group contents should be removed.
        group_label : str
            Group label to remove.
        """
        self._orchestrator.remove_group(file_label, group_label)
        self._emit_spectrum_tree_changed()
        self._emit_undo_redo_state()

    def remove_file(self, file_label: str) -> None:
        """
        Remove all spectra associated with a given file label.

        Parameters
        ----------
        file_label : str
            File label whose spectra should be removed.
        """
        self._orchestrator.remove_file(file_label)
        self._emit_spectrum_tree_changed()
        self._emit_undo_redo_state()

    def dump_collection(
        self,
        path: str | Path | None = None,
        *,
        indent: int | None = None,
    ) -> None:
        """
        Persist the collection and metadata to disk and emit signals.

        Gzip and compression level follow ``AppParameters`` (see options dialog).

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

        Plain vs gzip is auto-detected from the path or file header.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file.
        mode : {\"append\", \"replace\", \"new\"}, default=\"replace\"
            Loading mode passed to the orchestrator.
        """
        self._orchestrator.load_collection(path, mode=mode)  # type: ignore[arg-type]
        self._emit_spectrum_tree_changed()
        self._emit_collection_and_undo_redo()

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

    def _emit_spectrum_tree_changed(self) -> None:
        """Emit spectrum treeChanged signal."""
        self.spectrumTreeChanged.emit()

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

    # ------------------------------------------------------------------
    # NN service helpers
    # ------------------------------------------------------------------

    def load_nn_model(self, model_path: str | Path) -> None:
        """
        Load or reload the NN model used by the segmenter pipeline.

        Parameters
        ----------
        model_path : str or Path
            Path to the NN model file.
        """
        self._orchestrator.load_nn_model(model_path)

    # ------------------------------------------------------------------
    # App parameters
    # ------------------------------------------------------------------

    def get_app_parameters(self) -> AppParameters:
        """
        Return the current application parameters used by the orchestrator.

        Returns
        -------
        AppParameters
            Mutable parameters instance backing app services.
        """
        return self._orchestrator.params

    def apply_app_parameters(self, params: AppParameters) -> None:
        """
        Apply updated application parameters to the orchestrator.

        Parameters
        ----------
        params : AppParameters
            Updated parameters to use for subsequent operations.
        """
        self._orchestrator._params = params
        self._orchestrator.reconfigure_services_from_params()
