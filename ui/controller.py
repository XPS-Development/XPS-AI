from pathlib import Path
from typing import Any, Literal, Sequence

from PySide6.QtCore import QObject, Signal

from app.command.changes import BaseChange, ParameterField
from app.command.commands import (
    Command,
    CompositeCommand,
    CreateBackgroundCommand,
    CreatePeakCommand,
    CreateRegionCommand,
    CreateSpectrumCommand,
    RemoveObjectCommand,
    RemoveMetadataCommand,
    SetMetadataCommand,
    UpdateMultipleParameterValuesCommand,
    UpdateParameterCommand,
    UpdateRegionSliceCommand,
)
from app.orchestration import AppOrchestrator, AppParameters, QueryService
from core.collection import CoreCollection
from core.metadata import Metadata
from core.objects import Spectrum


class ControllerWrapper(QObject):
    """
    Qt-aware wrapper around :class:`AppOrchestrator`.

    This wrapper owns a core collection and orchestrator instance, exposes a
    small, UI-friendly API surface, and emits Qt signals whenever the
    underlying model or undo/redo state changes. UI updates are split into
    ``spectrumHierarchyChanged``, ``plotNeedsRefresh``, ``propertiesNeedsRefresh``,
    and ``documentStateChanged`` so views refresh only when necessary.

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

    spectrumHierarchyChanged: Signal = Signal()
    plotNeedsRefresh: Signal = Signal()
    propertiesNeedsRefresh: Signal = Signal()
    documentStateChanged: Signal = Signal()
    undoRedoStateChanged: Signal = Signal(bool, bool)
    selectionChanged: Signal = Signal(object, object)

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
        cmd = self._orchestrator.peek_undo_command()
        if cmd is None:
            raise IndexError("Nothing to undo")
        self._orchestrator.undo()
        self._emit_ui_for_command(cmd)
        self._emit_undo_redo_state()

    def redo(self) -> None:
        """
        Redo the last undone command and emit signals.
        """
        cmd = self._orchestrator.peek_redo_command()
        if cmd is None:
            raise IndexError("Nothing to redo")
        self._orchestrator.redo()
        self._emit_ui_for_command(cmd)
        self._emit_undo_redo_state()

    def import_spectra(self, path: str | Path) -> None:
        """
        Import spectra from a file and emit signals.

        Parameters
        ----------
        path : str or Path
            Path to the spectra file.
        """
        self._orchestrator.import_spectra(path)
        self.spectrumHierarchyChanged.emit()
        self.documentStateChanged.emit()
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
        self._emit_fit_data_changed()

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
        self._emit_fit_data_changed()

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
        self._emit_fit_data_changed()

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
        self._emit_fit_data_changed()

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
        self._emit_fit_data_changed()

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
        self._emit_fit_data_changed()

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
        self._emit_fit_data_changed()

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
        self.spectrumHierarchyChanged.emit()
        self.documentStateChanged.emit()
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
        self._emit_fit_data_changed()

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
        self._emit_fit_data_changed()

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
        self._emit_fit_data_changed()

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
        self.spectrumHierarchyChanged.emit()
        self.propertiesNeedsRefresh.emit()
        self.documentStateChanged.emit()
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
        self.spectrumHierarchyChanged.emit()
        self.documentStateChanged.emit()
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
        self.spectrumHierarchyChanged.emit()
        self.documentStateChanged.emit()
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
        self.spectrumHierarchyChanged.emit()
        self.documentStateChanged.emit()
        self._emit_undo_redo_state()

    def remove_object(self, obj_id: str) -> None:
        """
        Remove an object and its descendants and emit signals.

        Parameters
        ----------
        obj_id : str
            Identifier of the object to remove.
        """
        obj: Any = None
        if self._orchestrator.query.check_object_exists(obj_id):
            obj = self._collection.get(obj_id)
        self._orchestrator.remove_object(obj_id)
        if isinstance(obj, Spectrum):
            self.spectrumHierarchyChanged.emit()
        self.plotNeedsRefresh.emit()
        self.propertiesNeedsRefresh.emit()
        self.documentStateChanged.emit()
        self._emit_undo_redo_state()

    def remove_metadata(self, obj_id: str) -> None:
        """
        Remove metadata for an object and emit signals.

        Parameters
        ----------
        obj_id : str
            Identifier of the object whose metadata to remove.
        """
        self._orchestrator.remove_metadata(obj_id)
        self.spectrumHierarchyChanged.emit()
        self.documentStateChanged.emit()
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
        self.spectrumHierarchyChanged.emit()
        self.plotNeedsRefresh.emit()
        self.propertiesNeedsRefresh.emit()
        self.documentStateChanged.emit()
        self._emit_undo_redo_state()

    def remove_spectrum(self, spectrum_id: str) -> None:
        """
        Remove a spectrum and emit signals.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum to remove.
        """
        self._orchestrator.full_remove_object(spectrum_id)
        self.spectrumHierarchyChanged.emit()
        self.plotNeedsRefresh.emit()
        self.propertiesNeedsRefresh.emit()
        self.documentStateChanged.emit()
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
        self.spectrumHierarchyChanged.emit()
        self.plotNeedsRefresh.emit()
        self.propertiesNeedsRefresh.emit()
        self.documentStateChanged.emit()
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
        self.spectrumHierarchyChanged.emit()
        self.plotNeedsRefresh.emit()
        self.propertiesNeedsRefresh.emit()
        self.documentStateChanged.emit()
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
        self.documentStateChanged.emit()
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
        self.emit_full_ui_refresh()

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

    def export_peak_parameters(
        self,
        spectrum_id: str,
        path: str | Path,
        *,
        normalized: bool = False,
        separator: str = ",",
        use_xps_peak_names: bool = False,
        precision: int | None = None,
    ) -> None:
        """
        Export peak parameters to a CSV-like file.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum.
        path : str or Path
            Output file path.
        normalized : bool, optional
            If True, export normalized parameters.
        separator : str, optional
            Column separator character.
        use_xps_peak_names : bool, optional
            If True, apply pseudo-voigt XPS aliases.
        """
        self._orchestrator.export_peak_parameters(
            spectrum_id=spectrum_id,
            path=path,
            normalized=normalized,
            separator=separator,
            use_xps_peak_names=use_xps_peak_names,
            precision=precision,
        )

    def export_spectrum(
        self,
        spectrum_id: str,
        path: str | Path,
        *,
        normalized: bool = False,
        separator: str = ",",
        include_evaluated_components: bool = False,
        include_background: bool = True,
        include_difference: bool = True,
        precision: int | None = None,
    ) -> None:
        """
        Export spectrum data to a CSV-like file.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum.
        path : str or Path
            Output file path.
        normalized : bool, optional
            If True, export normalized data.
        separator : str, optional
            Column separator character.
        include_evaluated_components : bool, optional
            If True, include evaluated model columns.
        include_background : bool, optional
            If True, include background model column.
        include_difference : bool, optional
            If True, include residual/difference column.
        """
        self._orchestrator.export_spectrum(
            spectrum_id=spectrum_id,
            path=path,
            normalized=normalized,
            separator=separator,
            include_evaluated_components=include_evaluated_components,
            include_background=include_background,
            include_difference=include_difference,
            precision=precision,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def emit_full_ui_refresh(self) -> None:
        """
        Emit all UI invalidation signals after a document-wide change.

        Used after loading a collection or creating a new workspace.
        """
        self.spectrumHierarchyChanged.emit()
        self.plotNeedsRefresh.emit()
        self.propertiesNeedsRefresh.emit()
        self.documentStateChanged.emit()
        self._emit_undo_redo_state()

    def _emit_fit_data_changed(self) -> None:
        """Emit signals after fit/slice/model changes that affect plot and properties."""
        self.plotNeedsRefresh.emit()
        self.propertiesNeedsRefresh.emit()
        self.documentStateChanged.emit()
        self._emit_undo_redo_state()

    def _emit_ui_for_command(self, cmd: Command) -> None:
        """Emit controller signals appropriate for the given undo/redo command."""
        hierarchy, plot, properties, document = self._ui_flags_for_command(cmd)
        if hierarchy:
            self.spectrumHierarchyChanged.emit()
        if plot:
            self.plotNeedsRefresh.emit()
        if properties:
            self.propertiesNeedsRefresh.emit()
        if document:
            self.documentStateChanged.emit()

    def _ui_flags_for_command(self, cmd: Command) -> tuple[bool, bool, bool, bool]:
        """
        Return (hierarchy, plot, properties, document) invalidation flags for a command.

        Used for undo/redo so only affected views refresh.
        """
        if isinstance(cmd, CompositeCommand):
            h = p = pr = d = False
            for sub in cmd.commands:
                sh, sp, spr, sd = self._ui_flags_for_command(sub)
                h |= sh
                p |= sp
                pr |= spr
                d |= sd
            return (h, p, pr, d)
        if isinstance(
            cmd,
            (UpdateParameterCommand, UpdateRegionSliceCommand, UpdateMultipleParameterValuesCommand),
        ):
            return (False, True, True, True)
        if isinstance(cmd, SetMetadataCommand):
            return (True, False, True, True)
        if isinstance(cmd, RemoveMetadataCommand):
            return (True, False, False, True)
        if isinstance(cmd, RemoveObjectCommand):
            return self._ui_flags_for_remove_object_command(cmd)
        if isinstance(cmd, CreateSpectrumCommand):
            return (True, True, True, True)
        if isinstance(cmd, CreateRegionCommand | CreatePeakCommand | CreateBackgroundCommand):
            return (False, True, True, True)
        return (True, True, True, True)

    def _ui_flags_for_remove_object_command(self, cmd: RemoveObjectCommand) -> tuple[bool, bool, bool, bool]:
        """
        Return (hierarchy, plot, properties, document) for a remove-object command.

        When the detached subtree root is a spectrum, the spectrum tree changes.
        """
        objs = cmd.objs
        if objs is None or len(objs) == 0:
            return (True, True, True, True)
        root = objs[0]
        hierarchy = isinstance(root, Spectrum)
        return (hierarchy, True, True, True)

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

        Refreshes all views that depend on app settings (spectrum tree, plot,
        properties, and document/title/status).

        Parameters
        ----------
        params : AppParameters
            Updated parameters to use for subsequent operations.
        """
        self._orchestrator._params = params
        self._orchestrator.reconfigure_services_from_params()
        self.spectrumHierarchyChanged.emit()
        self.plotNeedsRefresh.emit()
        self.propertiesNeedsRefresh.emit()
        self.documentStateChanged.emit()
