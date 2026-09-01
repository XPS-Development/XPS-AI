"""Qt controller wrapping :class:`AppOrchestrator` with UI signals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

from PySide6.QtCore import QObject, Signal

from app.command.changes import ParameterField
from app.command.commands import Command
from app.command.refresh import UiRefresh
from app.orchestration import AppOrchestrator
from app.parameters import AppParameters
from app.query_service import QueryService
from core.collection import CoreCollection
from core.metadata import Metadata
from core.objects import Spectrum

_T = TypeVar("_T")


class ControllerWrapper(QObject):
    """
    Qt-aware wrapper around :class:`AppOrchestrator`.

    Owns selection state, forwards mutations to the orchestrator, and emits Qt
    signals when the underlying model or undo/redo state changes.
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

    @property
    def collection(self) -> CoreCollection:
        """Underlying core collection."""
        return self._collection

    @property
    def orchestrator(self) -> AppOrchestrator:
        """Underlying application orchestrator."""
        return self._orchestrator

    @property
    def query(self) -> QueryService:
        """Read-only query façade bound to the orchestrator."""
        return self._orchestrator.query

    @property
    def selected_spectrum_id(self) -> str | None:
        """Identifier of the currently selected spectrum."""
        return self._selected_spectrum_id

    @property
    def selected_region_id(self) -> str | None:
        """Identifier of the currently selected region."""
        return self._selected_region_id

    def set_selection(self, spectrum_id: str | None, region_id: str | None = None) -> None:
        """Update the current spectrum/region selection."""
        if spectrum_id == self._selected_spectrum_id and region_id == self._selected_region_id:
            return
        self._selected_spectrum_id = spectrum_id
        self._selected_region_id = region_id
        self.selectionChanged.emit(spectrum_id, region_id)

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def undo(self) -> None:
        """Undo the last executed command and emit signals."""
        cmd = self._orchestrator.peek_undo_command()
        if cmd is None:
            return
        self._orchestrator.undo()
        self._emit_ui_for_command(cmd)

    def redo(self) -> None:
        """Redo the last undone command and emit signals."""
        cmd = self._orchestrator.peek_redo_command()
        if cmd is None:
            return
        self._orchestrator.redo()
        self._emit_ui_for_command(cmd)

    def import_spectra(self, path: str | Path) -> None:
        """Import spectra from a file and emit signals."""
        self._mutate(
            UiRefresh.HIERARCHY | UiRefresh.DOCUMENT,
            self._orchestrator.import_spectra,
            path,
        )

    def run_segmenter(self, spectrum_ids: Sequence[str]) -> None:
        """Run the segmenter pipeline and emit signals."""
        self._mutate(UiRefresh.FIT, self._orchestrator.run_segmenter, spectrum_ids)

    def optimize_regions(
        self,
        *,
        region_ids: Sequence[str] | None = None,
        spectrum_ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run optimization for regions and emit signals."""
        self._mutate(
            UiRefresh.FIT,
            self._orchestrator.optimize_regions,
            region_ids=region_ids,
            spectrum_ids=spectrum_ids,
            **kwargs,
        )

    def auto_fit_spectra(self, spectrum_ids: Sequence[str], **kwargs: Any) -> None:
        """Run the segmenter then optimize regions for the given spectra."""
        self._mutate(UiRefresh.FIT, self._orchestrator.auto_fit, spectrum_ids, **kwargs)

    def update_parameter(
        self,
        component_id: str,
        name: str,
        parameter_field: ParameterField,
        new_value: str | bool | float | None,
        *,
        normalized: bool = False,
    ) -> None:
        """Update a single parameter attribute and emit signals."""
        self._mutate(
            UiRefresh.FIT,
            self._orchestrator.update_parameter,
            component_id=component_id,
            name=name,
            parameter_field=parameter_field,
            new_value=new_value,
            normalized=normalized,
        )

    def update_parameters(
        self,
        component_id: str,
        parameters: dict[str, str | bool | float],
        *,
        normalized: bool = False,
    ) -> None:
        """Update multiple parameter values at once and emit signals."""
        self._mutate(
            UiRefresh.FIT,
            self._orchestrator.update_parameters,
            component_id=component_id,
            parameters=parameters,
            normalized=normalized,
        )

    def update_region_slice(
        self,
        region_id: str,
        start: int | float | None = None,
        stop: int | float | None = None,
        mode: Literal["value", "index"] = "index",
    ) -> None:
        """Update the slice of a region and emit signals."""
        self._mutate(
            UiRefresh.FIT,
            self._orchestrator.update_region_slice,
            region_id,
            start,
            stop,
            mode=mode,
        )

    def replace_peak_model(
        self,
        peak_id: str,
        new_model_name: str,
        parameters: dict[str, float] | None = None,
    ) -> None:
        """Replace a peak's model and emit signals."""
        self._mutate(
            UiRefresh.FIT,
            self._orchestrator.replace_peak_model,
            peak_id,
            new_model_name,
            parameters=parameters,
        )

    def replace_background_model(
        self,
        region_id: str,
        new_model_name: str,
        parameters: dict[str, float] | None = None,
        background_id: str | None = None,
    ) -> None:
        """Replace the background model for a region and emit signals."""
        self._mutate(
            UiRefresh.FIT,
            self._orchestrator.replace_background_model,
            region_id,
            new_model_name,
            parameters=parameters,
            background_id=background_id,
        )

    def create_spectrum(
        self,
        x: Any,
        y: Any,
        spectrum_id: str | None = None,
    ) -> None:
        """Create a new spectrum and emit signals."""
        self._mutate(
            UiRefresh.ALL,
            self._orchestrator.create_spectrum,
            x=x,
            y=y,
            spectrum_id=spectrum_id,
        )

    def create_region(
        self,
        spectrum_id: str,
        start: int | float | None = None,
        stop: int | float | None = None,
        region_id: str | None = None,
        mode: Literal["value", "index"] = "index",
    ) -> None:
        """Create a new region and emit signals."""
        self._mutate(
            UiRefresh.FIT,
            self._orchestrator.create_region,
            spectrum_id=spectrum_id,
            start=start,
            stop=stop,
            region_id=region_id,
            mode=mode,
        )

    def create_peak(
        self,
        region_id: str,
        model_name: str,
        parameters: dict[str, float] | None = None,
        peak_id: str | None = None,
    ) -> None:
        """Create a new peak component and emit signals."""
        self._mutate(
            UiRefresh.FIT,
            self._orchestrator.create_peak,
            region_id=region_id,
            model_name=model_name,
            parameters=parameters,
            peak_id=peak_id,
        )

    def create_peak_and_return_id(
        self,
        region_id: str,
        model_name: str,
        parameters: dict[str, float] | None = None,
        peak_id: str | None = None,
    ) -> str:
        """
        Create a peak and return its identifier.

        Parameters
        ----------
        region_id : str
            Parent region identifier.
        model_name : str
            Registered peak model name.
        parameters : dict[str, float] or None, optional
            Explicit parameter values.
        peak_id : str or None, optional
            Optional explicit peak identifier.

        Returns
        -------
        str
            Identifier of the created peak.
        """
        return self._mutate(
            UiRefresh.FIT,
            self._orchestrator.create_peak_and_return_id,
            region_id=region_id,
            model_name=model_name,
            parameters=parameters,
            peak_id=peak_id,
        )

    def create_background(
        self,
        region_id: str,
        model_name: str,
        parameters: dict[str, float] | None = None,
        background_id: str | None = None,
    ) -> None:
        """Create or replace a background component and emit signals."""
        self._mutate(
            UiRefresh.FIT,
            self._orchestrator.create_background,
            region_id=region_id,
            model_name=model_name,
            parameters=parameters,
            background_id=background_id,
        )

    def set_metadata(self, obj_id: str, metadata: Metadata) -> None:
        """Set metadata for an object and emit signals."""
        self._mutate(UiRefresh.METADATA, self._orchestrator.set_metadata, obj_id, metadata)

    def rename_spectrum(self, spectrum_id: str, new_name: str) -> None:
        """Rename a single spectrum and emit collection/undo-redo signals."""
        self._mutate(
            UiRefresh.HIERARCHY | UiRefresh.DOCUMENT,
            self._orchestrator.rename_spectrum,
            spectrum_id,
            new_name,
        )

    def rename_group(self, file_label: str, old_group_label: str, new_group_label: str) -> None:
        """Rename a group within a file and emit collection/undo-redo signals."""
        self._mutate(
            UiRefresh.HIERARCHY | UiRefresh.DOCUMENT,
            self._orchestrator.rename_group,
            file_label,
            old_group_label,
            new_group_label,
        )

    def rename_file(self, old_file_label: str, new_file_label: str) -> None:
        """Rename a file bucket and emit collection/undo-redo signals."""
        self._mutate(
            UiRefresh.HIERARCHY | UiRefresh.DOCUMENT,
            self._orchestrator.rename_file,
            old_file_label,
            new_file_label,
        )

    def remove_object(self, obj_id: str) -> None:
        """Remove an object and its descendants and emit signals."""
        flags = UiRefresh.FIT
        if self._orchestrator.query.check_object_exists(obj_id):
            if isinstance(self._collection.get(obj_id), Spectrum):
                flags |= UiRefresh.HIERARCHY
        self._mutate(flags, self._orchestrator.remove_object, obj_id)

    def remove_metadata(self, obj_id: str) -> None:
        """Remove metadata for an object and emit signals."""
        self._mutate(
            UiRefresh.HIERARCHY | UiRefresh.DOCUMENT,
            self._orchestrator.remove_metadata,
            obj_id,
        )

    def full_remove_object(self, obj_id: str) -> None:
        """Remove an object, all descendants, and their metadata and emit signals."""
        self._mutate(UiRefresh.ALL, self._orchestrator.full_remove_object, obj_id)

    def remove_spectrum(self, spectrum_id: str) -> None:
        """Remove a spectrum and emit signals."""
        self._mutate(UiRefresh.ALL, self._orchestrator.full_remove_object, spectrum_id)

    def remove_group(self, file_label: str, group_label: str) -> None:
        """Remove all spectra belonging to a given file/group combination."""
        self._mutate(UiRefresh.ALL, self._orchestrator.remove_group, file_label, group_label)

    def remove_file(self, file_label: str) -> None:
        """Remove all spectra associated with a given file label."""
        self._mutate(UiRefresh.ALL, self._orchestrator.remove_file, file_label)

    def dump_collection(
        self,
        path: str | Path | None = None,
        *,
        indent: int | None = None,
    ) -> None:
        """Persist the collection and metadata to disk and emit signals."""
        self._mutate(
            UiRefresh.DOCUMENT,
            self._orchestrator.dump_collection,
            path=path,
            indent=indent,
        )

    def load_collection(
        self,
        path: str | Path,
        *,
        mode: Literal["append", "replace"] = "replace",
    ) -> None:
        """Load collection and metadata from disk and emit signals."""
        self._orchestrator.load_collection(path, mode=mode)
        self.emit_full_ui_refresh()

    def set_default_save_path(self, path: str | Path) -> None:
        """Set the default save path in the serialization service."""
        self._orchestrator.set_default_save_path(path)

    def get_default_save_path(self) -> Path | None:
        """Return the current default save path."""
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
        """Export peak parameters to a CSV-like file."""
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
        """Export spectrum data to a CSV-like file."""
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

    def load_nn_model(self, model_path: str | Path) -> None:
        """Load or reload the NN model used by the segmenter pipeline."""
        self._orchestrator.load_nn_model(model_path)

    def get_app_parameters(self) -> AppParameters:
        """Return the current application parameters used by the orchestrator."""
        return self._orchestrator.params

    def apply_app_parameters(self, params: AppParameters) -> None:
        """Apply updated application parameters to the orchestrator."""
        self._orchestrator._params = params
        self._orchestrator.reconfigure_services_from_params()
        self._emit_refresh(UiRefresh.ALL)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def emit_full_ui_refresh(self) -> None:
        """Emit all UI invalidation signals after a document-wide change."""
        self._emit_refresh(UiRefresh.ALL)

    def _mutate(
        self,
        flags: UiRefresh,
        fn: Callable[..., _T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Run an orchestrator mutation and emit the requested UI refresh flags."""
        result = fn(*args, **kwargs)
        self._emit_refresh(flags)
        return result

    def _emit_refresh(self, flags: UiRefresh) -> None:
        """Emit Qt signals for the given refresh bit flags."""
        if flags & UiRefresh.HIERARCHY:
            self.spectrumHierarchyChanged.emit()
        if flags & UiRefresh.PLOT:
            self.plotNeedsRefresh.emit()
        if flags & UiRefresh.PROPERTIES:
            self.propertiesNeedsRefresh.emit()
        if flags & UiRefresh.DOCUMENT:
            self.documentStateChanged.emit()
        self._emit_undo_redo_state()

    def _emit_ui_for_command(self, cmd: Command) -> None:
        """Emit controller signals appropriate for the given undo/redo command."""
        self._emit_refresh(cmd.combined_ui_refresh())

    def _emit_undo_redo_state(self) -> None:
        """Emit the current undo/redo capability state."""
        self.undoRedoStateChanged.emit(
            self._orchestrator.can_undo,
            self._orchestrator.can_redo,
        )
