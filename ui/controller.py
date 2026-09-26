"""Qt controller wrapping :class:`AppOrchestrator` with UI signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

from PySide6.QtCore import QObject, Signal

from app.command.refresh import UiRefresh
from app.orchestration import AppOrchestrator
from app.parameters import AppParameters
from core.collection import CoreCollection

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from app.command.changes import ParameterField
    from app.command.commands import Command
    from app.query_service import QueryService
    from app.usecases.analysis import FitScopePreview
    from core.metadata import Metadata

_T = TypeVar("_T")


class ControllerWrapper(QObject):
    """
    Qt-aware wrapper around :class:`AppOrchestrator`.

    Owns selection state, forwards mutations to the orchestrator, and emits Qt
    signals from command :class:`~app.command.refresh.UiRefresh` flags.
    """

    spectrumHierarchyChanged: Signal = Signal()
    plotNeedsRefresh: Signal = Signal()
    propertiesNeedsRefresh: Signal = Signal()
    documentStateChanged: Signal = Signal()
    undoRedoStateChanged: Signal = Signal(bool, bool)
    selectionChanged: Signal = Signal(object, object, object)

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
        self._selected_component_id: str | None = None

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
    def can_undo(self) -> bool:
        """True if there is at least one command to undo."""
        return self._orchestrator.can_undo

    @property
    def can_redo(self) -> bool:
        """True if there is at least one command to redo."""
        return self._orchestrator.can_redo

    @property
    def is_dirty(self) -> bool:
        """True if there are unsaved document changes."""
        return self._orchestrator.is_dirty

    @property
    def selected_spectrum_id(self) -> str | None:
        """Identifier of the currently selected spectrum."""
        return self._selected_spectrum_id

    @property
    def selected_region_id(self) -> str | None:
        """Identifier of the currently selected region."""
        return self._selected_region_id

    @property
    def selected_component_id(self) -> str | None:
        """Identifier of the currently selected peak or background component."""
        return self._selected_component_id

    def set_selection(
        self,
        spectrum_id: str | None,
        region_id: str | None = None,
        component_id: str | None = None,
    ) -> None:
        """
        Update the current spectrum/region/component selection.

        Parameters
        ----------
        spectrum_id : str or None
            Selected spectrum, or ``None`` to clear selection.
        region_id : str or None, optional
            Selected region within the spectrum.
        component_id : str or None, optional
            Selected peak or background within the region.
        """
        if (
            spectrum_id == self._selected_spectrum_id
            and region_id == self._selected_region_id
            and component_id == self._selected_component_id
        ):
            return
        self._selected_spectrum_id = spectrum_id
        self._selected_region_id = region_id
        self._selected_component_id = component_id
        self.selectionChanged.emit(spectrum_id, region_id, component_id)

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

    def import_spectra(self, path: str | Path | Sequence[str | Path]) -> None:
        """Import spectra from one or more files and emit signals."""
        self._mutate(self._orchestrator.import_spectra, path)

    def copy_decomposition(
        self,
        source_spectrum_id: str,
        target_spectrum_ids: Sequence[str],
        link_flags: Mapping[tuple[str, str], bool],
        *,
        rescale_intensities: bool = True,
        overwrite_targets: set[str] | frozenset[str] | None = None,
        optimize_after: bool = False,
    ) -> list[str]:
        """Copy a spectrum decomposition onto targets and emit signals."""
        result: list[str] = []

        def _run() -> None:
            nonlocal result
            result = self._orchestrator.copy_decomposition(
                source_spectrum_id,
                target_spectrum_ids,
                link_flags,
                rescale_intensities=rescale_intensities,
                overwrite_targets=overwrite_targets,
                optimize_after=optimize_after,
            )

        self._mutate(_run)
        return result

    def run_segmenter(self, spectrum_ids: Sequence[str]) -> None:
        """Run the segmenter pipeline and emit signals."""
        self._mutate(self._orchestrator.run_segmenter, spectrum_ids)

    def preview_fit_scope(
        self,
        *,
        region_ids: Sequence[str] | None = None,
        spectrum_ids: Sequence[str] | None = None,
    ) -> FitScopePreview:
        """
        Preview expression-closed regions for an upcoming optimize (read-only).

        Parameters
        ----------
        region_ids
            Regions the user selected.
        spectrum_ids
            Spectra whose regions form the selection.

        Returns
        -------
        FitScopePreview
            Selected vs expanded region sets for a confirmation dialog.
        """
        return self._orchestrator.preview_fit_scope(
            region_ids=region_ids,
            spectrum_ids=spectrum_ids,
        )

    def optimize_regions(
        self,
        *,
        region_ids: Sequence[str] | None = None,
        spectrum_ids: Sequence[str] | None = None,
        expand_linked: bool = True,
        **kwargs: Any,
    ) -> None:
        """Run optimization for regions and emit signals."""
        self._mutate(
            self._orchestrator.optimize_regions,
            region_ids=region_ids,
            spectrum_ids=spectrum_ids,
            expand_linked=expand_linked,
            **kwargs,
        )

    def auto_fit(self, spectrum_ids: Sequence[str], **kwargs: Any) -> None:
        """Run the segmenter then optimize regions for the given spectra."""
        self._mutate(self._orchestrator.auto_fit, spectrum_ids, **kwargs)

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
            self._orchestrator.update_parameter,
            component_id=component_id,
            name=name,
            parameter_field=parameter_field,
            new_value=new_value,
            normalized=normalized,
        )

    def preview_parameter_value(
        self,
        component_id: str,
        name: str,
        value: float,
        *,
        normalized: bool = False,
    ) -> None:
        """Live-preview a parameter value without recording undo."""
        self._mutate(
            self._orchestrator.preview_parameter_value,
            component_id,
            name,
            value,
            normalized=normalized,
        )

    def commit_parameter_preview(
        self,
        component_id: str,
        name: str,
        old_value: float,
        new_value: float,
        *,
        normalized: bool = False,
    ) -> None:
        """Record undo for a value already applied via :meth:`preview_parameter_value`."""
        self._mutate(
            self._orchestrator.commit_parameter_preview,
            component_id,
            name,
            old_value,
            new_value,
            normalized=normalized,
        )

    def preview_region_slice(
        self,
        region_id: str,
        start: int | float,
        stop: int | float,
        *,
        mode: Literal["value", "index"] = "value",
    ) -> None:
        """Live-preview a region slice without recording undo."""
        self._mutate(
            self._orchestrator.preview_region_slice,
            region_id,
            start,
            stop,
            mode=mode,
        )

    def commit_region_slice_preview(
        self,
        region_id: str,
        old_start_index: int,
        old_stop_index: int,
    ) -> None:
        """Record undo for a slice already applied via :meth:`preview_region_slice`."""
        self._mutate(
            self._orchestrator.commit_region_slice_preview,
            region_id,
            old_start_index,
            old_stop_index,
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
            self._orchestrator.update_region_slice,
            region_id,
            start,
            stop,
            mode=mode,
        )

    def split_region(self, region_id: str, x: float) -> None:
        """Split a region at axis position ``x`` and emit signals."""
        self._mutate(self._orchestrator.split_region, region_id, x)

    def replace_peak_model(
        self,
        peak_id: str,
        new_model_name: str,
        parameters: dict[str, float] | None = None,
    ) -> None:
        """Replace a peak's model and emit signals."""
        self._mutate(
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
            self._orchestrator.create_region,
            spectrum_id=spectrum_id,
            start=start,
            stop=stop,
            region_id=region_id,
            mode=mode,
        )

    def create_cursor_peak(self, region_id: str, cen: float, height: float) -> None:
        """Create a pseudo-Voigt peak from a plot click and emit signals."""
        self._mutate(self._orchestrator.create_cursor_peak, region_id, cen, height)

    def create_peak(
        self,
        region_id: str,
        model_name: str,
        parameters: dict[str, float] | None = None,
        peak_id: str | None = None,
        peak_index: int | None = None,
    ) -> None:
        """Create a new peak component and emit signals."""
        self._mutate(
            self._orchestrator.create_peak,
            region_id=region_id,
            model_name=model_name,
            parameters=parameters,
            peak_id=peak_id,
            peak_index=peak_index,
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
            self._orchestrator.create_background,
            region_id=region_id,
            model_name=model_name,
            parameters=parameters,
            background_id=background_id,
        )

    def set_metadata(self, obj_id: str, metadata: Metadata) -> None:
        """Set metadata for an object and emit signals."""
        self._mutate(self._orchestrator.set_metadata, obj_id, metadata)

    def rename_spectrum(self, spectrum_id: str, new_name: str) -> None:
        """Rename a single spectrum and emit UI refresh signals."""
        self._mutate(self._orchestrator.rename_spectrum, spectrum_id, new_name)

    def rename_component(self, component_id: str, new_name: str | None) -> None:
        """Set a peak/background display name and emit UI refresh signals."""
        self._mutate(self._orchestrator.rename_component, component_id, new_name)

    def rename_group(self, file_label: str, old_group_label: str, new_group_label: str) -> None:
        """Rename a group within a file and emit UI refresh signals."""
        self._mutate(
            self._orchestrator.rename_group,
            file_label,
            old_group_label,
            new_group_label,
        )

    def rename_file(self, old_file_label: str, new_file_label: str) -> None:
        """Rename a file bucket and emit UI refresh signals."""
        self._mutate(self._orchestrator.rename_file, old_file_label, new_file_label)

    def remove_object(self, obj_id: str) -> None:
        """Remove an object and its descendants and emit signals."""

        def _run() -> None:
            self._orchestrator.remove_object(obj_id)
            self._ensure_selection_valid()

        self._mutate(_run)

    def remove_metadata(self, obj_id: str) -> None:
        """Remove metadata for an object and emit signals."""
        self._mutate(self._orchestrator.remove_metadata, obj_id)

    def full_remove_object(self, obj_id: str) -> None:
        """Remove an object, all descendants, and their metadata and emit signals."""

        def _run() -> None:
            self._orchestrator.full_remove_object(obj_id)
            self._ensure_selection_valid()

        self._mutate(_run)

    def remove_group(self, file_label: str, group_label: str) -> None:
        """Remove all spectra belonging to a given file/group combination."""

        def _run() -> None:
            self._orchestrator.remove_group(file_label, group_label)
            self._ensure_selection_valid()

        self._mutate(_run)

    def remove_file(self, file_label: str) -> None:
        """Remove all spectra associated with a given file label."""

        def _run() -> None:
            self._orchestrator.remove_file(file_label)
            self._ensure_selection_valid()

        self._mutate(_run)

    def dump_collection(
        self,
        path: str | Path | None = None,
        *,
        indent: int | None = None,
    ) -> None:
        """Persist the collection and metadata to disk and emit signals."""
        self._mutate(
            self._orchestrator.dump_collection,
            path=path,
            indent=indent,
        )

    def load_collection(
        self,
        path: str | Path,
        *,
        mode: Literal["append", "replace"] | None = None,
    ) -> None:
        """Load collection and metadata from disk and emit signals."""
        self._orchestrator.load_collection(path, mode=mode)
        self.emit_full_ui_refresh()

    def new_collection(self) -> None:
        """Clear the document and emit a full UI refresh."""
        self._mutate(self._orchestrator.new_collection)

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
        self._orchestrator.apply_params(params)
        self._emit_refresh(UiRefresh.ALL)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def emit_full_ui_refresh(self) -> None:
        """Emit all UI invalidation signals after a document-wide change."""
        self._emit_refresh(UiRefresh.ALL)

    def _mutate(
        self,
        fn: Callable[..., _T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Run an orchestrator mutation and emit refresh flags from the command layer."""
        result = fn(*args, **kwargs)
        self._emit_refresh(self._orchestrator.consume_pending_ui_refresh())
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
        self._ensure_selection_valid()
        self._emit_refresh(cmd.combined_ui_refresh())

    def _ensure_selection_valid(self) -> None:
        """Clear selection fields that no longer exist in the collection."""
        spectrum_id = self._selected_spectrum_id
        region_id = self._selected_region_id
        component_id = self._selected_component_id
        query = self.query

        if spectrum_id is not None and not query.check_object_exists(spectrum_id):
            self.set_selection(None)
            return
        if region_id is not None and not query.check_object_exists(region_id):
            self.set_selection(spectrum_id, None)
            return
        if component_id is not None and not query.check_object_exists(component_id):
            self.set_selection(spectrum_id, region_id, None)

    def _emit_undo_redo_state(self) -> None:
        """Emit the current undo/redo capability state."""
        self.undoRedoStateChanged.emit(
            self._orchestrator.can_undo,
            self._orchestrator.can_redo,
        )
