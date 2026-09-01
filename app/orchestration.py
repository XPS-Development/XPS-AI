"""
Thin orchestration layer for spectra analysis.

Aggregates app services and the command/change pipeline into a single entry point
for running services, applying changes (create/update/metadata/remove), and undo/redo.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from numpy.typing import NDArray

from core.collection import CoreCollection
from core.dto import ComponentDTO
from core.metadata import Metadata
from core.services import CoreContext

from .automatization import AutomatizationAdapter
from .command.changes import (
    BaseChange,
    CreateRegion,
    CreateSpectrum,
    FullRemoveObject,
    ParameterField,
    RemoveMetadata,
    RemoveObject,
    SetMetadata,
    UpdateMultipleParameterValues,
    UpdateParameter,
)
from .command.commands import Command
from .command.core import CommandExecutor, UndoRedoStack, create_default_registry
from .csv_export import CSVExportService
from .error_dump import apply_safe_execution_to_class
from .import_service import import_spectra as import_spectra_changes
from .nn_service import NNService
from .optimization import OptimizationService
from .parameters import AppParameters
from .query_service import QueryService
from .serialization import SerializationService
from .usecases import AnalysisUseCases, EditingUseCases, HierarchyUseCases


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
            peak_model_name=params.default_peak_model,
            background_model_name=params.default_background_model,
        )
        self._optimization = OptimizationService()
        self._automatization = AutomatizationAdapter()
        self._serialization = SerializationService()
        self._csv_export = CSVExportService()
        self._editing = EditingUseCases(self._query, self._automatization, params)
        self._analysis = AnalysisUseCases(self._query, self._nn, self._optimization, params)
        self._hierarchy = HierarchyUseCases(self._query)

    @property
    def core_collection(self) -> CoreCollection:
        """The core spectrum collection (mutable state)."""
        return self._core_collection

    @property
    def ctx(self) -> CoreContext:
        """
        Core services context (internal; for tests and low-level access only).

        Prefer using :attr:`query` for read operations in application code.
        """
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
    def params(self) -> AppParameters:
        """
        Application parameters governing import, NN, optimization, and serialization.

        Returns
        -------
        AppParameters
            Mutable parameter set used by the orchestrator and its services.
        """
        return self._params

    def reconfigure_services_from_params(self) -> None:
        """
        Reconfigure internal services to reflect the current AppParameters values.

        Currently this refreshes the NN service with updated NN-related
        parameters; other services read parameters lazily when invoked.
        """
        self._nn = NNService(
            model_path=self._params.nn_model_path,
            pred_threshold=self._params.nn_pred_threshold,
            smooth=self._params.nn_smooth,
            interp_num=self._params.nn_interp_num,
            peak_model_name=self._params.default_peak_model,
            background_model_name=self._params.default_background_model,
        )
        self._analysis.set_nn(self._nn)

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

    def peek_undo_command(self) -> Command | None:
        """
        Return the command that would be undone next, without modifying stacks.

        Returns
        -------
        Command or None
            The top undo command, or None if nothing to undo.
        """
        return self._executor.peek_undo_command()

    def peek_redo_command(self) -> Command | None:
        """
        Return the command that would be redone next, without modifying stacks.

        Returns
        -------
        Command or None
            The top redo command, or None if nothing to redo.
        """
        return self._executor.peek_redo_command()

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
        change = self._analysis.run_segmenter(spectrum_ids)
        if change is None:
            return
        self.execute(change)

    def auto_fit(self, spectrum_ids: Sequence[str], **kwargs) -> None:
        """
        Run the NN segmenter then region optimization for the given spectra.

        This performs two separate command executions (two undo steps): first
        :meth:`run_segmenter`, then :meth:`optimize_regions`, so optimization
        sees regions created by the segmenter.

        Parameters
        ----------
        spectrum_ids : Sequence of str
            Parent spectrum identifiers to process.
        **kwargs
            Forwarded to :meth:`optimize_regions` (merged with
            ``AppParameters.optimization_kwargs``).

        Notes
        -----
        If ``spectrum_ids`` is empty, this method returns without doing anything.
        """
        if not spectrum_ids:
            return
        self.run_segmenter(spectrum_ids)
        self.optimize_regions(spectrum_ids=spectrum_ids, **kwargs)

    def optimize_regions(
        self,
        *,
        region_ids: Sequence[str] | None = None,
        spectrum_ids: Sequence[str] | None = None,
        **kwargs,
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
        self.execute(
            self._analysis.optimize_regions(
                region_ids=region_ids,
                spectrum_ids=spectrum_ids,
                **kwargs,
            )
        )

    # ---- Parameters and models ----

    def update_parameter(
        self,
        component_id: str,
        name: str,
        parameter_field: ParameterField,
        new_value: str | bool | float | None,
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
        parameters_f: dict[str, float] = {k: float(v) for k, v in parameters.items()}
        self.execute(
            UpdateMultipleParameterValues(
                component_id=component_id,
                parameters=parameters_f,
                normalized=normalized,
            )
        )

    def update_region_slice(
        self,
        region_id: str,
        start: int | float | None = None,
        stop: int | float | None = None,
        mode: Literal["value", "index"] = "index",
    ) -> None:
        """Update the index slice of an existing region; executed as a command."""
        self.execute(self._editing.update_region_slice(region_id, start, stop, mode=mode))

    def replace_peak_model(
        self,
        peak_id: str,
        new_model_name: str,
        parameters: dict[str, float] | None = None,
    ) -> None:
        """Replace a peak's model (and optionally parameters); executed as a command."""
        self.execute(
            self._editing.replace_peak_model(
                peak_id,
                new_model_name,
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
            self._editing.replace_background_model(
                region_id,
                new_model_name,
                parameters=parameters,
                background_id=background_id,
            )
        )

    # ---- Create ----

    def create_spectrum(
        self,
        x: NDArray,
        y: NDArray,
        spectrum_id: str | None = None,
    ) -> None:
        """Create a new spectrum; executed as a command."""
        self.execute(CreateSpectrum(x=x, y=y, spectrum_id=spectrum_id))

    def create_region(
        self,
        spectrum_id: str,
        start: int | float | None = None,
        stop: int | float | None = None,
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
        self.execute(
            self._editing.create_peak(
                region_id,
                model_name,
                parameters=parameters,
                peak_id=peak_id,
            )
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
        change, resolved_id = self._editing.create_peak_and_return_id(
            region_id,
            model_name,
            parameters=parameters,
            peak_id=peak_id,
        )
        self.execute(change)
        return resolved_id

    def create_background(
        self,
        region_id: str,
        model_name: str,
        parameters: dict[str, float] | None = None,
        background_id: str | None = None,
    ) -> None:
        """Create or replace a background component; executed as a command."""
        self.execute(
            self._editing.create_background(
                region_id,
                model_name,
                parameters=parameters,
                background_id=background_id,
            )
        )

    # ---- Metadata ----

    def set_metadata(self, obj_id: str, metadata: Metadata) -> None:
        """Set metadata for an object; executed as a command."""
        self.execute(SetMetadata(obj_id=obj_id, metadata=metadata))

    def rename_spectrum(self, spectrum_id: str, new_name: str) -> None:
        """
        Rename a single spectrum by updating its SpectrumMetadata.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum to rename.
        new_name : str
            New display name for the spectrum.
        """
        self.execute(self._hierarchy.rename_spectrum(spectrum_id, new_name))

    def rename_group(self, file_label: str, old_group_label: str, new_group_label: str) -> None:
        """
        Rename a group within a file by updating SpectrumMetadata.group.

        Parameters
        ----------
        file_label : str
            File label whose group should be renamed.
        old_group_label : str
            Existing group label.
        new_group_label : str
            New group label.
        """
        change = self._hierarchy.rename_group(file_label, old_group_label, new_group_label)
        if change is not None:
            self.execute(change)

    def rename_file(self, old_file_label: str, new_file_label: str) -> None:
        """
        Rename a file bucket by updating SpectrumMetadata.file for all spectra.

        Parameters
        ----------
        old_file_label : str
            Existing file label.
        new_file_label : str
            New file label.
        """
        change = self._hierarchy.rename_file(old_file_label, new_file_label)
        if change is not None:
            self.execute(change)

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
        change = self._hierarchy.remove_group(file_label, group_label)
        if change is not None:
            self.execute(change)

    def remove_file(self, file_label: str) -> None:
        """
        Remove all spectra associated with a given file label.

        Parameters
        ----------
        file_label : str
            File label whose spectra should be removed.
        """
        change = self._hierarchy.remove_file(file_label)
        if change is not None:
            self.execute(change)

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
        Gzip output and compression level are controlled only by
        ``AppParameters.default_serialization_use_gzip`` and
        ``AppParameters.default_serialization_compresslevel``.

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
            raise ValueError(
                "path is required when AppParameters.default_serialization_path is not set"
            )
        resolved_indent = (
            indent if indent is not None else self._params.default_serialization_indent
        )
        self._serialization.dump(
            path=resolved_path,
            collection=self._core_collection,
            metadata_service=self.__ctx.metadata,
            indent=resolved_indent,
            use_gzip=self._params.default_serialization_use_gzip,
            compresslevel=self._params.default_serialization_compresslevel,
        )
        self.set_default_save_path(resolved_path)

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
        Plain vs gzip input is auto-detected (``.gz`` suffix or gzip magic bytes).

        Parameters
        ----------
        path : str or Path
            Path to the JSON file (plain or gzip-compressed).
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

        self.set_default_save_path(path)

        if resolved_mode == "replace":
            self._executor.clear()

    def new_collection(self) -> None:
        """
        Clear collection, metadata, undo stack, and reset save path.

        Used when creating a new document (e.g. File > New).
        """
        self._core_collection.clear()
        self.__ctx.metadata.clear()
        self._params.default_serialization_path = None
        self._serialization.mark_dirty()
        self._executor.clear()

    # ---- CSV export ----

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
        Export peak parameters for all peaks in a spectrum to a CSV-like file.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum whose peaks are exported.
        path : str or Path
            Output file path.
        normalized : bool, optional
            If True, export normalized parameters.
        separator : str, optional
            Column separator character.
        use_xps_peak_names : bool, optional
            If True, apply pseudo-voigt XPS aliases.
        """
        components: list[ComponentDTO] = []
        for region_id in self._query.get_regions_ids(spectrum_id):
            for peak_id in self._query.get_peaks_ids(region_id):
                components.append(self._query.get_component_dto(peak_id, normalized=normalized))
        self._csv_export.export_spectrum_peak_parameters(
            path,
            tuple(components),
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
        Export a full spectrum DTO representation to a CSV-like file.

        Parameters
        ----------
        spectrum_id : str
            Identifier of the spectrum to export.
        path : str or Path
            Output file path.
        normalized : bool, optional
            If True, export normalized spectrum data.
        separator : str, optional
            Column separator character.
        include_evaluated_components : bool, optional
            If True, include evaluated model columns.
        include_background : bool, optional
            If True, include background model column.
        include_difference : bool, optional
            If True, include residual/difference column.
        """
        spectrum_repr = self._query.get_spectrum_dto_repr(spectrum_id, normalized=normalized)
        self._csv_export.export_spectrum(
            path,
            spectrum_repr,
            separator=separator,
            include_evaluated_components=include_evaluated_components,
            include_background=include_background,
            include_difference=include_difference,
            precision=precision,
        )

    def set_default_save_path(self, path: str | Path | None) -> None:
        """Set the default save path (stored in AppParameters)."""
        self._params.default_serialization_path = path

    def get_default_save_path(self) -> Path | None:
        """Return the default save path from AppParameters."""
        p = self._params.default_serialization_path
        return Path(p) if p is not None else None


apply_safe_execution_to_class(AppOrchestrator)
