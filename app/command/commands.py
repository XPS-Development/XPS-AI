"""
Command objects: executable operations with undo support.

Commands are created from Change instances via CommandRegistry.
They encapsulate "how" to apply and undo changes against the core data model.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict
from typing import ClassVar

from core.metadata import Metadata
from core.objects import Background, CoreObject, Peak, Spectrum
from core.services import ComponentService, CoreContext, RegionService, SpectrumService

from .changes import (
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
from .refresh import UiRefresh


class Command(ABC):
    """Base class for commands that apply and undo changes against the core model."""

    ui_refresh: ClassVar[UiRefresh] = UiRefresh.ALL

    def combined_ui_refresh(self) -> UiRefresh:
        """Return UI invalidation flags for this command (including composites)."""
        return self.ui_refresh

    @classmethod
    @abstractmethod
    def from_change(cls, change: BaseChange, ctx: CoreContext) -> "Command":
        """Build a command from a change DTO."""

    @abstractmethod
    def apply(self, ctx: CoreContext) -> None:
        """Apply the command to the core model."""

    @abstractmethod
    def undo(self, ctx: CoreContext) -> None:
        """Revert the command on the core model."""


class UpdateParameterCommand(Command):
    """Command that updates a single parameter attribute; supports undo via stored old value."""

    ui_refresh = UiRefresh.FIT

    def __init__(
        self,
        component_id: str,
        name: str,
        parameter_field: ParameterField,
        new_value: str | bool | float | None,
        old_value: str | bool | float | None = None,
        normalized: bool = False,
    ) -> None:
        """
        Initialize an update parameter command.

        Parameters
        ----------
        component_id : str
            ID of the component.
        name : str
            Parameter name.
        parameter_field : str
            Field of the parameter to update.
        new_value : str | bool | float
            New value for the parameter field.
        old_value : str | bool | float | None, optional
            Old value for undo (typically set by from_change).
        normalized : bool, default=False
            Whether the new value is normalized.
        """
        self.component_id = component_id
        self.name = name
        self.parameter_field = parameter_field
        self.new_value = new_value
        self._old_value = old_value
        self.normalized = normalized

    @classmethod
    def from_change(
        cls,
        change: BaseChange,
        ctx: CoreContext,
    ) -> "UpdateParameterCommand":
        """
        Create an UpdateParameterCommand from a change, initializing undo state.

        Parameters
        ----------
        change : UpdateParameter
            The change to convert to a command.
        ctx : CoreContext
            Application context for reading current state.

        Returns
        -------
        UpdateParameterCommand
            Command instance with old value initialized for undo.
        """
        if not isinstance(change, UpdateParameter):
            raise TypeError(f"Expected UpdateParameter change, got {type(change).__name__}")

        param = ctx.component.get_parameter(change.component_id, change.name, change.normalized)
        old_value = param[change.parameter_field]
        return cls(
            component_id=change.component_id,
            name=change.name,
            parameter_field=change.parameter_field,
            new_value=change.new_value,
            old_value=old_value,
            normalized=change.normalized,
        )

    def apply(self, ctx: CoreContext) -> None:
        """Set the parameter to the new value."""
        ctx.component.set_parameter(
            self.component_id,
            self.name,
            normalized=self.normalized,
            **{self.parameter_field: self.new_value},
        )

    def undo(self, ctx: CoreContext) -> None:
        """Restore the previous parameter value."""
        if self._old_value is None:
            raise RuntimeError("Command was not applied")
        ctx.component.set_parameter(
            self.component_id,
            self.name,
            normalized=self.normalized,
            **{self.parameter_field: self._old_value},
        )


class UpdateRegionSliceCommand(Command):
    """Command that updates a region's index slice; stores indices only (values converted in from_change)."""

    ui_refresh = UiRefresh.FIT

    def __init__(
        self,
        region_id: str,
        new_start: int,
        new_stop: int,
        old_start: int,
        old_stop: int,
    ) -> None:
        """
        Initialize an update region slice command.

        Parameters
        ----------
        region_id : str
            ID of the region.
        new_start : int
            New start index.
        new_stop : int
            New stop index.
        old_start : int
            Old start index for undo.
        old_stop : int
            Old stop index for undo.
        """
        self.region_id = region_id
        self.new_start = new_start
        self.new_stop = new_stop
        self.old_start = old_start
        self.old_stop = old_stop

    @classmethod
    def from_change(cls, change: BaseChange, ctx: CoreContext) -> "UpdateRegionSliceCommand":
        """
        Create an UpdateRegionSliceCommand from a change, initializing undo state.

        Converts value-mode start/stop to indices so the command works only with indices.
        """
        if not isinstance(change, UpdateRegionSlice):
            raise TypeError(f"Expected UpdateRegionSlice change, got {type(change).__name__}")

        old_start, old_stop = ctx.region.get_slice(change.region_id, mode="index")
        spectrum_id = ctx.query.get_parent(change.region_id)

        if change.mode == "value":
            new_start = (
                old_start
                if change.start is None
                else ctx.region._convert_value_to_index(spectrum_id, change.start)
            )
            stop_idx = (
                old_stop
                if change.stop is None
                else ctx.region._convert_value_to_index(spectrum_id, change.stop)
            )
            assert stop_idx is not None
            new_stop = stop_idx + 1
            assert new_start is not None
        else:
            new_start = old_start if change.start is None else int(change.start)
            new_stop = old_stop if change.stop is None else int(change.stop)

        if not ctx.region._check_slice(spectrum_id, new_start, new_stop):
            new_start, new_stop = ctx.region._get_bound_indices(spectrum_id)

        return cls(
            region_id=change.region_id,
            new_start=new_start,
            new_stop=new_stop,
            old_start=old_start,
            old_stop=old_stop,
        )

    def apply(self, ctx: CoreContext) -> None:
        """Set the region slice to the new bounds."""
        ctx.region.update_slice(self.region_id, self.new_start, self.new_stop, mode="index")

    def undo(self, ctx: CoreContext) -> None:
        """Restore the previous region slice bounds."""
        ctx.region.update_slice(self.region_id, self.old_start, self.old_stop, mode="index")


class UpdateMultipleParameterValuesCommand(Command):
    """Command that updates multiple parameter values; stores old values for undo."""

    ui_refresh = UiRefresh.FIT

    def __init__(
        self,
        component_id: str,
        parameters: dict[str, float],
        old_values: dict[str, float] | None = None,
        normalized: bool = False,
    ) -> None:
        """
        Initialize an update multiple parameter values command.

        Parameters
        ----------
        component_id : str
            ID of the component.
        parameters : dict[str, float]
            New parameter values.
        old_values : dict[str, float] | None, optional
            Old parameter values for undo (typically set by from_change).
        normalized : bool, default=False
            Whether the new values are normalized.
        """
        self.component_id = component_id
        self.parameters = parameters
        self._old_values = old_values
        self.normalized = normalized

    @classmethod
    def from_change(
        cls, change: BaseChange, ctx: CoreContext
    ) -> "UpdateMultipleParameterValuesCommand":
        """
        Create an UpdateMultipleParameterValuesCommand from a change, storing old values.

        Parameters
        ----------
        change : UpdateMultipleParameterValues
            The change to convert to a command.
        ctx : CoreContext
            Application context for reading current state.

        Returns
        -------
        UpdateMultipleParameterValuesCommand
            Command instance with old values stored for undo.
        """
        if not isinstance(change, UpdateMultipleParameterValues):
            raise TypeError(
                f"Expected UpdateMultipleParameterValues change, got {type(change).__name__}"
            )

        all_params = ctx.component.get_parameters(change.component_id, normalized=change.normalized)
        old_values: dict[str, float] = {}
        for param_name in change.parameters.keys():
            raw_val = all_params[param_name]["value"]
            if not isinstance(raw_val, (int, float)):
                raise TypeError(f"Expected numeric parameter value, got {type(raw_val).__name__}")
            old_values[param_name] = float(raw_val)

        return cls(
            component_id=change.component_id,
            parameters=change.parameters,
            old_values=old_values,
            normalized=change.normalized,
        )

    def apply(self, ctx: CoreContext) -> None:
        """Write the new parameter values onto the component."""
        ctx.component.set_values(self.component_id, self.parameters, normalized=self.normalized)

    def undo(self, ctx: CoreContext) -> None:
        """Restore the previous parameter values."""
        if self._old_values is None:
            raise RuntimeError("Command was not applied")
        ctx.component.set_values(self.component_id, self._old_values, normalized=self.normalized)


class SetMetadataCommand(Command):
    """
    Base command for setting metadata; stores previous metadata for undo.

    Subclasses implement from_change to extract obj_id and metadata from
    the specific change type. Uses unified MetadataService.get_metadata/set_metadata.
    """

    ui_refresh = UiRefresh.METADATA

    def __init__(
        self,
        obj_id: str,
        metadata: Metadata,
        old_metadata: Metadata | None,
    ) -> None:
        self.obj_id = obj_id
        self.metadata = metadata
        self._old_metadata = old_metadata

    @classmethod
    def from_change(cls, change: BaseChange, ctx: CoreContext) -> "SetMetadataCommand":
        """
        Create a SetMetadataCommand from a change.

        Parameters
        ----------
        change : SetMetadata
            The change to convert to a command.
        ctx : CoreContext
            Application context for reading current state.

        Returns
        -------
        SetMetadataCommand
            Command instance.
        """
        if not isinstance(change, SetMetadata):
            raise TypeError(f"Expected SetMetadata change, got {type(change).__name__}")

        # NOTE: object may not exist in collection yet
        old_metadata = ctx.metadata.get_metadata(change.obj_id)
        return cls(
            obj_id=change.obj_id,
            metadata=change.metadata,
            old_metadata=old_metadata,
        )

    def apply(self, ctx: CoreContext) -> None:
        """Store the new metadata for the object."""
        ctx.metadata.set_metadata(self.obj_id, self.metadata)

    def undo(self, ctx: CoreContext) -> None:
        """Restore previous metadata, or remove it if none existed."""
        if self._old_metadata is None:
            ctx.metadata.remove_metadata(self.obj_id)
        else:
            ctx.metadata.set_metadata(self.obj_id, self._old_metadata)


class RemoveMetadataCommand(Command):
    """
    Command that removes metadata for an object by ID.

    Stores old metadata for undo; no-op if object had no metadata.
    """

    ui_refresh = UiRefresh.HIERARCHY | UiRefresh.DOCUMENT

    def __init__(
        self,
        obj_id: str,
        old_metadata: Metadata | None = None,
    ) -> None:
        """
        Initialize a remove metadata command.

        Parameters
        ----------
        obj_id : str
            ID of the object whose metadata to remove.
        old_metadata : Metadata | None, optional
            Stored metadata for undo, or None if none existed.
        """
        self.obj_id = obj_id
        self._old_metadata = old_metadata
        self._applied = False

    @classmethod
    def from_change(cls, change: BaseChange, ctx: CoreContext) -> "RemoveMetadataCommand":
        """
        Create a RemoveMetadataCommand from a change, storing old metadata for undo.

        Parameters
        ----------
        change : RemoveMetadata
            The change to convert to a command.
        ctx : CoreContext
            Application context for reading current state.

        Returns
        -------
        RemoveMetadataCommand
            Command instance with old metadata stored for undo.
        """
        if not isinstance(change, RemoveMetadata):
            raise TypeError(f"Expected RemoveMetadata change, got {type(change).__name__}")

        if not ctx.query.check_object_exists(change.obj_id):
            raise ValueError(f"Object with ID {change.obj_id} does not exist in collection")
        old_metadata = ctx.metadata.get_metadata(change.obj_id)
        return cls(obj_id=change.obj_id, old_metadata=old_metadata)

    def apply(self, ctx: CoreContext) -> None:
        """Remove metadata for the object."""
        ctx.metadata.remove_metadata(self.obj_id)

    def undo(self, ctx: CoreContext) -> None:
        """Restore the previous metadata if it existed."""
        if self._old_metadata is not None:
            ctx.metadata.set_metadata(self.obj_id, self._old_metadata)


class RemoveObjectCommand(Command):
    """
    Base command for removing objects from the collection.

    Uses detach which handles cascading removal of children automatically.
    """

    def combined_ui_refresh(self) -> UiRefresh:
        """Spectrum removals also invalidate the hierarchy tree."""
        if self.objs:
            root = self.objs[0]
            if isinstance(root, Spectrum):
                return UiRefresh.ALL
        return UiRefresh.FIT

    def __init__(self, obj_id: str) -> None:
        """
        Initialize a remove object command.

        Parameters
        ----------
        obj_id : str
            ID of the object to remove.
        """
        self.obj_id: str = obj_id
        self.objs: list[CoreObject] | None = None

    @classmethod
    def from_change(cls, change: BaseChange, ctx: CoreContext) -> "RemoveObjectCommand":
        """Create a RemoveObjectCommand from a change."""
        if not isinstance(change, RemoveObject):
            raise TypeError(f"Expected RemoveObject change, got {type(change).__name__}")

        if not ctx.query.check_object_exists(change.obj_id):
            raise ValueError(f"Object with ID {change.obj_id} does not exist in collection")
        return cls(obj_id=change.obj_id)

    def apply(self, ctx: CoreContext) -> None:
        """Remove the object and all its children from the collection."""
        self.objs = ctx.query.detach(self.obj_id)

    def undo(self, ctx: CoreContext) -> None:
        """Restore the object and all its children to the collection."""
        if self.objs is None:
            raise RuntimeError("Command was not applied")
        for obj in self.objs:
            ctx.query.attach(obj)


class CreateObjectCommand(Command):
    """Base command for adding objects to the collection."""

    create_obj_fn: Callable[..., CoreObject]

    def __init__(self, **params) -> None:
        """
        Initialize an add object command.

        Parameters
        ----------
        **params : Any
            Parameters to pass to the create function.
        """
        self.obj = self.create_obj_fn(**params)

    def apply(self, ctx: CoreContext) -> None:
        """Add the object to the collection."""
        ctx.query.attach(self.obj)

    def undo(self, ctx: CoreContext) -> None:
        """Remove the object from the collection."""
        if not ctx.query.check_object_exists(self.obj.id_):
            raise RuntimeError("Command was not applied")
        ctx.query.detach(self.obj)


class CreateSpectrumCommand(CreateObjectCommand):
    """Command that creates a spectrum."""

    ui_refresh = UiRefresh.ALL

    create_obj_fn = staticmethod(SpectrumService._create_spectrum_obj)

    @classmethod
    def from_change(cls, change: BaseChange, ctx: CoreContext) -> "CreateSpectrumCommand":
        """
        Create a CreateSpectrumCommand from a change.

        Parameters
        ----------
        change : CreateSpectrum
            The change to convert to a command.
        ctx : CoreContext
            Application context.

        Returns
        -------
        CreateSpectrumCommand
            Command instance.
        """
        if not isinstance(change, CreateSpectrum):
            raise TypeError(f"Expected CreateSpectrum change, got {type(change).__name__}")
        return cls(**asdict(change))


class CreateRegionCommand(CreateObjectCommand):
    """Command that creates a region; stores ID for undo."""

    ui_refresh = UiRefresh.FIT

    create_obj_fn = staticmethod(RegionService._create_region_obj)

    @classmethod
    def from_change(cls, change: BaseChange, ctx: CoreContext) -> "CreateRegionCommand":
        """
        Create a CreateRegionCommand from a change.

        Parameters
        ----------
        change : CreateRegion
            The change to convert to a command.
        ctx : CoreContext
            Application context.

        Returns
        -------
        CreateRegionCommand
            Command instance.
        """
        if not isinstance(change, CreateRegion):
            raise TypeError(f"Expected CreateRegion change, got {type(change).__name__}")

        bounds_start, bounds_stop = ctx.region._get_bound_indices(change.spectrum_id)
        if change.start is None or change.stop is None:
            start = bounds_start
            stop = bounds_stop
        elif change.mode == "value":
            start_idx = ctx.region._convert_value_to_index(change.spectrum_id, change.start)
            assert start_idx is not None
            start = start_idx
            stop_idx = ctx.region._convert_value_to_index(change.spectrum_id, change.stop)
            assert stop_idx is not None
            stop = stop_idx + 1
        else:
            start = int(change.start)
            stop = int(change.stop)

        if not ctx.region._check_slice(change.spectrum_id, start, stop):
            start, stop = bounds_start, bounds_stop

        # Command works with indices only; pass converted start/stop, drop mode.
        d = asdict(change)
        d["start"] = start
        d["stop"] = stop
        d.pop("mode", None)

        return cls(**d)


class CreatePeakCommand(CreateObjectCommand):
    """Command that creates a peak; stores ID for undo."""

    ui_refresh = UiRefresh.FIT

    create_obj_fn = staticmethod(ComponentService._create_component_obj)

    @classmethod
    def from_change(cls, change: BaseChange, ctx: CoreContext) -> "CreatePeakCommand":
        """
        Create a CreatePeakCommand from a change.

        Parameters
        ----------
        change : CreatePeak
            The change to convert to a command.
        ctx : CoreContext
            Application context.

        Returns
        -------
        CreatePeakCommand
            Command instance.
        """
        if not isinstance(change, CreatePeak):
            raise TypeError(f"Expected CreatePeak change, got {type(change).__name__}")
        params = dict(asdict(change))
        params["component_id"] = params.pop("peak_id", None)
        return cls(**params, expected_type=Peak)


class CreateBackgroundCommand(CreateObjectCommand):
    """Command that creates or replaces a background; stores old background for undo."""

    ui_refresh = UiRefresh.FIT

    create_obj_fn = staticmethod(ComponentService._create_component_obj)

    @classmethod
    def from_change(cls, change: BaseChange, ctx: CoreContext) -> "CreateBackgroundCommand":
        """
        Create a CreateBackgroundCommand from a change.

        Parameters
        ----------
        change : CreateBackground
            The change to convert to a command.
        ctx : CoreContext
            Application context.

        Returns
        -------
        CreateBackgroundCommand
            Command instance.
        """
        if not isinstance(change, CreateBackground):
            raise TypeError(f"Expected CreateBackground change, got {type(change).__name__}")
        params = dict(asdict(change))
        params["component_id"] = params.pop("background_id", None)
        return cls(**params, expected_type=Background)


class CompositeCommand(Command):
    """Command that executes multiple commands as a batch."""

    @classmethod
    def from_change(cls, change: BaseChange, ctx: CoreContext) -> "CompositeCommand":
        """Not used; CompositeCommand is built from CompositeChange by the registry."""
        raise NotImplementedError(
            "CompositeCommand is built from CompositeChange by CommandRegistry"
        )

    def __init__(self, *, commands: list[Command]) -> None:
        """
        Initialize a composite command.

        Parameters
        ----------
        commands : list[Command]
            List of commands to execute.
        """
        self.commands = commands

    def apply(self, ctx: CoreContext) -> None:
        """Apply all commands in order."""
        for cmd in self.commands:
            cmd.apply(ctx)

    def undo(self, ctx: CoreContext) -> None:
        """Undo all commands in reverse order."""
        for cmd in reversed(self.commands):
            cmd.undo(ctx)

    def combined_ui_refresh(self) -> UiRefresh:
        """Aggregate refresh flags from child commands."""
        flags = UiRefresh(0)
        for cmd in self.commands:
            flags |= cmd.combined_ui_refresh()
        return flags


class ReplacePeakModelCommand(CompositeCommand):
    """Command that replaces a peak's model; stores old peak for undo."""

    @staticmethod
    def _parse_change(
        change: ReplacePeakModel, ctx: CoreContext
    ) -> tuple[RemoveObject, CreatePeak]:
        """Adapter for ReplacePeakModel change to RemoveObject and CreatePeak."""
        rm_ch = RemoveObject(change.peak_id)
        create_ch = CreatePeak(
            region_id=ctx.query.get_parent(change.peak_id),
            model_name=change.new_model_name,
            parameters=change.parameters,
            peak_id=change.peak_id,
        )
        return rm_ch, create_ch

    @classmethod
    def from_change(cls, change: BaseChange, ctx: CoreContext) -> "CompositeCommand":
        """
        Create a ReplacePeakModelCommand from a change, storing old peak.

        Parameters
        ----------
        change : ReplacePeakModel
            The change to convert to a command.
        ctx : CoreContext
            Application context for reading current state.

        Returns
        -------
        ReplacePeakModelCommand
            Command instance with old peak stored for undo.
        """
        if not isinstance(change, ReplacePeakModel):
            raise TypeError(f"Expected ReplacePeakModel change, got {type(change).__name__}")

        rm_ch, create_ch = cls._parse_change(change, ctx)
        rm_cmd = RemoveObjectCommand.from_change(rm_ch, ctx)
        create_cmd = CreatePeakCommand.from_change(create_ch, ctx)
        return cls(commands=[rm_cmd, create_cmd])


class ReplaceBackgroundModelCommand(CompositeCommand):
    """Command that replaces a background's model; stores old background for undo."""

    @staticmethod
    def _parse_change(
        change: ReplaceBackgroundModel, ctx: CoreContext
    ) -> tuple[RemoveObject | None, CreateBackground]:
        """Adapter for ReplaceBackgroundModel change to RemoveObject and CreateBackground."""
        create_ch = CreateBackground(
            region_id=change.region_id,
            model_name=change.new_model_name,
            parameters=change.parameters,
            background_id=change.background_id,
        )
        bg_id = ctx.query.get_background(change.region_id)
        rm_ch = RemoveObject(bg_id) if bg_id else None
        return rm_ch, create_ch

    @classmethod
    def from_change(cls, change: BaseChange, ctx: CoreContext) -> "CompositeCommand":
        """
        Create a ReplaceBackgroundModelCommand from a change.

        Parameters
        ----------
        change : ReplaceBackgroundModel
            The change to convert to a command.
        ctx : CoreContext
            Application context for reading current state.

        Returns
        -------
        ReplaceBackgroundModelCommand
            Command instance.
        """
        if not isinstance(change, ReplaceBackgroundModel):
            raise TypeError(f"Expected ReplaceBackgroundModel change, got {type(change).__name__}")

        rm_ch, create_ch = cls._parse_change(change, ctx)
        commands: list[Command] = []
        if rm_ch is not None:
            commands.append(RemoveObjectCommand.from_change(rm_ch, ctx))
        commands.append(CreateBackgroundCommand.from_change(create_ch, ctx))
        return cls(commands=commands)


class FullRemoveObjectCommand(CompositeCommand):
    """
    Command that removes an object and its metadata from both collection and metadata store.

    Translates FullRemoveObject to RemoveMetadataCommand(s) and RemoveObjectCommand.
    Cascades: removes metadata for all descendants with metadata, then removes the subtree.
    """

    @classmethod
    def from_change(cls, change: BaseChange, ctx: CoreContext) -> "FullRemoveObjectCommand":
        """
        Create a FullRemoveObjectCommand from a change.

        Parameters
        ----------
        change : FullRemoveObject
            The change to convert to a command.
        ctx : CoreContext
            Application context for reading current state.

        Returns
        -------
        FullRemoveObjectCommand
            Command instance.
        """
        if not isinstance(change, FullRemoveObject):
            raise TypeError(f"Expected FullRemoveObject change, got {type(change).__name__}")

        if not ctx.query.check_object_exists(change.obj_id):
            raise ValueError(f"Object with ID {change.obj_id} does not exist in collection")
        subtree = ctx.query.get_subtree(change.obj_id)
        metadata_commands: list[Command] = []
        for obj_id in subtree:
            rm_meta_cmd = RemoveMetadataCommand.from_change(RemoveMetadata(obj_id), ctx)
            metadata_commands.append(rm_meta_cmd)
        rm_obj_cmd = RemoveObjectCommand.from_change(RemoveObject(change.obj_id), ctx)
        return cls(commands=[*metadata_commands, rm_obj_cmd])
