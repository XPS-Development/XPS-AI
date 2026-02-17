"""
Command objects: executable operations with undo support.

Commands are created from Change instances via CommandRegistry.
They encapsulate "how" to apply and undo changes against the core data model.
"""

from dataclasses import asdict
from abc import ABC, abstractmethod

from core.services import SpectrumService, RegionService, ComponentService
from core.metadata import Metadata
from core.objects import Peak, Background, CoreObject
from .utils import ApplicationContext
from .changes import (
    BaseChange,
    UpdateParameter,
    UpdateRegionSlice,
    UpdateMultipleParameterValues,
    RemoveObject,
    RemoveMetadata,
    FullRemoveObject,
    CreateSpectrum,
    CreateRegion,
    CreatePeak,
    CreateBackground,
    ReplacePeakModel,
    ReplaceBackgroundModel,
    SetMetadata,
)

from typing import Callable, Any


class Command(ABC):
    """Base class for commands that apply and undo changes against the core model."""

    @classmethod
    @abstractmethod
    def from_change(cls, change: BaseChange, ctx: ApplicationContext) -> "Command": ...
    @abstractmethod
    def apply(self, ctx: ApplicationContext) -> None: ...
    @abstractmethod
    def undo(self, ctx: ApplicationContext) -> None: ...


class UpdateParameterCommand(Command):
    """
    Command that updates a single parameter attribute; supports undo via stored old value.
    """

    def __init__(
        self,
        component_id: str,
        name: str,
        parameter_field: str,
        new_value: str | bool | float,
        old_value: str | bool | float | None = None,
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
        """
        self.component_id = component_id
        self.name = name
        self.parameter_field = parameter_field
        self.new_value = new_value
        self._old_value = old_value

    @classmethod
    def from_change(
        cls,
        change: UpdateParameter,
        ctx: ApplicationContext,
    ) -> "UpdateParameterCommand":
        """
        Create an UpdateParameterCommand from a change, initializing undo state.

        Parameters
        ----------
        change : UpdateParameter
            The change to convert to a command.
        ctx : ApplicationContext
            Application context for reading current state.

        Returns
        -------
        UpdateParameterCommand
            Command instance with old value initialized for undo.
        """
        param = ctx.component.get_parameter(change.component_id, change.name)
        old_value = getattr(param, change.parameter_field)
        return cls(
            component_id=change.component_id,
            name=change.name,
            parameter_field=change.parameter_field,
            new_value=change.new_value,
            old_value=old_value,
        )

    def apply(self, ctx: ApplicationContext) -> None:
        ctx.component.set_parameter(self.component_id, self.name, **{self.parameter_field: self.new_value})

    def undo(self, ctx: ApplicationContext) -> None:
        if self._old_value is None:
            raise RuntimeError("Command was not applied")
        ctx.component.set_parameter(self.component_id, self.name, **{self.parameter_field: self._old_value})


class UpdateRegionSliceCommand(Command):
    """Command that updates a region's index slice; stores previous slice for undo."""

    def __init__(self, region_id: str, new_start: int, new_stop: int, old_start: int, old_stop: int) -> None:
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
    def from_change(cls, change: UpdateRegionSlice, ctx: ApplicationContext) -> "UpdateRegionSliceCommand":
        """
        Create an UpdateRegionSliceCommand from a change, initializing undo state.

        Parameters
        ----------
        change : UpdateRegionSlice
            The change to convert to a command.
        ctx : ApplicationContext
            Application context for reading current state.

        Returns
        -------
        UpdateRegionSliceCommand
            Command instance with old slice values initialized for undo.
        """
        old_slice = ctx.region.get_slice(change.region_id)
        return cls(
            region_id=change.region_id,
            new_start=change.start,
            new_stop=change.stop,
            old_start=old_slice.start,
            old_stop=old_slice.stop,
        )

    def apply(self, ctx: ApplicationContext) -> None:
        ctx.region.update_slice(self.region_id, self.new_start, self.new_stop)

    def undo(self, ctx: ApplicationContext) -> None:
        ctx.region.update_slice(self.region_id, self.old_start, self.old_stop)


class UpdateMultipleParameterValuesCommand(Command):
    """Command that updates multiple parameter values; stores old values for undo."""

    def __init__(
        self,
        component_id: str,
        parameters: dict[str, float],
        old_values: dict[str, float] | None = None,
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
        """
        self.component_id = component_id
        self.parameters = parameters
        self._old_values = old_values

    @classmethod
    def from_change(
        cls, change: UpdateMultipleParameterValues, ctx: ApplicationContext
    ) -> "UpdateMultipleParameterValuesCommand":
        """
        Create an UpdateMultipleParameterValuesCommand from a change, storing old values.

        Parameters
        ----------
        change : UpdateMultipleParameterValues
            The change to convert to a command.
        ctx : ApplicationContext
            Application context for reading current state.

        Returns
        -------
        UpdateMultipleParameterValuesCommand
            Command instance with old values stored for undo.
        """
        old_values = {}
        all_params = ctx.component.get_parameters(change.component_id)
        for param_name in change.parameters:
            if param_name in all_params:
                old_values[param_name] = all_params[param_name].value
        return cls(component_id=change.component_id, parameters=change.parameters, old_values=old_values)

    def apply(self, ctx: ApplicationContext) -> None:
        ctx.component.set_values(self.component_id, self.parameters)

    def undo(self, ctx: ApplicationContext) -> None:
        if self._old_values is None:
            raise RuntimeError("Command was not applied")
        ctx.component.set_values(self.component_id, self._old_values)


class SetMetadataCommand(Command):
    """
    Base command for setting metadata; stores previous metadata for undo.

    Subclasses implement from_change to extract obj_id and metadata from
    the specific change type. Uses unified MetadataService.get_metadata/set_metadata.
    """

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
    def from_change(cls, change: SetMetadata, ctx: ApplicationContext) -> "SetMetadataCommand":
        """
        Create a SetMetadataCommand from a change.

        Parameters
        ----------
        change : SetMetadata
            The change to convert to a command.
        ctx : ApplicationContext
            Application context for reading current state.

        Returns
        -------
        SetMetadataCommand
            Command instance.
        """
        # NOTE: object may not exist in collection yet
        old_metadata = ctx.metadata.get_metadata(change.obj_id)
        return cls(
            obj_id=change.obj_id,
            metadata=change.metadata,
            old_metadata=old_metadata,
        )

    def apply(self, ctx: ApplicationContext) -> None:
        ctx.metadata.set_metadata(self.obj_id, self.metadata)

    def undo(self, ctx: ApplicationContext) -> None:
        if self._old_metadata is None:
            ctx.metadata.remove_metadata(self.obj_id)
        else:
            ctx.metadata.set_metadata(self.obj_id, self._old_metadata)


class RemoveMetadataCommand(Command):
    """
    Command that removes metadata for an object by ID.

    Stores old metadata for undo; no-op if object had no metadata.
    """

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
    def from_change(cls, change: RemoveMetadata, ctx: ApplicationContext) -> "RemoveMetadataCommand":
        """
        Create a RemoveMetadataCommand from a change, storing old metadata for undo.

        Parameters
        ----------
        change : RemoveMetadata
            The change to convert to a command.
        ctx : ApplicationContext
            Application context for reading current state.

        Returns
        -------
        RemoveMetadataCommand
            Command instance with old metadata stored for undo.
        """
        if not ctx.collection.check_object_exists(change.obj_id):
            raise ValueError(f"Object with ID {change.obj_id} does not exist in collection")
        old_metadata = ctx.metadata.get_metadata(change.obj_id)
        return cls(obj_id=change.obj_id, old_metadata=old_metadata)

    def apply(self, ctx: ApplicationContext) -> None:
        ctx.metadata.remove_metadata(self.obj_id)

    def undo(self, ctx: ApplicationContext) -> None:
        if self._old_metadata is not None:
            ctx.metadata.set_metadata(self.obj_id, self._old_metadata)


class RemoveObjectCommand(Command):
    """
    Base command for removing objects from the collection.

    Uses detach which handles cascading removal of children automatically.
    """

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
    def from_change(cls, change: RemoveObject, ctx: ApplicationContext) -> "RemoveObjectCommand":
        """
        Create a RemoveObjectCommand from a change.
        """
        if not ctx.collection.check_object_exists(change.obj_id):
            raise ValueError(f"Object with ID {change.obj_id} does not exist in collection")
        return cls(obj_id=change.obj_id)

    def apply(self, ctx: ApplicationContext) -> None:
        """Remove the object and all its children from the collection."""
        self.objs = ctx.collection.detach(self.obj_id)

    def undo(self, ctx: ApplicationContext) -> None:
        """Restore the object and all its children to the collection."""
        if self.objs is None:
            raise RuntimeError("Command was not applied")
        for obj in self.objs:
            ctx.collection.attach(obj)


class CreateObjectCommand(Command):
    """
    Base command for adding objects to the collection.
    """

    create_obj_fn: Callable[..., CoreObject]

    def __init__(self, **params: Any) -> None:
        """
        Initialize an add object command.

        Parameters
        ----------
        **params : Any
            Parameters to pass to the create function.
        """
        self.obj = self.create_obj_fn(**params)

    def apply(self, ctx: ApplicationContext) -> None:
        """Add the object to the collection."""
        ctx.collection.attach(self.obj)

    def undo(self, ctx: ApplicationContext) -> None:
        """Remove the object from the collection."""
        if not ctx.collection.check_object_exists(self.obj.id_):
            raise RuntimeError("Command was not applied")
        ctx.collection.detach(self.obj)


class CreateSpectrumCommand(CreateObjectCommand):
    """Command that creates a spectrum."""

    create_obj_fn = staticmethod(SpectrumService._create_spectrum_obj)

    @classmethod
    def from_change(cls, change: CreateSpectrum, ctx: ApplicationContext) -> "CreateSpectrumCommand":
        """
        Create a CreateSpectrumCommand from a change.

        Parameters
        ----------
        change : CreateSpectrum
            The change to convert to a command.
        ctx : ApplicationContext
            Application context.

        Returns
        -------
        CreateSpectrumCommand
            Command instance.
        """
        return cls(**asdict(change))


class CreateRegionCommand(CreateObjectCommand):
    """Command that creates a region; stores ID for undo."""

    create_obj_fn = staticmethod(RegionService._create_region_obj)

    @classmethod
    def from_change(cls, change: CreateRegion, ctx: ApplicationContext) -> "CreateRegionCommand":
        """
        Create a CreateRegionCommand from a change.

        Parameters
        ----------
        change : CreateRegion
            The change to convert to a command.
        ctx : ApplicationContext
            Application context.

        Returns
        -------
        CreateRegionCommand
            Command instance.

        Raises
        ------
        ValueError
            If the region slice is out of bounds.
        """
        if not ctx.region._check_slice(change.spectrum_id, change.start, change.stop):
            raise ValueError("Invalid region slice")
        return cls(**asdict(change))


class CreatePeakCommand(CreateObjectCommand):
    """Command that creates a peak; stores ID for undo."""

    create_obj_fn = staticmethod(ComponentService._create_component_obj)

    @classmethod
    def from_change(cls, change: CreatePeak, ctx: ApplicationContext) -> "CreatePeakCommand":
        """
        Create a CreatePeakCommand from a change.

        Parameters
        ----------
        change : CreatePeak
            The change to convert to a command.
        ctx : ApplicationContext
            Application context.

        Returns
        -------
        CreatePeakCommand
            Command instance.
        """
        params = dict(asdict(change))
        params["component_id"] = params.pop("peak_id", None)
        return cls(**params, expected_type=Peak)


class CreateBackgroundCommand(CreateObjectCommand):
    """Command that creates or replaces a background; stores old background for undo."""

    create_obj_fn = staticmethod(ComponentService._create_component_obj)

    @classmethod
    def from_change(cls, change: CreateBackground, ctx: ApplicationContext) -> "CreateBackgroundCommand":
        """
        Create a CreateBackgroundCommand from a change.

        Parameters
        ----------
        change : CreateBackground
            The change to convert to a command.
        ctx : ApplicationContext
            Application context.

        Returns
        -------
        CreateBackgroundCommand
            Command instance.
        """
        params = dict(asdict(change))
        params["component_id"] = params.pop("background_id", None)
        return cls(**params, expected_type=Background)


class CompositeCommand(Command):
    """Command that executes multiple commands as a batch."""

    @classmethod
    def from_change(cls, change: BaseChange, ctx: ApplicationContext) -> "CompositeCommand":
        """Not used; CompositeCommand is built from CompositeChange by the registry."""
        raise NotImplementedError("CompositeCommand is built from CompositeChange by CommandRegistry")

    def __init__(self, *, commands: list[Command]) -> None:
        """
        Initialize a composite command.

        Parameters
        ----------
        commands : list[Command]
            List of commands to execute.
        """
        self.commands = commands

    def apply(self, ctx: ApplicationContext) -> None:
        """Apply all commands in order."""
        for cmd in self.commands:
            cmd.apply(ctx)

    def undo(self, ctx: ApplicationContext) -> None:
        """Undo all commands in reverse order."""
        for cmd in reversed(self.commands):
            cmd.undo(ctx)


class ReplacePeakModelCommand(CompositeCommand):
    """Command that replaces a peak's model; stores old peak for undo."""

    @staticmethod
    def _parse_change(change: ReplacePeakModel, ctx: ApplicationContext) -> tuple[RemoveObject, CreatePeak]:
        """
        Adapter for ReplacePeakModel change to RemoveObject and CreatePeak.
        """
        rm_ch = RemoveObject(change.peak_id)
        create_ch = CreatePeak(
            region_id=ctx.collection.get_parent(change.peak_id),
            model_name=change.new_model_name,
            parameters=change.parameters,
            peak_id=change.peak_id,
        )
        return rm_ch, create_ch

    @classmethod
    def from_change(cls, change: ReplacePeakModel, ctx: ApplicationContext) -> "CompositeCommand":
        """
        Create a ReplacePeakModelCommand from a change, storing old peak.

        Parameters
        ----------
        change : ReplacePeakModel
            The change to convert to a command.
        ctx : ApplicationContext
            Application context for reading current state.

        Returns
        -------
        ReplacePeakModelCommand
            Command instance with old peak stored for undo.
        """
        rm_ch, create_ch = cls._parse_change(change, ctx)
        rm_cmd = RemoveObjectCommand.from_change(rm_ch, ctx)
        create_cmd = CreatePeakCommand.from_change(create_ch, ctx)
        return cls(commands=[rm_cmd, create_cmd])


class ReplaceBackgroundModelCommand(CompositeCommand):
    """Command that replaces a background's model; stores old background for undo."""

    @staticmethod
    def _parse_change(
        change: ReplaceBackgroundModel, ctx: ApplicationContext
    ) -> tuple[RemoveObject | None, CreateBackground]:
        """
        Adapter for ReplaceBackgroundModel change to RemoveObject and CreateBackground.
        """
        create_ch = CreateBackground(
            region_id=change.region_id,
            model_name=change.new_model_name,
            parameters=change.parameters,
            background_id=change.background_id,
        )
        bg_id = ctx.collection.get_background(change.region_id)
        rm_ch = RemoveObject(bg_id) if bg_id else None
        return rm_ch, create_ch

    @classmethod
    def from_change(cls, change: ReplaceBackgroundModel, ctx: ApplicationContext) -> "CompositeCommand":
        """
        Create a ReplaceBackgroundModelCommand from a change.

        Parameters
        ----------
        change : ReplaceBackgroundModel
            The change to convert to a command.
        ctx : ApplicationContext
            Application context for reading current state.

        Returns
        -------
        ReplaceBackgroundModelCommand
            Command instance.
        """
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
    def from_change(cls, change: FullRemoveObject, ctx: ApplicationContext) -> "FullRemoveObjectCommand":
        """
        Create a FullRemoveObjectCommand from a change.

        Parameters
        ----------
        change : FullRemoveObject
            The change to convert to a command.
        ctx : ApplicationContext
            Application context for reading current state.

        Returns
        -------
        FullRemoveObjectCommand
            Command instance.
        """
        if not ctx.collection.check_object_exists(change.obj_id):
            raise ValueError(f"Object with ID {change.obj_id} does not exist in collection")
        subtree = ctx.collection.get_subtree(change.obj_id)
        metadata_commands: list[Command] = []
        for obj_id in subtree:
            rm_meta_cmd = RemoveMetadataCommand.from_change(RemoveMetadata(obj_id), ctx)
            metadata_commands.append(rm_meta_cmd)
        rm_obj_cmd = RemoveObjectCommand.from_change(RemoveObject(change.obj_id), ctx)
        return cls(commands=[*metadata_commands, rm_obj_cmd])
