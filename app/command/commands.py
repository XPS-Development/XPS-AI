from dataclasses import dataclass, field

from core.math_models import BaseBackgroundModel
from .utils import ApplicationContext
from .changes import SetParameterValue, ParameterChange, UpdateRegionSlice, RemoveComponent

from typing import Literal, Protocol


class Command(Protocol):
    """Protocol for command-like objects with apply and undo."""

    def apply(self, ctx: ApplicationContext) -> None: ...
    def undo(self, ctx: ApplicationContext) -> None: ...


@dataclass(frozen=True)
class ComponentSnapshot:
    """
    Snapshot of a component sufficient to re-create it for undo.

    Used by RemoveComponentCommand to restore a removed component.
    """

    component_id: str
    region_id: str
    model_name: str
    kind: Literal["peak", "background"]
    parameters: dict[str, float]


@dataclass(frozen=True)
class ParameterChangeCommand:
    """
    Command that updates a single parameter attribute; supports undo via stored old value.
    """

    component_id: str
    name: str
    parameter_field: str
    new_value: str | bool | float
    _old_value: str | bool | float | None = field(default=None, repr=False)

    @classmethod
    def from_change(
        cls,
        change: SetParameterValue | ParameterChange,
        ctx: ApplicationContext,
    ) -> "ParameterChangeCommand":
        """
        Create a ParameterChangeCommand from a change, initializing undo state.

        Parameters
        ----------
        change : SetParameterValue | ParameterChange
            The change to convert to a command.
        ctx : ApplicationContext
            Application context for reading current state.

        Returns
        -------
        ParameterChangeCommand
            Command instance with old value initialized for undo.
        """
        param = ctx.component.get_parameter(change.component_id, change.name)
        if isinstance(change, SetParameterValue):
            old_value: float = param.value
            return cls(
                component_id=change.component_id,
                name=change.name,
                parameter_field="value",
                new_value=change.new_value,
                _old_value=old_value,
            )
        else:  # ParameterChange
            old_value = getattr(param, change.parameter_field)
            return cls(
                component_id=change.component_id,
                name=change.name,
                parameter_field=change.parameter_field,
                new_value=change.new_value,
                _old_value=old_value,
            )

    def apply(self, ctx: ApplicationContext) -> None:
        ctx.component.set_parameter(self.component_id, self.name, **{self.parameter_field: self.new_value})

    def undo(self, ctx: ApplicationContext) -> None:
        if self._old_value is None:
            raise RuntimeError("Command was not applied")
        ctx.component.set_parameter(self.component_id, self.name, **{self.parameter_field: self._old_value})


@dataclass(frozen=True)
class UpdateRegionSliceCommand:
    """Command that updates a region's index slice; stores previous slice for undo."""

    region_id: str
    new_start: int
    new_stop: int
    old_start: int
    old_stop: int

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
        region = ctx.collection._get(change.region_id)
        old_slice = region.slice_
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


@dataclass(frozen=True)
class RemoveComponentCommand:
    """Command that removes a component; stores snapshot for undo (re-add)."""

    component_id: str
    _snapshot: ComponentSnapshot | None = field(default=None, repr=False)

    @classmethod
    def from_change(cls, change: RemoveComponent, ctx: ApplicationContext) -> "RemoveComponentCommand":
        """
        Create a RemoveComponentCommand from a change, initializing undo state.

        Parameters
        ----------
        change : RemoveComponent
            The change to convert to a command.
        ctx : ApplicationContext
            Application context for reading current state.

        Returns
        -------
        RemoveComponentCommand
            Command instance with component snapshot initialized for undo.
        """
        region_id = ctx.collection.get_parent(change.component_id)
        model = ctx.component.get_model(change.component_id)
        kind: Literal["peak", "background"] = (
            "background" if isinstance(model, BaseBackgroundModel) else "peak"
        )
        params = {n: p.value for n, p in ctx.component.get_parameters(change.component_id).items()}
        snapshot = ComponentSnapshot(
            component_id=change.component_id,
            region_id=region_id,
            model_name=model.name,
            kind=kind,
            parameters=params,
        )
        return cls(component_id=change.component_id, _snapshot=snapshot)

    def apply(self, ctx: ApplicationContext) -> None:
        ctx.component.remove_component(self.component_id)

    def undo(self, ctx: ApplicationContext) -> None:
        if self._snapshot is None:
            raise RuntimeError("Command was not applied")
        s = self._snapshot
        if s.kind == "peak":
            ctx.component.create_peak(s.region_id, s.model_name, s.parameters, s.component_id)
        else:
            ctx.component.replace_background(s.region_id, s.model_name, s.parameters, s.component_id)


@dataclass(frozen=True)
class CommandBatch:
    """
    Aggregates multiple commands for batch execution.

    Useful for grouping related commands (e.g., multiple parameter changes
    after optimization) so they can be undone as a single unit.
    """

    commands: list[Command] = field(default_factory=list)

    def apply(self, ctx: ApplicationContext) -> None:
        """Apply all commands in order."""
        for cmd in self.commands:
            cmd.apply(ctx)

    def undo(self, ctx: ApplicationContext) -> None:
        """Undo all commands in reverse order."""
        for cmd in reversed(self.commands):
            cmd.undo(ctx)

    def add(self, cmd: Command) -> None:
        """
        Add a command to the batch.

        Parameters
        ----------
        cmd : Command
            Command to add to the batch.
        """
        self.commands.append(cmd)
