"""
Core command infrastructure.

Command implementations and CommandRegistry live here to avoid circular imports
with .changes.
"""

from .changes import (
    BaseChange,
    UpdateParameter,
    UpdateRegionSlice,
    RemoveObject,
    CreateSpectrum,
    CreateRegion,
    CreatePeak,
    CreateBackground,
    ReplacePeakModel,
    ReplaceBackgroundModel,
    UpdateMultipleParameterValues,
    SetSpectrumMetadata,
    SetRegionMetadata,
    SetPeakMetadata,
    CompositeChange,
)
from .commands import (
    Command,
    UpdateParameterCommand,
    UpdateRegionSliceCommand,
    RemoveObjectCommand,
    CreateSpectrumCommand,
    CreateRegionCommand,
    CreatePeakCommand,
    CreateBackgroundCommand,
    ReplacePeakModelCommand,
    ReplaceBackgroundModelCommand,
    UpdateMultipleParameterValuesCommand,
    SetSpectrumMetadataCommand,
    SetRegionMetadataCommand,
    SetPeakMetadataCommand,
    CompositeCommand,
)
from .utils import ApplicationContext


class UndoRedoStack:
    """
    Undo/redo stack for command history, similar to typical UI applications.

    Maintains separate undo and redo stacks. Executing a new command clears
    the redo stack (branching is discarded).
    """

    def __init__(self) -> None:
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []

    @property
    def can_undo(self) -> bool:
        """True if there is at least one command to undo."""
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        """True if there is at least one command to redo."""
        return len(self._redo_stack) > 0

    def push(self, cmd: Command) -> None:
        """
        Push a newly executed command onto the undo stack and clear the redo stack.

        Parameters
        ----------
        cmd : Command
            The command that was just applied.
        """
        self._undo_stack.append(cmd)
        self._redo_stack.clear()

    def pop_undo(self) -> Command:
        """
        Pop the last command from the undo stack.

        Returns
        -------
        Command
            The command to undo.

        Raises
        ------
        IndexError
            If the undo stack is empty.
        """
        return self._undo_stack.pop()

    def push_redo(self, cmd: Command) -> None:
        """
        Push an undone command onto the redo stack.

        Parameters
        ----------
        cmd : Command
            The command that was just undone.
        """
        self._redo_stack.append(cmd)

    def pop_redo(self) -> Command:
        """
        Pop the last command from the redo stack.

        Returns
        -------
        Command
            The command to redo.

        Raises
        ------
        IndexError
            If the redo stack is empty.
        """
        return self._redo_stack.pop()

    def push_undo(self, cmd: Command) -> None:
        """
        Push a redone command back onto the undo stack.

        Parameters
        ----------
        cmd : Command
            The command that was just redone.
        """
        self._undo_stack.append(cmd)


class CommandRegistry:
    """
    Registry mapping Change types to Command types for extensible command creation.
    """

    def __init__(self) -> None:
        self._registry: dict[type[BaseChange], type[Command]] = {}

    def register(self, change_type: type[BaseChange], command_type: type[Command]) -> None:
        """
        Register a mapping from a Change type to a Command type.

        Parameters
        ----------
        change_type : type[BaseChange]
            The Change class to register.
        command_type : type[Command]
            The Command class that handles this Change type.
        """
        self._registry[change_type] = command_type

    def build(self, change: BaseChange, ctx: ApplicationContext) -> Command:
        """
        Build a Command from a Change using the registered mapping.

        Parameters
        ----------
        change : BaseChange
            The change to convert to a command.
        ctx : ApplicationContext
            Application context for command initialization.

        Returns
        -------
        Command
            Command instance with undo state initialized.

        Raises
        ------
        KeyError
            If the change type is not registered.
        """
        # Handle BatchChange specially to avoid circular dependency
        if isinstance(change, CompositeChange):
            commands = []
            for sub_change in change.changes:
                cmd = self.build(sub_change, ctx)
                commands.append(cmd)
            return CompositeCommand(commands=commands)

        cmd_cls = self._registry[type(change)]
        return cmd_cls.from_change(change, ctx)


def create_default_registry() -> CommandRegistry:
    """
    Create a CommandRegistry with default mappings registered.

    Returns
    -------
    CommandRegistry
        Registry with all default change-to-command mappings.
    """
    registry = CommandRegistry()
    registry.register(UpdateParameter, UpdateParameterCommand)
    registry.register(UpdateRegionSlice, UpdateRegionSliceCommand)
    registry.register(RemoveObject, RemoveObjectCommand)
    registry.register(CreateSpectrum, CreateSpectrumCommand)
    registry.register(CreateRegion, CreateRegionCommand)
    registry.register(CreatePeak, CreatePeakCommand)
    registry.register(CreateBackground, CreateBackgroundCommand)
    registry.register(ReplacePeakModel, ReplacePeakModelCommand)
    registry.register(ReplaceBackgroundModel, ReplaceBackgroundModelCommand)
    registry.register(UpdateMultipleParameterValues, UpdateMultipleParameterValuesCommand)
    registry.register(SetSpectrumMetadata, SetSpectrumMetadataCommand)
    registry.register(SetRegionMetadata, SetRegionMetadataCommand)
    registry.register(SetPeakMetadata, SetPeakMetadataCommand)
    # CompositeChange is handled specially in build() method
    return registry


class CommandExecutor:
    """
    Applies Change objects via CommandRegistry, runs commands, and manages undo/redo.
    """

    def __init__(
        self,
        ctx: ApplicationContext,
        stack: UndoRedoStack,
        registry: CommandRegistry | None = None,
    ) -> None:
        self.ctx = ctx
        self.stack = stack
        self._registry = registry if registry is not None else create_default_registry()

    def execute(self, change: BaseChange) -> None:
        """Map change to command, apply it, and push to the undo stack (clears redo)."""
        cmd = self._registry.build(change, self.ctx)
        cmd.apply(self.ctx)
        self.stack.push(cmd)

    def undo(self) -> None:
        """Pop the last command, undo it, and push to the redo stack."""
        if not self.stack.can_undo:
            raise IndexError("Nothing to undo")
        cmd = self.stack.pop_undo()
        cmd.undo(self.ctx)
        self.stack.push_redo(cmd)

    def redo(self) -> None:
        """Pop from redo stack, re-apply the command, and push to the undo stack."""
        if not self.stack.can_redo:
            raise IndexError("Nothing to redo")
        cmd = self.stack.pop_redo()
        cmd.apply(self.ctx)
        self.stack.push_undo(cmd)
