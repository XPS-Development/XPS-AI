"""
Core command infrastructure.

Command implementations and CommandRegistry live here to avoid circular imports
with .changes.
"""

from core.services import CoreContext
from .changes import (
    BaseChange,
    UpdateParameter,
    UpdateRegionSlice,
    RemoveObject,
    RemoveMetadata,
    FullRemoveObject,
    CreateSpectrum,
    CreateRegion,
    CreatePeak,
    CreateBackground,
    ReplacePeakModel,
    ReplaceBackgroundModel,
    UpdateMultipleParameterValues,
    SetMetadata,
    CompositeChange,
)
from .commands import (
    Command,
    UpdateParameterCommand,
    UpdateRegionSliceCommand,
    RemoveObjectCommand,
    RemoveMetadataCommand,
    FullRemoveObjectCommand,
    CreateSpectrumCommand,
    CreateRegionCommand,
    CreatePeakCommand,
    CreateBackgroundCommand,
    ReplacePeakModelCommand,
    ReplaceBackgroundModelCommand,
    UpdateMultipleParameterValuesCommand,
    SetMetadataCommand,
    CompositeCommand,
)


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

    def clear_all(self) -> None:
        """Clear both undo and redo stacks."""
        self._undo_stack.clear()
        self._redo_stack.clear()

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

    def peek_undo(self) -> Command | None:
        """
        Return the next command that would be undone, without popping.

        Returns
        -------
        Command or None
            The top of the undo stack, or None if the stack is empty.
        """
        if not self._undo_stack:
            return None
        return self._undo_stack[-1]

    def peek_redo(self) -> Command | None:
        """
        Return the next command that would be redone, without popping.

        Returns
        -------
        Command or None
            The top of the redo stack, or None if the stack is empty.
        """
        if not self._redo_stack:
            return None
        return self._redo_stack[-1]


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

    def build(self, change: BaseChange, ctx: CoreContext) -> Command:
        """
        Build a Command from a Change using the registered mapping.

        Parameters
        ----------
        change : BaseChange
            The change to convert to a command.
        ctx : CoreContext
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
    registry.register(RemoveMetadata, RemoveMetadataCommand)
    registry.register(FullRemoveObject, FullRemoveObjectCommand)
    registry.register(CreateSpectrum, CreateSpectrumCommand)
    registry.register(CreateRegion, CreateRegionCommand)
    registry.register(CreatePeak, CreatePeakCommand)
    registry.register(CreateBackground, CreateBackgroundCommand)
    registry.register(ReplacePeakModel, ReplacePeakModelCommand)
    registry.register(ReplaceBackgroundModel, ReplaceBackgroundModelCommand)
    registry.register(UpdateMultipleParameterValues, UpdateMultipleParameterValuesCommand)
    registry.register(SetMetadata, SetMetadataCommand)
    return registry


class CommandExecutor:
    """
    Applies Change objects via CommandRegistry, runs commands, and manages undo/redo.
    """

    def __init__(
        self,
        ctx: CoreContext,
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

    def peek_undo_command(self) -> Command | None:
        """Return the command that would be undone next, without modifying stacks."""
        return self.stack.peek_undo()

    def peek_redo_command(self) -> Command | None:
        """Return the command that would be redone next, without modifying stacks."""
        return self.stack.peek_redo()

    def clear(self) -> None:
        """Clear both undo and redo stacks."""
        self.stack.clear_all()
