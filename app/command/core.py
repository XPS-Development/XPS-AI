"""
Core command infrastructure.

Command implementations and CommandRegistry live here to avoid circular imports
with .changes.
"""

from .changes import (
    BaseChange,
    ParameterChange,
    SetParameterValue,
    UpdateRegionSlice,
    RemoveComponent,
)
from .commands import (
    Command,
    ParameterChangeCommand,
    UpdateRegionSliceCommand,
    RemoveComponentCommand,
    CommandBatch,
)
from .utils import ApplicationContext


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
        cmd_cls = self._registry[type(change)]
        return cmd_cls.from_change(change, ctx)


class CommandHistory:
    """
    Stack of applied commands for undo support.

    Stores Command instances in LIFO order. push() adds a command after
    it has been applied; pop() returns the last command for undo.
    """

    def __init__(self) -> None:
        self._stack: list[Command] = []

    def push(self, cmd: Command) -> None:
        """Append a command to the history (after it has been applied)."""
        self._stack.append(cmd)

    def pop(self) -> Command:
        """
        Remove and return the last command for undo.

        Returns
        -------
        Command
            The most recently pushed command.

        Raises
        ------
        IndexError
            If the history is empty.
        """
        return self._stack.pop()

    @property
    def can_undo(self) -> bool:
        """True if at least one command can be undone."""
        return len(self._stack) > 0

    def clear(self) -> None:
        """Remove all commands from the history."""
        self._stack.clear()


def create_default_registry() -> CommandRegistry:
    """
    Create a CommandRegistry with default mappings registered.

    Returns
    -------
    CommandRegistry
        Registry with all default change-to-command mappings.
    """
    registry = CommandRegistry()
    registry.register(SetParameterValue, ParameterChangeCommand)
    registry.register(ParameterChange, ParameterChangeCommand)
    registry.register(UpdateRegionSlice, UpdateRegionSliceCommand)
    registry.register(RemoveComponent, RemoveComponentCommand)
    return registry


class CommandExecutor:
    """
    Applies Change objects via CommandRegistry, runs commands, and manages history.
    """

    def __init__(
        self,
        ctx: ApplicationContext,
        history: CommandHistory,
        registry: CommandRegistry | None = None,
    ) -> None:
        self.ctx = ctx
        self.history = history
        self._registry = registry if registry is not None else create_default_registry()

    def execute(self, change: BaseChange) -> None:
        """Map change to command, apply it, and push to history."""
        cmd = self._registry.build(change, self.ctx)
        cmd.apply(self.ctx)
        self.history.push(cmd)

    def undo(self) -> None:
        """Pop the last command and undo it."""
        if not self.history.can_undo:
            raise IndexError("Nothing to undo")
        cmd = self.history.pop()
        cmd.undo(self.ctx)
