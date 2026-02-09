"""
Command layer: Change objects, Command protocol, CommandRegistry, CommandExecutor.

Application services produce Change objects; CommandRegistry turns them into
Command objects (with undo state); CommandExecutor applies changes and
manages history.
"""

from .changes import (
    BaseChange,
    ParameterChange,
    SetParameterValue,
    UpdateRegionSlice,
    RemoveComponent,
)
from .core import (
    CommandRegistry,
    CommandHistory,
    CommandExecutor,
    create_default_registry,
)
from .utils import ApplicationContext
from .commands import (
    Command,
    CommandBatch,
    ParameterChangeCommand,
    UpdateRegionSliceCommand,
    RemoveComponentCommand,
)


__all__ = [
    "ApplicationContext",
    "BaseChange",
    "Command",
    "CommandBatch",
    "CommandExecutor",
    "CommandHistory",
    "CommandRegistry",
    "ComponentSnapshot",
    "ParameterChange",
    "ParameterChangeCommand",
    "RemoveComponent",
    "RemoveComponentCommand",
    "SetParameterValue",
    "UpdateRegionSlice",
    "UpdateRegionSliceCommand",
    "create_default_registry",
]
