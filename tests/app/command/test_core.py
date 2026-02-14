"""Tests for UndoRedoStack, CommandRegistry, CommandExecutor, and create_default_registry."""

import pytest

from app.command.core import (
    UndoRedoStack,
    CommandRegistry,
    CommandExecutor,
    create_default_registry,
)
from app.command.changes import (
    UpdateParameter,
    UpdateRegionSlice,
    RemoveObject,
    UpdateMultipleParameterValues,
    CompositeChange,
)
from app.command.commands import (
    Command,
    UpdateParameterCommand,
    UpdateRegionSliceCommand,
    UpdateMultipleParameterValuesCommand,
    CompositeCommand,
)


class _DummyCommand(Command):
    """Minimal command for stack tests."""

    @classmethod
    def from_change(cls, change, ctx):
        return cls()

    def apply(self, ctx):
        pass

    def undo(self, ctx):
        pass


def test_undo_redo_stack_empty_cannot_undo_or_redo():
    """can_undo and can_redo are False when stack is empty."""
    stack = UndoRedoStack()
    assert stack.can_undo is False
    assert stack.can_redo is False


def test_undo_redo_stack_push_adds_to_undo_clears_redo():
    """push adds command to undo stack and clears redo."""
    stack = UndoRedoStack()
    cmd = _DummyCommand()
    stack.push(cmd)
    assert stack.can_undo is True
    assert stack.can_redo is False


def test_undo_redo_stack_pop_undo_returns_last_command():
    """pop_undo returns the last pushed command."""
    stack = UndoRedoStack()
    cmd = _DummyCommand()
    stack.push(cmd)
    popped = stack.pop_undo()
    assert popped is cmd
    assert stack.can_undo is False


def test_undo_redo_stack_pop_undo_empty_raises():
    """pop_undo raises IndexError when undo stack is empty."""
    stack = UndoRedoStack()
    with pytest.raises(IndexError):
        stack.pop_undo()


def test_undo_redo_stack_push_redo_pop_redo_flow():
    """push_redo and pop_redo work for redo flow."""
    stack = UndoRedoStack()
    cmd = _DummyCommand()
    stack.push(cmd)
    undone = stack.pop_undo()
    stack.push_redo(undone)
    assert stack.can_redo is True
    redone = stack.pop_redo()
    assert redone is cmd


def test_undo_redo_stack_pop_redo_empty_raises():
    """pop_redo raises IndexError when redo stack is empty."""
    stack = UndoRedoStack()
    with pytest.raises(IndexError):
        stack.pop_redo()


def test_undo_redo_stack_push_undo_after_redo():
    """push_undo adds redone command back to undo stack."""
    stack = UndoRedoStack()
    cmd = _DummyCommand()
    stack.push(cmd)
    undone = stack.pop_undo()
    stack.push_redo(undone)
    redone = stack.pop_redo()
    stack.push_undo(redone)
    assert stack.can_undo is True
    assert stack.can_redo is False


def test_undo_redo_stack_new_push_clears_redo():
    """Pushing a new command clears the redo stack (branching)."""
    stack = UndoRedoStack()
    cmd1 = _DummyCommand()
    cmd2 = _DummyCommand()
    stack.push(cmd1)
    stack.pop_undo()
    stack.push_redo(cmd1)
    assert stack.can_redo is True
    stack.push(cmd2)
    assert stack.can_undo is True
    assert stack.can_redo is False


def test_command_registry_build_returns_correct_command(app_context, peak_id, region_id):
    """build returns the correct Command for each registered Change type."""
    registry = create_default_registry()

    up_change = UpdateParameter(peak_id, "cen", "value", 5.0)
    cmd = registry.build(up_change, app_context)
    assert isinstance(cmd, UpdateParameterCommand)
    assert cmd.new_value == 5.0

    slice_change = UpdateRegionSlice(region_id, 25, 175)
    cmd = registry.build(slice_change, app_context)
    assert isinstance(cmd, UpdateRegionSliceCommand)
    assert cmd.new_start == 25
    assert cmd.new_stop == 175

    multi_change = UpdateMultipleParameterValues(peak_id, {"cen": 2.0, "amp": 10.0})
    cmd = registry.build(multi_change, app_context)
    assert isinstance(cmd, UpdateMultipleParameterValuesCommand)
    assert cmd.parameters == {"cen": 2.0, "amp": 10.0}


def test_command_registry_build_unregistered_raises(app_context):
    """build with unregistered Change type raises KeyError."""
    registry = CommandRegistry()
    change = UpdateParameter("p1", "cen", "value", 1.0)
    with pytest.raises(KeyError):
        registry.build(change, app_context)


def test_command_registry_build_composite_returns_composite_command(app_context, peak_id, region_id):
    """build with CompositeChange returns CompositeCommand with sub-commands."""
    registry = create_default_registry()
    changes = [
        UpdateParameter(peak_id, "cen", "value", 3.0),
        UpdateRegionSlice(region_id, 30, 170),
    ]
    composite_change = CompositeChange(changes=changes)
    cmd = registry.build(composite_change, app_context)
    assert isinstance(cmd, CompositeCommand)
    assert len(cmd.commands) == 2
    assert isinstance(cmd.commands[0], UpdateParameterCommand)
    assert isinstance(cmd.commands[1], UpdateRegionSliceCommand)


def test_create_default_registry_has_all_mappings():
    """create_default_registry returns registry with all default mappings."""
    from app.command.changes import SetSpectrumMetadata, SetRegionMetadata, SetPeakMetadata

    registry = create_default_registry()
    assert UpdateParameter in registry._registry
    assert UpdateRegionSlice in registry._registry
    assert RemoveObject in registry._registry
    assert UpdateMultipleParameterValues in registry._registry
    assert SetSpectrumMetadata in registry._registry
    assert SetRegionMetadata in registry._registry
    assert SetPeakMetadata in registry._registry


def test_command_executor_execute_applies_and_pushes(app_context, peak_id):
    """execute builds command, applies it, and pushes to stack."""
    stack = UndoRedoStack()
    executor = CommandExecutor(app_context, stack)
    change = UpdateParameter(peak_id, "cen", "value", 7.0)

    executor.execute(change)

    assert stack.can_undo is True
    param = app_context.component.get_parameter(peak_id, "cen")
    assert param.value == 7.0


def test_command_executor_undo_pops_undoes_pushes_redo(app_context, peak_id):
    """undo pops command, undoes it, and pushes to redo stack."""
    stack = UndoRedoStack()
    executor = CommandExecutor(app_context, stack)
    change = UpdateParameter(peak_id, "cen", "value", 9.0)
    executor.execute(change)
    original_value = 0.0

    executor.undo()

    assert stack.can_redo is True
    param = app_context.component.get_parameter(peak_id, "cen")
    assert param.value == original_value


def test_command_executor_undo_empty_raises(app_context):
    """undo raises IndexError when nothing to undo."""
    stack = UndoRedoStack()
    executor = CommandExecutor(app_context, stack)
    with pytest.raises(IndexError, match="Nothing to undo"):
        executor.undo()


def test_command_executor_redo_pops_applies_pushes_undo(app_context, peak_id):
    """redo pops from redo, re-applies command, and pushes to undo."""
    stack = UndoRedoStack()
    executor = CommandExecutor(app_context, stack)
    change = UpdateParameter(peak_id, "cen", "value", 11.0)
    executor.execute(change)
    executor.undo()

    executor.redo()

    assert stack.can_undo is True
    param = app_context.component.get_parameter(peak_id, "cen")
    assert param.value == 11.0


def test_command_executor_redo_empty_raises(app_context):
    """redo raises IndexError when nothing to redo."""
    stack = UndoRedoStack()
    executor = CommandExecutor(app_context, stack)
    with pytest.raises(IndexError, match="Nothing to redo"):
        executor.redo()


def test_command_executor_full_cycle_restores_state(app_context, peak_id):
    """execute -> undo -> redo restores state."""
    stack = UndoRedoStack()
    executor = CommandExecutor(app_context, stack)
    change = UpdateParameter(peak_id, "cen", "value", 13.0)
    executor.execute(change)
    executor.undo()
    executor.redo()

    param = app_context.component.get_parameter(peak_id, "cen")
    assert param.value == 13.0
