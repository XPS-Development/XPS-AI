"""Tests for UndoRedoStack, CommandRegistry, CommandExecutor, and create_default_registry."""

import pytest

from app.command.changes import (
    CompositeChange,
    RemoveObject,
    SetMetadata,
    UpdateMultipleParameterValues,
    UpdateParameter,
    UpdateRegionSlice,
)
from app.command.commands import (
    Command,
    CompositeCommand,
    UpdateMultipleParameterValuesCommand,
    UpdateParameterCommand,
    UpdateRegionSliceCommand,
)
from app.command.core import (
    CommandExecutor,
    CommandRegistry,
    UndoRedoStack,
    create_default_registry,
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


def test_undo_redo_stack_initial_state_not_dirty():
    """Empty stack with default save marker is clean."""
    stack = UndoRedoStack()
    assert stack.is_dirty is False


def test_undo_redo_stack_push_marks_dirty():
    """A new command moves the stack away from the saved depth."""
    stack = UndoRedoStack()
    stack.push(_DummyCommand())
    assert stack.is_dirty is True


def test_undo_redo_stack_mark_saved_clears_dirty():
    """mark_saved records the current depth as the on-disk baseline."""
    stack = UndoRedoStack()
    stack.push(_DummyCommand())
    stack.mark_saved()
    assert stack.is_dirty is False


def test_undo_redo_stack_undo_back_to_saved_depth_is_clean():
    """Undoing post-save edits back to the saved depth leaves the document clean."""
    stack = UndoRedoStack()
    stack.push(_DummyCommand())
    stack.mark_saved()
    stack.push(_DummyCommand())
    assert stack.is_dirty is True
    stack.pop_undo()
    assert stack.is_dirty is False


def test_undo_redo_stack_undo_before_saved_depth_is_dirty():
    """Undoing past the saved depth leaves the document dirty."""
    stack = UndoRedoStack()
    stack.push(_DummyCommand())
    stack.mark_saved()
    stack.push(_DummyCommand())
    stack.pop_undo()
    stack.pop_undo()
    assert stack.is_dirty is True


def test_undo_redo_stack_mark_unsaved():
    """mark_unsaved forces dirty even when the undo stack is empty."""
    stack = UndoRedoStack()
    stack.mark_unsaved()
    assert stack.is_dirty is True


def test_undo_redo_stack_clear_all_does_not_reset_save_marker():
    """clear_all only empties stacks; the caller must update the save marker."""
    stack = UndoRedoStack()
    stack.push(_DummyCommand())
    stack.mark_saved()
    stack.clear_all()
    assert stack.is_dirty is True
    stack.mark_saved()
    assert stack.is_dirty is False


def test_command_registry_build_returns_correct_command(ctx, peak_id, region_id):
    """build returns the correct Command for each registered Change type."""
    registry = create_default_registry()

    up_change = UpdateParameter(peak_id, "cen", "value", 5.0)
    cmd = registry.build(up_change, ctx)
    assert isinstance(cmd, UpdateParameterCommand)
    assert cmd.new_value == 5.0

    slice_change = UpdateRegionSlice(region_id, 25, 175)
    cmd = registry.build(slice_change, ctx)
    assert isinstance(cmd, UpdateRegionSliceCommand)
    assert cmd.new_start == 25
    assert cmd.new_stop == 175

    multi_change = UpdateMultipleParameterValues(peak_id, {"cen": 2.0, "amp": 10.0})
    cmd = registry.build(multi_change, ctx)
    assert isinstance(cmd, UpdateMultipleParameterValuesCommand)
    assert cmd.parameters == {"cen": 2.0, "amp": 10.0}


def test_command_registry_build_unregistered_raises(ctx):
    """build with unregistered Change type raises KeyError."""
    registry = CommandRegistry()
    change = UpdateParameter("p1", "cen", "value", 1.0)
    with pytest.raises(KeyError):
        registry.build(change, ctx)


def test_command_registry_build_composite_returns_composite_command(ctx, peak_id, region_id):
    """build with CompositeChange returns CompositeCommand with sub-commands."""
    registry = create_default_registry()
    changes = [
        UpdateParameter(peak_id, "cen", "value", 3.0),
        UpdateRegionSlice(region_id, 30, 170),
    ]
    composite_change = CompositeChange(changes=changes)
    cmd = registry.build(composite_change, ctx)
    assert isinstance(cmd, CompositeCommand)
    assert len(cmd.commands) == 2
    assert isinstance(cmd.commands[0], UpdateParameterCommand)
    assert isinstance(cmd.commands[1], UpdateRegionSliceCommand)


def test_create_default_registry_has_all_mappings():
    """create_default_registry returns registry with all default mappings."""
    registry = create_default_registry()
    assert UpdateParameter in registry._registry
    assert UpdateRegionSlice in registry._registry
    assert RemoveObject in registry._registry
    assert UpdateMultipleParameterValues in registry._registry
    assert SetMetadata in registry._registry


def test_command_executor_execute_applies_and_pushes(ctx, peak_id):
    """execute builds command, applies it, pushes to stack, and returns the command."""
    from app.command.refresh import UiRefresh

    stack = UndoRedoStack()
    executor = CommandExecutor(ctx, stack)
    change = UpdateParameter(peak_id, "cen", "value", 7.0)

    cmd = executor.execute(change)

    assert stack.can_undo is True
    assert cmd.combined_ui_refresh() == UiRefresh.FIT
    param = ctx.component.get_parameter(peak_id, "cen")
    assert param["value"] == 7.0


def test_command_executor_undo_pops_undoes_pushes_redo(ctx, peak_id):
    """undo pops command, undoes it, and pushes to redo stack."""
    stack = UndoRedoStack()
    executor = CommandExecutor(ctx, stack)
    change = UpdateParameter(peak_id, "cen", "value", 9.0)
    executor.execute(change)
    original_value = 0.0

    executor.undo()

    assert stack.can_redo is True
    param = ctx.component.get_parameter(peak_id, "cen")
    assert param["value"] == original_value


def test_command_executor_undo_empty_raises(ctx):
    """undo raises IndexError when nothing to undo."""
    stack = UndoRedoStack()
    executor = CommandExecutor(ctx, stack)
    with pytest.raises(IndexError, match="Nothing to undo"):
        executor.undo()


def test_command_executor_redo_pops_applies_pushes_undo(ctx, peak_id):
    """redo pops from redo, re-applies command, and pushes to undo."""
    stack = UndoRedoStack()
    executor = CommandExecutor(ctx, stack)
    change = UpdateParameter(peak_id, "cen", "value", 11.0)
    executor.execute(change)
    executor.undo()

    executor.redo()

    assert stack.can_undo is True
    param = ctx.component.get_parameter(peak_id, "cen")
    assert param["value"] == 11.0


def test_command_executor_redo_empty_raises(ctx):
    """redo raises IndexError when nothing to redo."""
    stack = UndoRedoStack()
    executor = CommandExecutor(ctx, stack)
    with pytest.raises(IndexError, match="Nothing to redo"):
        executor.redo()


def test_command_executor_full_cycle_restores_state(ctx, peak_id):
    """execute -> undo -> redo restores state."""
    stack = UndoRedoStack()
    executor = CommandExecutor(ctx, stack)
    change = UpdateParameter(peak_id, "cen", "value", 13.0)
    executor.execute(change)
    executor.undo()
    executor.redo()

    param = ctx.component.get_parameter(peak_id, "cen")
    assert param["value"] == 13.0
