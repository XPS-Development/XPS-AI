"""Tests for individual Command classes: from_change, apply, undo."""

import pytest

from core.metadata import SpectrumMetadata

from app.command.changes import (
    UpdateParameter,
    UpdateRegionSlice,
    UpdateMultipleParameterValues,
    RemoveObject,
    CreateSpectrum,
    CreateRegion,
    CreatePeak,
    CreateBackground,
    ReplacePeakModel,
    ReplaceBackgroundModel,
    SetMetadata,
)
from app.command.commands import (
    UpdateParameterCommand,
    UpdateRegionSliceCommand,
    UpdateMultipleParameterValuesCommand,
    RemoveObjectCommand,
    CreateSpectrumCommand,
    CreateRegionCommand,
    CreatePeakCommand,
    CreateBackgroundCommand,
    CompositeCommand,
    ReplacePeakModelCommand,
    ReplaceBackgroundModelCommand,
    SetMetadataCommand,
)


def test_update_parameter_command_from_change_captures_old_value(app_context, peak_id):
    """from_change captures old value from context."""
    change = UpdateParameter(peak_id, "cen", "value", 5.0)
    cmd = UpdateParameterCommand.from_change(change, app_context)
    assert cmd.new_value == 5.0
    assert cmd._old_value == 0.0


def test_update_parameter_command_apply_updates_parameter(app_context, peak_id):
    """apply updates the parameter value."""
    change = UpdateParameter(peak_id, "cen", "value", 5.0)
    cmd = UpdateParameterCommand.from_change(change, app_context)
    cmd.apply(app_context)
    param = app_context.component.get_parameter(peak_id, "cen")
    assert param.value == 5.0


def test_update_parameter_command_undo_restores_old_value(app_context, peak_id):
    """undo restores the old value."""
    change = UpdateParameter(peak_id, "cen", "value", 5.0)
    cmd = UpdateParameterCommand.from_change(change, app_context)
    cmd.apply(app_context)
    cmd.undo(app_context)
    param = app_context.component.get_parameter(peak_id, "cen")
    assert param.value == 0.0


def test_update_parameter_command_undo_without_apply_raises(app_context, peak_id):
    """undo without prior apply raises RuntimeError."""
    cmd = UpdateParameterCommand(
        component_id=peak_id,
        name="cen",
        parameter_field="value",
        new_value=5.0,
        old_value=None,
    )
    with pytest.raises(RuntimeError, match="Command was not applied"):
        cmd.undo(app_context)


def test_update_region_slice_command_from_change_captures_old_slice(app_context, region_id):
    """from_change captures old slice from context."""
    change = UpdateRegionSlice(region_id, 25, 175)
    cmd = UpdateRegionSliceCommand.from_change(change, app_context)
    assert cmd.new_start == 25
    assert cmd.new_stop == 175
    assert cmd.old_start == 20
    assert cmd.old_stop == 181


def test_update_region_slice_command_apply_undo_roundtrip(app_context, region_id):
    """apply updates slice; undo restores old slice."""
    change = UpdateRegionSlice(region_id, 25, 175)
    cmd = UpdateRegionSliceCommand.from_change(change, app_context)
    cmd.apply(app_context)
    sl = app_context.region.get_slice(region_id)
    assert sl.start == 25
    assert sl.stop == 175

    cmd.undo(app_context)
    sl = app_context.region.get_slice(region_id)
    assert sl.start == 20
    assert sl.stop == 181


def test_update_multiple_parameter_values_command_roundtrip(app_context, peak_id):
    """from_change captures old values; apply/undo roundtrip."""
    change = UpdateMultipleParameterValues(peak_id, {"cen": 2.0, "amp": 10.0})
    cmd = UpdateMultipleParameterValuesCommand.from_change(change, app_context)
    cmd.apply(app_context)
    params = app_context.component.get_parameters(peak_id)
    assert params["cen"].value == 2.0
    assert params["amp"].value == 10.0

    cmd.undo(app_context)
    params = app_context.component.get_parameters(peak_id)
    assert params["cen"].value == 0.0
    assert params["amp"].value == 1.0


def test_remove_object_command_from_change_nonexistent_raises(app_context):
    """from_change raises ValueError if object does not exist."""
    change = RemoveObject("nonexistent_id")
    with pytest.raises(ValueError, match="does not exist in collection"):
        RemoveObjectCommand.from_change(change, app_context)


def test_remove_object_command_apply_detaches(app_context, peak_id):
    """apply detaches object from collection."""
    change = RemoveObject(peak_id)
    cmd = RemoveObjectCommand.from_change(change, app_context)
    cmd.apply(app_context)
    assert not app_context.collection.check_object_exists(peak_id)


def test_remove_object_command_undo_reattaches(app_context, peak_id):
    """undo re-attaches detached objects."""
    change = RemoveObject(peak_id)
    cmd = RemoveObjectCommand.from_change(change, app_context)
    cmd.apply(app_context)
    cmd.undo(app_context)
    assert app_context.collection.check_object_exists(peak_id)


def test_remove_object_command_undo_without_apply_raises(app_context, peak_id):
    """undo without apply raises RuntimeError."""
    cmd = RemoveObjectCommand(obj_id=peak_id)
    with pytest.raises(RuntimeError, match="Command was not applied"):
        cmd.undo(app_context)


def test_create_spectrum_command_apply_undo(app_context, x_axis, simple_gauss, noise):
    """from_change + apply adds spectrum; undo removes it."""
    y = simple_gauss + noise + 1.0
    change = CreateSpectrum(x=x_axis, y=y, spectrum_id="s2")
    cmd = CreateSpectrumCommand.from_change(change, app_context)
    cmd.apply(app_context)
    assert app_context.collection.check_object_exists("s2")

    cmd.undo(app_context)
    assert not app_context.collection.check_object_exists("s2")


def test_create_region_command_from_change_invalid_slice_raises(app_context, spectrum_id):
    """from_change raises ValueError for invalid slice (start >= stop)."""
    change = CreateRegion(spectrum_id=spectrum_id, start=50, stop=30, region_id="r2")
    with pytest.raises(ValueError):
        CreateRegionCommand.from_change(change, app_context)


def test_create_region_command_from_change_out_of_bounds_raises(app_context, spectrum_id):
    """from_change raises ValueError for invalid slice (out of bounds)."""
    change = CreateRegion(spectrum_id=spectrum_id, start=-1, stop=50, region_id="r2")
    with pytest.raises(ValueError):
        CreateRegionCommand.from_change(change, app_context)


def test_create_region_command_apply_undo(app_context, spectrum_id):
    """Valid slice: apply adds region; undo removes it."""
    change = CreateRegion(spectrum_id=spectrum_id, start=50, stop=100, region_id="r2")
    cmd = CreateRegionCommand.from_change(change, app_context)
    cmd.apply(app_context)
    assert app_context.collection.check_object_exists("r2")

    cmd.undo(app_context)
    assert not app_context.collection.check_object_exists("r2")


def test_create_peak_command_apply_undo(app_context, region_id):
    """from_change + apply adds peak; undo removes it."""
    change = CreatePeak(
        region_id=region_id,
        model_name="pseudo-voigt",
        parameters={"cen": 1.0, "amp": 5.0},
        peak_id="p2",
    )
    cmd = CreatePeakCommand.from_change(change, app_context)
    cmd.apply(app_context)
    assert app_context.collection.check_object_exists("p2")

    cmd.undo(app_context)
    assert not app_context.collection.check_object_exists("p2")


def test_create_background_command_apply_undo(app_context, region_id):
    """from_change + apply adds background; undo removes it."""
    change = CreateBackground(
        region_id=region_id,
        model_name="linear",
        parameters={"i1": 0.1, "i2": 0.5},
        background_id="b2",
    )
    cmd = CreateBackgroundCommand.from_change(change, app_context)
    cmd.apply(app_context)
    assert app_context.collection.check_object_exists("b2")

    cmd.undo(app_context)
    assert not app_context.collection.check_object_exists("b2")


def test_composite_command_apply_runs_in_order(app_context, peak_id, region_id):
    """apply runs sub-commands in order."""
    cmd1 = UpdateParameterCommand.from_change(UpdateParameter(peak_id, "cen", "value", 3.0), app_context)
    cmd2 = UpdateRegionSliceCommand.from_change(UpdateRegionSlice(region_id, 30, 170), app_context)
    composite = CompositeCommand(commands=[cmd1, cmd2])
    composite.apply(app_context)

    param = app_context.component.get_parameter(peak_id, "cen")
    assert param.value == 3.0
    sl = app_context.region.get_slice(region_id)
    assert sl.start == 30
    assert sl.stop == 170


def test_composite_command_undo_runs_reverse_order(app_context, peak_id, region_id):
    """undo runs sub-commands in reverse order."""
    cmd1 = UpdateParameterCommand.from_change(UpdateParameter(peak_id, "cen", "value", 3.0), app_context)
    cmd2 = UpdateRegionSliceCommand.from_change(UpdateRegionSlice(region_id, 30, 170), app_context)
    composite = CompositeCommand(commands=[cmd1, cmd2])
    composite.apply(app_context)
    composite.undo(app_context)

    param = app_context.component.get_parameter(peak_id, "cen")
    assert param.value == 0.0
    sl = app_context.region.get_slice(region_id)
    assert sl.start == 20
    assert sl.stop == 181


def test_replace_peak_model_command_roundtrip(app_context, peak_id):
    """ReplacePeakModelCommand apply/undo roundtrip."""
    change = ReplacePeakModel(
        peak_id=peak_id,
        new_model_name="pseudo-voigt",
        parameters={"cen": 2.0, "amp": 10.0},
    )
    cmd = ReplacePeakModelCommand.from_change(change, app_context)
    cmd.apply(app_context)
    param = app_context.component.get_parameter(peak_id, "cen")
    assert param.value == 2.0

    cmd.undo(app_context)
    params = app_context.component.get_parameters(peak_id)
    assert params["cen"].value == 0.0
    assert params["amp"].value == 1.0


def test_replace_background_model_command_roundtrip(app_context, region_id):
    """ReplaceBackgroundModelCommand apply/undo roundtrip."""
    change = ReplaceBackgroundModel(
        region_id=region_id,
        new_model_name="linear",
        parameters={"i1": 0.2, "i2": 1.0},
    )
    cmd = ReplaceBackgroundModelCommand.from_change(change, app_context)
    cmd.apply(app_context)
    bg_id = app_context.collection.get_background(region_id)
    assert bg_id is not None
    params = app_context.component.get_parameters(bg_id)
    assert params["i1"].value == 0.2
    assert params["i2"].value == 1.0

    cmd.undo(app_context)
    bg_id = app_context.collection.get_background(region_id)
    assert bg_id is not None
    params = app_context.component.get_parameters(bg_id)
    assert params["const"].value == 1.0


def test_set_metadata_command_from_change_captures_old(app_context, spectrum_id):
    """from_change captures old spectrum metadata (None when none exists)."""
    metadata = SpectrumMetadata(name="Sample", group="Group1", file="data.vms")
    change = SetMetadata(obj_id=spectrum_id, metadata=metadata)
    cmd = SetMetadataCommand.from_change(change, app_context)
    assert cmd.metadata == metadata
    assert cmd._old_metadata is None


def test_set_metadata_command_apply_undo_roundtrip(app_context, spectrum_id):
    """apply sets metadata; undo restores old metadata (removes when none existed)."""
    metadata = SpectrumMetadata(name="Sample A", group="Group 1", file="/path/file.vms")
    change = SetMetadata(obj_id=spectrum_id, metadata=metadata)
    cmd = SetMetadataCommand.from_change(change, app_context)
    cmd.apply(app_context)
    assert app_context.metadata.get_metadata(spectrum_id) == metadata

    cmd.undo(app_context)
    assert app_context.metadata.get_metadata(spectrum_id) is None


def test_metadata_command_execute_via_executor(app_context, spectrum_id):
    """CommandExecutor executes SetMetadata and supports undo/redo."""
    from app.command.core import CommandExecutor, UndoRedoStack

    stack = UndoRedoStack()
    executor = CommandExecutor(app_context, stack)
    metadata = SpectrumMetadata(name="Sample", group="Group", file="file.vms")
    change = SetMetadata(obj_id=spectrum_id, metadata=metadata)

    executor.execute(change)
    assert app_context.metadata.get_metadata(spectrum_id) == metadata

    executor.undo()
    assert app_context.metadata.get_metadata(spectrum_id) is None

    executor.redo()
    assert app_context.metadata.get_metadata(spectrum_id) == metadata


def test_replace_background_model_command_no_existing_background(x_axis, simple_gauss, noise):
    """ReplaceBackgroundModel when region has no existing background (rm_ch is None)."""
    from app.command.utils import ApplicationContext
    from core.collection import CoreCollection
    from core.objects import Spectrum, Region

    collection = CoreCollection()
    x, y = x_axis, simple_gauss + noise + 1.0
    s = Spectrum(x, y, id_="s1")
    r = Region(slice(20, 181), parent_id=s.id_, id_="r1")
    collection.add(s)
    collection.add(r)
    ctx = ApplicationContext.from_collection(collection)

    change = ReplaceBackgroundModel(
        region_id="r1",
        new_model_name="linear",
        parameters={"i1": 0.1, "i2": 0.5},
    )
    cmd = ReplaceBackgroundModelCommand.from_change(change, ctx)
    cmd.apply(ctx)
    assert ctx.collection.get_background("r1") is not None

    cmd.undo(ctx)
    assert ctx.collection.get_background("r1") is None
