from dataclasses import replace

import numpy as np
import pytest

from app.command.refresh import UiRefresh
from app.orchestration import AppOrchestrator
from app.parameters import AppParameters
from core.io.serialization import deserialize, serialize
from core.math_models import PseudoVoigtPeakModel
from core.objects import Peak, Region, Spectrum
from core.post_processing import LinkConstraints, MatchingSettings, MetricDefinition, match_series


@pytest.fixture
def orchestrator(simple_collection):
    x = np.linspace(0, 10, 30)
    simple_collection.add(Spectrum(x, x + 1, id_="s2"))
    simple_collection.add(Region(slice(0, 20), parent_id="s2", id_="r2"))
    simple_collection.add(
        Peak(
            model=PseudoVoigtPeakModel(),
            region_id="r2",
            component_id="p2",
            amp=2,
            cen=0.2,
            sig=1,
            frac=0,
        )
    )
    return AppOrchestrator(simple_collection, AppParameters())


def run(orchestrator, series_id):
    job = orchestrator.prepare_post_processing(series_id)
    orchestrator.apply_post_processing_result(job, match_series(job.matching))
    return orchestrator.query.get_post_processing_series(series_id)


def create(orchestrator):
    return orchestrator.create_post_processing_series("Series", ["s1", "s2", "s1"])


def test_series_and_result_are_undoable_single_commands(orchestrator):
    sid = create(orchestrator)
    assert orchestrator.query.get_post_processing_series(sid).series.spectrum_ids == ("s1", "s2")
    assert orchestrator.is_dirty
    flags = orchestrator.consume_pending_ui_refresh()
    assert flags == UiRefresh.POST_PROCESSING | UiRefresh.DOCUMENT
    matched = run(orchestrator, sid)
    assert matched.exportable
    assert matched.series.result.tracks[0].component_ids == ("p1", "p2")
    orchestrator.undo()
    assert orchestrator.query.get_post_processing_series(sid).series.result is None
    orchestrator.redo()
    assert orchestrator.query.get_post_processing_series(sid) == matched
    orchestrator.undo()
    orchestrator.undo()
    assert orchestrator.query.get_post_processing_series_ids() == ()
    orchestrator.redo()
    assert orchestrator.query.get_post_processing_series(sid).series.result is None


def test_edit_and_delete_series_preserve_full_undo_state(orchestrator):
    sid = create(orchestrator)
    original = run(orchestrator, sid)
    orchestrator.edit_post_processing_series(
        sid, spectrum_ids=("s2", "s1"), settings=MatchingSettings(energy_gate=3)
    )
    edited = orchestrator.query.get_post_processing_series(sid)
    assert edited.stale
    assert edited.series.result == original.series.result
    assert edited.series.spectrum_ids == ("s2", "s1")
    orchestrator.undo()
    assert orchestrator.query.get_post_processing_series(sid) == original
    orchestrator.remove_post_processing_series(sid)
    assert orchestrator.query.get_post_processing_series_ids() == ()
    orchestrator.undo()
    assert orchestrator.query.get_post_processing_series(sid) == original


def test_fit_changes_invalidate_and_undo_restores_snapshot(orchestrator):
    sid = create(orchestrator)
    before = run(orchestrator, sid)
    orchestrator.update_parameter("p1", "amp", "value", 5)
    after = orchestrator.query.get_post_processing_series(sid)
    assert after.stale
    assert not after.exportable
    assert after.series.result == before.series.result
    orchestrator.undo()
    assert orchestrator.query.get_post_processing_series(sid) == before
    orchestrator.redo()
    assert orchestrator.query.get_post_processing_series(sid).stale


def test_removal_preserves_step_and_undo_restores_availability(orchestrator):
    sid = create(orchestrator)
    original = run(orchestrator, sid)
    orchestrator.full_remove_object("s1")
    current = orchestrator.query.get_post_processing_series(sid)
    assert current.unavailable_spectrum_ids == ("s1",)
    assert current.series.spectrum_ids == ("s1", "s2")
    assert current.series.result == original.series.result
    assert not current.exportable
    job = orchestrator.prepare_post_processing(sid)
    assert not job.matching.steps[0].available
    assert job.matching.steps[0].peaks == ()
    orchestrator.undo()
    assert orchestrator.query.get_post_processing_series(sid) == original


def test_columns_and_names_update_values_without_invalidating_matching(orchestrator):
    sid = create(orchestrator)
    run(orchestrator, sid)
    metric = MetricDefinition(
        "shift", "Energy shift", "position_shift", ("track/p1",), reference_spectrum_id="s1"
    )
    orchestrator.edit_post_processing_series(
        sid, name="Renamed", metrics=(metric,), column_order=("shift", "position")
    )
    orchestrator.name_post_processing_track(sid, "track/p1", "Carbon")
    state = orchestrator.query.get_post_processing_series(sid)
    assert not state.stale
    assert [cell.value for cell in state.values] == [0, 0.2]
    assert state.series.track_names == (("p1", "Carbon"),)
    assert state.series.column_order == ("shift", "position")
    assert run(orchestrator, sid).series.track_names == state.series.track_names
    orchestrator.name_post_processing_track(sid, "track/p1", None)
    assert orchestrator.query.get_post_processing_series(sid).series.track_names == ()
    orchestrator.undo()
    assert orchestrator.query.get_post_processing_series(sid).series.track_names == (
        ("p1", "Carbon"),
    )


def test_dangling_metric_operands_require_selection_after_recalculation(orchestrator):
    sid = create(orchestrator)
    run(orchestrator, sid)
    metric = MetricDefinition("area", "Area", "area", ("track/p1",))
    orchestrator.edit_post_processing_series(sid, metrics=(metric,), spectrum_ids=("s2",))
    state = run(orchestrator, sid)
    assert state.series.result.tracks[0].id_ == "track/p2"
    assert state.values[0].reason == "operand_selection_required"
    assert state.conflicts == ("operand_selection_required:area:track/p1",)
    assert not state.exportable


def test_removed_reference_is_not_replaced(orchestrator):
    sid = create(orchestrator)
    run(orchestrator, sid)
    metric = MetricDefinition(
        "shift", "Shift", "position_shift", ("track/p1",), reference_spectrum_id="s2"
    )
    orchestrator.edit_post_processing_series(sid, metrics=(metric,), spectrum_ids=("s1",))
    state = run(orchestrator, sid)
    assert state.values[0].reason == "reference_selection_required"
    assert "reference_selection_required:shift" in state.conflicts


def test_named_anchor_can_move_inside_track_without_changing_identity(orchestrator):
    sid = orchestrator.create_post_processing_series("S", ["s2"])
    run(orchestrator, sid)
    orchestrator.name_post_processing_track(sid, "track/p2", "Named")
    orchestrator.edit_post_processing_series(sid, spectrum_ids=("s1", "s2"))
    state = run(orchestrator, sid)
    assert state.series.result.tracks[0].id_ == "track/p2"
    assert state.series.result.tracks[0].anchor_id == "p2"
    assert state.series.result.tracks[0].component_ids == ("p1", "p2")


def test_lost_named_anchor_is_explicit_conflict(orchestrator):
    sid = create(orchestrator)
    run(orchestrator, sid)
    orchestrator.name_post_processing_track(sid, "track/p1", "Named")
    orchestrator.edit_post_processing_series(sid, spectrum_ids=("s2",))
    job = orchestrator.prepare_post_processing(sid)
    with pytest.raises(ValueError, match="anchor disappeared"):
        orchestrator.apply_post_processing_result(job, match_series(job.matching))
    orchestrator.name_post_processing_track(sid, "track/p1", None)
    assert run(orchestrator, sid).exportable


def test_constraints_persist_across_recalculation(orchestrator):
    sid = create(orchestrator)
    run(orchestrator, sid)
    orchestrator.edit_post_processing_series(
        sid, constraints=LinkConstraints(forbidden=(("p1", "p2"),))
    )
    state = run(orchestrator, sid)
    assert len(state.series.result.tracks) == 2
    state = run(orchestrator, sid)
    assert len(state.series.result.tracks) == 2
    orchestrator.edit_post_processing_series(
        sid, constraints=LinkConstraints(required=(("p1", "p2"),))
    )
    state = run(orchestrator, sid)
    assert state.series.result.tracks[0].links[0].manual


@pytest.mark.parametrize(
    "mutation", ["fit", "settings", "columns", "remove_series", "new", "replace", "append"]
)
def test_outdated_jobs_are_rejected_without_adding_undo_entry(orchestrator, mutation):
    sid = create(orchestrator)
    job = orchestrator.prepare_post_processing(sid)
    result = match_series(job.matching)
    if mutation == "fit":
        orchestrator.update_parameter("p1", "amp", "value", 5)
    elif mutation == "settings":
        orchestrator.edit_post_processing_series(sid, settings=MatchingSettings(energy_gate=3))
    elif mutation == "columns":
        orchestrator.edit_post_processing_series(sid, column_order=("area",))
    elif mutation == "remove_series":
        orchestrator.remove_post_processing_series(sid)
    elif mutation == "new":
        orchestrator.new_collection()
    else:
        deserialize(
            serialize(orchestrator.core_collection),
            orchestrator.core_collection,
            orchestrator.ctx.metadata,
            mode=mutation,
        )
    previous = orchestrator.peek_undo_command()
    with pytest.raises((ValueError, KeyError), match=r"changed|series|[0-9a-f]+"):
        orchestrator.apply_post_processing_result(job, result)
    assert orchestrator.peek_undo_command() is previous


def test_wrong_result_is_rejected(orchestrator):
    sid = create(orchestrator)
    job = orchestrator.prepare_post_processing(sid)
    result = match_series(replace(job.matching, id_="other"))
    with pytest.raises(ValueError, match="does not belong"):
        orchestrator.apply_post_processing_result(job, result)


def test_document_new_and_replace_reset_series_and_undo(orchestrator, tmp_path):
    sid = create(orchestrator)
    before = run(orchestrator, sid)
    path = tmp_path / "series.json"
    orchestrator.dump_collection(path)
    assert not orchestrator.is_dirty
    orchestrator.new_collection()
    assert orchestrator.query.get_post_processing_series_ids() == ()
    orchestrator.load_collection(path)
    assert orchestrator.query.get_post_processing_series(sid) == before
    assert not orchestrator.can_undo
    assert not orchestrator.is_dirty


def test_merged_named_tracks_require_explicit_conflict_resolution(orchestrator):
    sid = create(orchestrator)
    orchestrator.edit_post_processing_series(
        sid, constraints=LinkConstraints(forbidden=(("p1", "p2"),))
    )
    run(orchestrator, sid)
    orchestrator.name_post_processing_track(sid, "track/p1", "First")
    orchestrator.name_post_processing_track(sid, "track/p2", "Second")
    orchestrator.edit_post_processing_series(sid, constraints=LinkConstraints())
    job = orchestrator.prepare_post_processing(sid)
    with pytest.raises(ValueError, match="Several named anchors merged"):
        orchestrator.apply_post_processing_result(job, match_series(job.matching))
    orchestrator.name_post_processing_track(sid, "track/p2", None)
    state = run(orchestrator, sid)
    assert state.series.track_names == (("p1", "First"),)


def test_reordered_constraints_report_conflicts_without_silent_removal(orchestrator):
    sid = create(orchestrator)
    orchestrator.edit_post_processing_series(
        sid, constraints=LinkConstraints(required=(("p1", "p2"),))
    )
    run(orchestrator, sid)
    orchestrator.edit_post_processing_series(sid, spectrum_ids=("s2", "s1"))
    state = orchestrator.query.get_post_processing_series(sid)
    assert state.stale
    assert state.conflicts[0].startswith("constraint_conflict:")
    assert state.series.constraints.required == (("p1", "p2"),)
    job = orchestrator.prepare_post_processing(sid)
    with pytest.raises(ValueError, match="adjacent"):
        match_series(job.matching)


def test_region_bounds_invalidate_but_spectrum_and_component_names_do_not(orchestrator):
    sid = create(orchestrator)
    run(orchestrator, sid)
    orchestrator.rename_component("p1", "New name")
    orchestrator.rename_spectrum("s1", "Renamed spectrum")
    assert not orchestrator.query.get_post_processing_series(sid).stale
    orchestrator.update_region_slice("r1", 30, 160)
    assert orchestrator.query.get_post_processing_series(sid).stale
    orchestrator.undo()
    assert not orchestrator.query.get_post_processing_series(sid).stale


def test_series_of_empty_spectrum_is_valid_and_order_is_explicit(orchestrator):
    orchestrator.create_spectrum(np.arange(5.0), np.arange(5.0) + 1, spectrum_id="empty")
    sid = orchestrator.create_post_processing_series("Empty", ["empty"])
    state = run(orchestrator, sid)
    assert state.exportable
    assert state.series.result.tracks == ()
    assert [step.spectrum_id for step in state.series.result.series.steps] == ["empty"]


def test_failed_load_retains_series_history_and_dirty_state(orchestrator, tmp_path):
    sid = create(orchestrator)
    before = run(orchestrator, sid)
    previous = orchestrator.peek_undo_command()
    path = tmp_path / "broken.json"
    path.write_text('{"version": "1.1", "objects": [], "post_processing": {}}')
    with pytest.raises(ValueError, match="Expected a JSON array"):
        orchestrator.load_collection(path)
    assert orchestrator.query.get_post_processing_series(sid) == before
    assert orchestrator.peek_undo_command() is previous
    assert orchestrator.is_dirty
