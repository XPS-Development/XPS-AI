from dataclasses import replace

import numpy as np
import pytest

from app.command.refresh import UiRefresh
from app.orchestration import AppOrchestrator
from app.parameters import AppParameters
from core.io.serialization import deserialize, serialize
from core.math_models import PseudoVoigtPeakModel
from core.objects import Peak, Region, Spectrum
from core.post_processing import LinkConstraints, MatchingSettings, match_series


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
    created = orchestrator.query.get_post_processing_series(sid)
    assert created.exportable
    assert orchestrator.is_dirty
    flags = orchestrator.consume_pending_ui_refresh()
    assert flags == UiRefresh.POST_PROCESSING | UiRefresh.DOCUMENT
    matched = run(orchestrator, sid)
    assert matched.exportable
    assert matched.series.result.tracks[0].component_ids == ("p1", "p2")
    orchestrator.undo()
    assert orchestrator.query.get_post_processing_series(sid) == created
    orchestrator.redo()
    assert orchestrator.query.get_post_processing_series(sid) == matched
    orchestrator.undo()
    orchestrator.undo()
    assert orchestrator.query.get_post_processing_series_ids() == ()
    orchestrator.redo()
    assert orchestrator.query.get_post_processing_series(sid) == created


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
    tid = orchestrator.create_analytical_table("Analysis", ((sid, "track/p1"),))
    before = orchestrator.query.get_analytical_table(tid)
    orchestrator.name_post_processing_track(sid, "track/p1", "Carbon")
    orchestrator.edit_post_processing_series(sid, name="Renamed")
    dto = orchestrator.query.get_analytical_table(tid)
    assert not dto.stale
    assert dto.table.columns[2].label.startswith("Renamed / Carbon")
    assert dto.state.sources == before.state.sources
    assert not hasattr(orchestrator.query.get_post_processing_series(sid).series, "metrics")


def test_dangling_metric_operands_require_selection_after_recalculation(orchestrator):
    sid = create(orchestrator)
    run(orchestrator, sid)
    tid = orchestrator.create_analytical_table("Analysis", ((sid, "track/p1"),))
    orchestrator.edit_analytical_table(tid, selections=())
    dto = orchestrator.query.get_analytical_table(tid)
    assert dto.conflicts[0].startswith("operand_selection_required")
    assert not dto.exportable


def test_removed_reference_is_not_replaced(orchestrator):
    sid = create(orchestrator)
    run(orchestrator, sid)
    tid = orchestrator.create_analytical_table("Analysis", ((sid, "track/p1"),))
    state = orchestrator.query.get_analytical_table(tid).state
    orchestrator.save_analytical_metric(
        tid,
        name="Shift",
        operation="position_shift",
        tracks=(state.sources[0].id_,),
        reference_row_id=state.rows[1].id_,
    )
    orchestrator.edit_post_processing_series(sid, spectrum_ids=("s1",))
    run(orchestrator, sid)
    orchestrator.edit_analytical_table(tid, refresh_alignment=True)
    dto = orchestrator.query.get_analytical_table(tid)
    assert dto.table.rows[0].cells[-1].reason == "reference_selection_required"
    assert dto.conflicts[0].startswith("reference_selection_required")


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
    "mutation", ["fit", "settings", "rename", "remove_series", "new", "replace", "append"]
)
def test_outdated_jobs_are_rejected_without_adding_undo_entry(orchestrator, mutation):
    sid = create(orchestrator)
    job = orchestrator.prepare_post_processing(sid)
    result = match_series(job.matching)
    if mutation == "fit":
        orchestrator.update_parameter("p1", "amp", "value", 5)
    elif mutation == "settings":
        orchestrator.edit_post_processing_series(sid, settings=MatchingSettings(energy_gate=3))
    elif mutation == "rename":
        orchestrator.edit_post_processing_series(sid, name="Changed")
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
    with pytest.raises(ValueError, match="Several named or selected anchors merged"):
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
    path.write_text('{"version": "1.2", "objects": [], "post_processing": {}}')
    with pytest.raises(ValueError, match="Expected a JSON array"):
        orchestrator.load_collection(path)
    assert orchestrator.query.get_post_processing_series(sid) == before
    assert orchestrator.peek_undo_command() is previous
    assert orchestrator.is_dirty


def test_synchronous_run_and_metric_editor_are_undoable(orchestrator):
    sid = create(orchestrator)
    orchestrator.run_post_processing(sid)
    tid = orchestrator.create_analytical_table("Analysis", ((sid, "track/p1"),))
    state = orchestrator.query.get_analytical_table(tid).state
    metric = state.metrics[0]
    orchestrator.save_analytical_metric(
        tid, name="Area", operation="area", tracks=(state.sources[0].id_,), metric_id=metric.id_
    )
    assert [r.cells[-1].value for r in orchestrator.query.get_analytical_table(tid).table.rows] == [
        1,
        2,
    ]
    orchestrator.remove_analytical_metric(tid, metric.id_)
    assert orchestrator.query.get_analytical_table(tid).state.metrics == ()
    orchestrator.undo()
    assert orchestrator.query.get_analytical_table(tid).state.metrics[0].name == "Area"
    orchestrator.undo()
    assert orchestrator.query.get_analytical_table(tid).state == state


def test_export_uses_query_values_and_describes_operands_and_units(orchestrator, tmp_path):
    import csv
    import json

    sid = create(orchestrator)
    orchestrator.run_post_processing(sid)
    tid = orchestrator.create_analytical_table("Analysis", ((sid, "track/p1"),))
    state = orchestrator.query.get_analytical_table(tid).state
    orchestrator.save_analytical_metric(
        tid,
        name="Shift",
        operation="position_shift",
        tracks=(state.sources[0].id_,),
        reference_row_id=state.rows[1].id_,
    )
    metric = orchestrator.query.get_analytical_table(tid).state.metrics[-1]
    path = tmp_path / "analysis.csv"
    orchestrator.export_analytical_table(tid, path, precision=4)
    rows = list(csv.DictReader(path.open()))
    assert rows[0][f"{metric.id_} [eV]"] == "-0.2000"
    metadata = json.loads(path.with_suffix(".calculations.json").read_text())
    assert metadata["definitions"][-1]["reference_row_id"] == state.rows[1].id_
    assert (
        metadata["source_series"][sid]["result"]["series"]["steps"][1]["peaks"][0]["position"]
        == 0.2
    )


def test_stale_or_conflicting_exports_do_not_write_files(orchestrator, tmp_path):
    sid = create(orchestrator)
    orchestrator.run_post_processing(sid)
    tid = orchestrator.create_analytical_table("Analysis", ((sid, "track/p1"),))
    orchestrator.update_parameter("p1", "amp", "value", 10)
    path = tmp_path / "no.csv"
    with pytest.raises(ValueError, match="resolve source conflicts"):
        orchestrator.export_analytical_table(tid, path)
    assert not path.exists()
    assert not path.with_suffix(".calculations.json").exists()
    orchestrator.undo()
    orchestrator.remove_post_processing_series(sid)
    with pytest.raises(ValueError, match="resolve source conflicts"):
        orchestrator.export_analytical_table(tid, path)
    assert not path.exists()


def test_unknown_formula_is_rejected_without_mutating_saved_columns(orchestrator):
    sid = create(orchestrator)
    orchestrator.run_post_processing(sid)
    tid = orchestrator.create_analytical_table("Analysis", ((sid, "track/p1"),))
    before = orchestrator.query.get_analytical_table(tid)
    with pytest.raises(ValueError, match="Unknown metric operation"):
        orchestrator.save_analytical_metric(
            tid, name="Invalid", operation="eval", tracks=(before.state.sources[0].id_,)
        )
    assert orchestrator.query.get_analytical_table(tid) == before


def test_navigation_uses_live_ancestry_and_does_not_select_another_spectrums_peak(orchestrator):
    sid = create(orchestrator)
    orchestrator.run_post_processing(sid)
    assert orchestrator.query.get_post_processing_source(sid, "s1", component_id="p2") == (
        "s1",
        None,
        None,
    )
    orchestrator.full_remove_object("p1")
    assert orchestrator.query.get_post_processing_source(sid, "s1", "track/p1") == (
        "s1",
        None,
        None,
    )
    orchestrator.full_remove_object("s1")
    assert orchestrator.query.get_post_processing_source(sid, "s1", "track/p1") is None
