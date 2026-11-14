import copy
import json

import numpy as np
import pytest

from app.orchestration import AppOrchestrator
from app.parameters import AppParameters
from core.collection import CoreCollection
from core.io.serialization import deserialize, dump, load, serialize
from core.math_models import PseudoVoigtPeakModel
from core.objects import Peak, Region, Spectrum


@pytest.fixture
def app():
    collection = CoreCollection()
    x = np.linspace(270, 390, 121)
    for prefix, centers in (("ag", (368,)), ("c", (284, 290))):
        for index in range(3):
            sid = f"{prefix}{index}"
            collection.add(Spectrum(x, x - 250, id_=sid))
            collection.add(Region(slice(0, len(x)), parent_id=sid, id_=f"r{sid}"))
            for j, center in enumerate(centers):
                if prefix == "c" and index == 2 and j == 1:
                    continue
                collection.add(
                    Peak(
                        model=PseudoVoigtPeakModel(),
                        region_id=f"r{sid}",
                        component_id=f"p{prefix}{j}-{index}",
                        amp=(j + 1) * (index + 1) * 10,
                        cen=center + index * 0.1,
                        sig=1,
                        frac=0.5,
                    )
                )
    app = AppOrchestrator(collection, AppParameters(default_serialization_use_gzip=False))
    ag = app.create_post_processing_series("Silver", ["ag0", "ag1", "ag2"])
    c = app.create_post_processing_series("Carbon", ["c0", "c1", "c2"])
    app.run_post_processing(ag)
    app.run_post_processing(c)
    tid = app.create_analytical_table(
        "Ag / C", ((ag, "track/pag0-0"), (c, "track/pc0-0"), (c, "track/pc1-0"))
    )
    return app, ag, c, tid


def test_three_trends_cross_series_and_reusable_sources(app):
    app, ag, c, tid = app
    state = app.query.get_analytical_table(tid).state
    ag_source, ca, cb = (s.id_ for s in state.sources)
    app.save_analytical_metric(tid, name="C total", operation="area_sum", tracks=(ca, cb))
    app.save_analytical_metric(
        tid,
        name="Ag / total C",
        operation="area_sum_ratio",
        tracks=(ag_source,),
        denominator=(ca, cb),
    )
    app.save_analytical_metric(
        tid, name="Delta E", operation="position_difference", tracks=(ag_source, ca)
    )
    dto = app.query.get_analytical_table(tid)
    assert dto.exportable
    assert dto.warnings
    assert [r.cells[-3].value for r in dto.table.rows] == [30, 60, None]
    assert [r.cells[-2].value for r in dto.table.rows] == [1 / 3, 1 / 3, None]
    assert all(r.cells[-1].value == pytest.approx(84) for r in dto.table.rows)
    assert dto.table.rows[2].cells[-2].reason == "missing_peak"
    assert len(dto.table.rows[0].cells[-2].sources) == 3
    other = app.create_analytical_table("Reuse", ((ag, "track/pag0-0"),))
    app.remove_analytical_table(other)
    assert app.query.get_post_processing_series(ag).exportable
    assert app.query.get_post_processing_series(c).exportable


def test_source_edits_rename_and_plots_are_undoable_without_changing_series(app):
    app, ag, c, tid = app
    before = app.query.get_analytical_table(tid)
    source_snapshots = (
        app.query.get_post_processing_series(ag),
        app.query.get_post_processing_series(c),
    )
    app.edit_analytical_table(
        tid, plotted_columns=(), column_order=tuple(reversed(before.state.column_order))
    )
    assert app.query.get_analytical_table(tid).state.plotted_columns == ()
    app.undo()
    assert app.query.get_analytical_table(tid) == before
    app.edit_analytical_table(tid, selections=((ag, "track/pag0-0"),))
    assert len(app.query.get_analytical_table(tid).state.sources) == 1
    assert app.query.get_analytical_table(tid).conflicts
    app.undo()
    assert app.query.get_analytical_table(tid) == before
    app.name_post_processing_track(ag, "track/pag0-0", "Ag3d 5/2")
    assert not app.query.get_analytical_table(tid).stale
    assert "Ag3d 5/2" in app.query.get_analytical_table(tid).table.columns[2].label
    assert source_snapshots[1] == app.query.get_post_processing_series(c)


def test_series_order_invalidates_alignment_and_requires_explicit_refresh(app):
    app, ag, _, tid = app
    before = app.query.get_analytical_table(tid)
    app.edit_post_processing_series(ag, spectrum_ids=("ag2", "ag1", "ag0"))
    assert app.query.get_analytical_table(tid).stale
    assert app.query.get_analytical_table(tid).state.rows == before.state.rows
    app.run_post_processing(ag)
    assert app.query.get_analytical_table(tid).stale
    app.edit_analytical_table(tid, refresh_alignment=True)
    dto = app.query.get_analytical_table(tid)
    assert not dto.stale
    assert dto.state.rows[0].points[0].spectrum_id == "ag2"
    app.undo()
    assert app.query.get_analytical_table(tid).state.rows == before.state.rows
    assert app.query.get_analytical_table(tid).stale


def test_deleted_spectrum_and_series_leave_rows_and_table_in_place(app):
    app, ag, _, tid = app
    original = app.query.get_analytical_table(tid)
    app.full_remove_object("ag1")
    dto = app.query.get_analytical_table(tid)
    assert dto.state.rows == original.state.rows
    assert dto.table.rows[1].cells[-3].reason == "unavailable_spectrum"
    assert not dto.exportable
    app.undo()
    assert app.query.get_analytical_table(tid) == original
    app.remove_post_processing_series(ag)
    assert app.query.get_analytical_table_ids() == (tid,)
    dto = app.query.get_analytical_table(tid)
    assert dto.conflicts[0].startswith("unavailable_series")
    assert dto.state.rows == original.state.rows
    app.undo()
    assert app.query.get_analytical_table(tid) == original


def test_manual_link_edit_refreshes_metrics_with_single_undo_step(app):
    app, ag, _, tid = app
    original = app.query.get_post_processing_series(ag)
    app.set_post_processing_link(ag, "pag0-0", "pag0-1", False)
    dto = app.query.get_post_processing_series(ag)
    assert not dto.stale
    assert len(dto.series.result.tracks) == 2
    assert app.query.get_analytical_table(tid).stale
    app.undo()
    assert app.query.get_post_processing_series(ag) == original
    assert app.query.get_analytical_table(tid).exportable
    app.set_post_processing_link(ag, "pag0-0", "pag0-1", True)
    assert app.query.get_post_processing_series(ag).series.result.tracks[0].links[0].manual
    app.run_post_processing(ag)
    assert app.query.get_post_processing_series(ag).series.result.tracks[0].links[0].manual


@pytest.mark.parametrize("gzip", [False, True])
def test_full_document_round_trip_and_new_replace(app, tmp_path, gzip):
    app, _, _, tid = app
    before = app.query.get_analytical_table(tid)
    path = tmp_path / ("analysis.gz" if gzip else "analysis.json")
    dump(app.core_collection, path, use_gzip=gzip)
    loaded = load(path, mode="new")
    assert isinstance(loaded, tuple)
    assert loaded[0].analytical_tables[tid] == before.state
    app.new_collection()
    assert app.query.get_analytical_table_ids() == ()
    app.load_collection(path)
    assert app.query.get_analytical_table(tid) == before


def test_append_remaps_all_qualified_series_references_and_table_ids(app):
    app, ag, c, tid = app
    data = serialize(app.core_collection)
    original = app.query.get_analytical_table(tid)
    deserialize(data, app.core_collection, app.ctx.metadata, mode="append")
    assert len(app.query.get_analytical_table_ids()) == 2
    imported_id = next(i for i in app.query.get_analytical_table_ids() if i != tid)
    imported = app.query.get_analytical_table(imported_id)
    assert imported.exportable
    assert {s.series_id for s in imported.state.sources}.isdisjoint({ag, c})
    assert set(dict(imported.state.dependencies)) == {s.series_id for s in imported.state.sources}
    assert imported.state.rows == original.state.rows
    assert imported.state.metrics == original.state.metrics
    assert app.query.get_analytical_table(tid) == original


@pytest.mark.parametrize("broken", ["source", "row", "metric", "duplicate"])
def test_malformed_table_load_is_transactional(app, broken):
    app, _, _, tid = app
    before = app.query.get_analytical_table(tid)
    data = copy.deepcopy(serialize(app.core_collection))
    raw = data["analytical_tables"][0]
    if broken == "source":
        raw["sources"][0]["track_id"] = "track/wrong"
    elif broken == "row":
        raw["rows"][0]["points"] = []
    elif broken == "metric":
        raw["metrics"][0]["operation"] = "eval"
    else:
        data["analytical_tables"].append(raw)
    with pytest.raises(ValueError, match=r"Invalid|Each row|Unknown|Duplicate"):
        deserialize(data, app.core_collection, app.ctx.metadata, mode="replace")
    assert app.query.get_analytical_table(tid) == before


def test_no_migration_from_intermediate_format_and_release_compatibility(app):
    app, _, _, _ = app
    data = serialize(app.core_collection)
    data["version"] = "1.1"
    with pytest.raises(ValueError, match="Version mismatch"):
        deserialize(data, mode="new")
    data = {"version": "1.0", "objects": data["objects"]}
    loaded = deserialize(data, mode="new")
    assert isinstance(loaded, tuple)
    assert not loaded[0].post_processing
    assert not loaded[0].analytical_tables


def test_export_contains_every_original_source_and_reference_row(app, tmp_path):
    app, _, _, tid = app
    state = app.query.get_analytical_table(tid).state
    app.save_analytical_metric(
        tid,
        name="Shift",
        operation="position_shift",
        tracks=(state.sources[0].id_,),
        reference_row_id=state.rows[1].id_,
    )
    path = tmp_path / "analysis.csv"
    app.export_analytical_table(tid, path)
    description = json.loads(path.with_suffix(".calculations.json").read_text())
    assert len(description["source_series"]) == 2
    assert description["definitions"][-1]["reference_row_id"] == state.rows[1].id_
    assert description["analytical_table"]["rows"][0]["points"][0]["spectrum_id"] == "ag0"
    assert path.exists()
