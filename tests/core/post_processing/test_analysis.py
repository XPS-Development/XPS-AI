from dataclasses import replace

import pytest

from core.post_processing import MatchingSeries, PeakDescriptor, SeriesStep, match_series
from core.post_processing.analysis import (
    AnalyticalTableState,
    TableMetric,
    TrendSource,
    align_rows,
    build_analysis,
    dependency_stamp,
)
from core.post_processing.state import PostProcessingSeries


def series(sid, positions, areas):
    steps = tuple(
        SeriesStep(
            f"{sid}{i}",
            tuple(
                PeakDescriptor(f"{sid}{j}-{i}", f"r{sid}{i}", "pseudo-voigt", pos, area, 2, 1)
                for j, (pos, area) in enumerate(zip(p, a, strict=True))
            ),
        )
        for i, (p, a) in enumerate(zip(positions, areas, strict=True))
    )
    result = match_series(MatchingSeries(sid, steps))
    return PostProcessingSeries(
        sid, sid, tuple(s.spectrum_id for s in steps), result=result, input_fingerprint="fp"
    )


@pytest.fixture
def data():
    ag = series("ag", [[368], [368.2], [368.4]], [[100], [200], [300]])
    c = series("c", [[284, 290], [284.1, 290.1], [284.2]], [[10, 30], [20, 20], [30]])
    sources = (
        TrendSource("Ag", "ag", "track/ag0-0", "ag0-0"),
        TrendSource("C-A", "c", "track/c0-0", "c0-0"),
        TrendSource("C-B", "c", "track/c1-0", "c1-0"),
    )
    saved = {"ag": ag, "c": c}
    rows = align_rows(sources, saved)
    return saved, AnalyticalTableState("analysis", "Ag / C", sources, rows)


def cells(data, op, operands, denominator=(), reference=None):
    saved, table = data
    metric = TableMetric("value", "Calculation", op, operands, denominator, reference)
    table = replace(table, metrics=(metric,))
    projection = build_analysis(table, saved, {})
    return metric, [row.cells[-1] for row in projection.rows]


@pytest.mark.parametrize(
    ("op", "operands", "denominator", "expected", "unit"),
    [
        ("position", ("Ag",), (), [368, 368.2, 368.4], "eV"),
        ("sig", ("Ag",), (), [1, 1, 1], "eV"),
        ("position_difference", ("Ag", "C-A"), (), [84, 84.1, 84.2], "eV"),
        ("position_difference", ("C-A", "Ag"), (), [-84, -84.1, -84.2], "eV"),
        ("area_sum", ("C-A", "C-B", "C-A"), (), [40, 40, None], "area"),
        ("area_ratio", ("Ag", "C-A"), (), [10, 10, 10], "1"),
        ("area_sum_ratio", ("Ag",), ("C-A", "C-B", "C-A"), [2.5, 5, None], "1"),
        ("area_fraction", ("C-A",), ("C-A", "C-B"), [25, 50, None], "%"),
        ("spectrum_area_fraction", ("C-A",), (), [25, 50, 100], "%"),
    ],
)
def test_cross_series_operations_and_incomplete_sums(
    data, op, operands, denominator, expected, unit
):
    metric, values = cells(data, op, operands, denominator)
    assert metric.unit == unit
    for value, expected_value in zip(values, expected, strict=True):
        if expected_value is None:
            assert value.value is None
            assert value.reason == "missing_peak"
        else:
            assert value.value == pytest.approx(expected_value)
    if op == "position_difference":
        assert {s.series_id for s in values[0].sources} == {"ag", "c"}


def test_reference_uses_explicit_row_id_and_missing_reference_is_not_replaced(data):
    reference = data[1].rows[1].id_
    _, values = cells(data, "position_shift", ("Ag",), reference=reference)
    assert [v.value for v in values] == pytest.approx([-0.2, 0, 0.2])
    _, values = cells(data, "position_shift", ("Ag",), reference="gone")
    assert all(v.reason == "reference_selection_required" for v in values)


def test_alignment_keeps_leading_internal_and_tail_gaps():
    a = series("a", [[368], [368.1]], [[10], [20]])
    b = series("b", [[], [284], [284.1], []], [[], [30], [40], []])
    sources = (
        TrendSource("A", "a", "track/a0-0", "a0-0"),
        TrendSource("B", "b", "track/b0-1", "b0-1"),
    )
    saved = {"a": a, "b": b}
    rows = align_rows(sources, saved)
    assert len(rows) == 4
    assert rows[0].points[1].spectrum_id == "b0"
    assert rows[0].points[1].component_id is None
    assert rows[1].points[1].component_id == "b0-1"
    assert rows[2].points[0].spectrum_id is None
    assert rows[2].points[1].component_id == "b0-2"
    assert rows[3].points[1].spectrum_id == "b3"
    assert align_rows(sources, saved, rows) == rows
    reordered = replace(a, spectrum_ids=tuple(reversed(a.spectrum_ids)))
    changed = align_rows(sources[:1], {"a": reordered}, rows)
    assert changed[0].id_ == rows[1].id_


def test_unavailable_sources_do_not_shift_saved_rows_or_invent_values(data):
    saved, table = data
    table = replace(
        table, metrics=(TableMetric("delta", "Delta", "position_difference", ("Ag", "C-A")),)
    )
    removed = build_analysis(table, {"ag": saved["ag"]}, {})
    assert all(row.cells[-1].reason == "unavailable_series" for row in removed.rows)
    available = {sid for series in saved.values() for sid in series.spectrum_ids} - {"c1"}
    removed_point = build_analysis(table, saved, {}, available)
    assert removed_point.rows[1].cells[-1].reason == "unavailable_spectrum"
    assert removed_point.rows[2].cells[-1].value == pytest.approx(84.2)
    assert [r.id_ for r in removed_point.rows] == [r.id_ for r in table.rows]


def test_zero_denominator_nonfinite_and_ambiguity(data):
    saved, table = data
    c = saved["c"]
    first_step = c.result.series.steps[0]
    changed = replace(first_step, peaks=tuple(replace(p, area=0) for p in first_step.peaks))
    result = replace(
        c.result, series=replace(c.result.series, steps=(changed, *c.result.series.steps[1:]))
    )
    modified = {**saved, "c": replace(c, result=result)}
    _, values = cells((modified, table), "area_sum_ratio", ("Ag",), ("C-A", "C-B"))
    assert values[0].reason == "zero_denominator"
    ag = saved["ag"]
    track = ag.result.tracks[0]
    ambiguous = replace(track, links=(replace(track.links[0], ambiguous=True), *track.links[1:]))
    modified = {**saved, "ag": replace(ag, result=replace(ag.result, tracks=(ambiguous,)))}
    _, values = cells((modified, table), "position_difference", ("Ag", "C-A"))
    assert all(v.status == "ambiguous" for v in values)
    assert dependency_stamp(
        replace(ag, name="Renamed", track_names=(("ag0-0", "Silver"),))
    ) == dependency_stamp(ag)


def test_position_and_sig_project_distinct_quantities_with_the_same_unit(data):
    saved, table = data
    table = replace(
        table,
        metrics=(
            TableMetric("energy", "Custom energy", "position", ("Ag",)),
            TableMetric("width", "Custom width", "sig", ("Ag",)),
        ),
    )
    columns = build_analysis(table, saved, {}).columns[-2:]
    assert [column.quantity for column in columns] == ["Position", "sig"]
    assert [column.unit for column in columns] == ["eV", "eV"]
