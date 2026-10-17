from typing import cast

import pytest

from core.post_processing import (
    ExcludedComponent,
    MatchingSeries,
    PeakDescriptor,
    SeriesStep,
    match_series,
)
from core.post_processing.state import PostProcessingSeries
from core.post_processing.tables import TableView, build_table


def saved():
    result = match_series(
        MatchingSeries(
            "series",
            (
                SeriesStep(
                    "s1",
                    (
                        PeakDescriptor("a", "r1", "pseudo-voigt", 280, 1, 2, 1),
                        PeakDescriptor("b", "r2", "asym", 290, 3, sig=1),
                    ),
                ),
                SeriesStep("empty"),
                SeriesStep(
                    "bad",
                    excluded=(
                        ExcludedComponent("invalid", "r3", "invalid_sig"),
                        ExcludedComponent("bg", "r3", "background"),
                    ),
                ),
                SeriesStep("deleted", available=False),
            ),
        )
    )
    return PostProcessingSeries(
        "series",
        "Series",
        ("s1", "empty", "bad", "deleted"),
        result=result,
        input_fingerprint="fingerprint",
        track_names=(("a", "A"),),
    )


def test_detailed_rows_preserve_provenance_widths_and_exclusion_reasons():
    table = build_table(saved(), {"s1": "Spectrum"})
    assert [(row.spectrum_id, row.component_id) for row in table.rows] == [
        ("s1", "a"),
        ("s1", "b"),
        ("empty", None),
        ("bad", "invalid"),
        ("deleted", None),
    ]
    assert all(len(row.cells) == len(table.columns) for row in table.rows)
    columns = {column.id_: index for index, column in enumerate(table.columns)}
    assert table.rows[0].region_id == "r1"
    assert table.rows[0].cells[columns["spectrum"]].value == "Spectrum"
    assert table.rows[0].cells[columns["track_name"]].value == "A"
    assert table.rows[0].cells[columns["fraction"]].value == 25
    assert "fwhm" not in columns
    assert table.columns[columns["fraction"]].label == "Rel. area"
    assert table.rows[1].cells[columns["sig"]].value == 1
    assert table.rows[2].cells[-1].reason == "empty_step"
    assert table.rows[3].cells[-1].reason == "invalid_sig"
    assert table.rows[4].cells[-1].reason == "unavailable_spectrum"


def test_selected_trend_retains_every_step_including_empty_and_unavailable():
    from core.post_processing.tables import build_trend_tables

    detailed, preview = build_trend_tables(saved(), {}, "track/a")
    assert len(detailed.rows) == 4
    assert [r.spectrum_id for r in detailed.rows] == ["s1", "empty", "bad", "deleted"]
    assert all(len(r.cells) == len(preview.columns) for r in preview.rows)
    assert preview.rows[0].cells[2].value == 280
    assert preview.rows[1].cells[2].reason == "missing_peak"


def test_no_snapshot_and_invalid_view():
    state = PostProcessingSeries("series", "S", ("s1",))
    assert build_table(state, {}).rows == ()
    with pytest.raises(ValueError, match="Unknown"):
        build_table(state, {}, cast(TableView, "invalid"))
