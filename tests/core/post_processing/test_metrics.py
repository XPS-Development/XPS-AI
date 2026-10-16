from dataclasses import replace

import pytest

from core.post_processing import MatchingSeries, PeakDescriptor, SeriesStep, match_series
from core.post_processing.metrics import MetricDefinition, calculate_metrics


def peak(cid, position, area):
    return PeakDescriptor(cid, "r", "pseudo-voigt", position, area, fwhm=1.0, sig=0.5)


@pytest.fixture
def result():
    return match_series(
        MatchingSeries(
            "series",
            (
                SeriesStep("s0", (peak("a", 280, 10), peak("b", 290, 30))),
                SeriesStep("s1", (peak("c", 281, 20), peak("d", 292, 20))),
                SeriesStep("empty"),
                SeriesStep("s3", (peak("e", 281, 10),)),
            ),
        )
    )


def values(result, operation, tracks=("track/a",), **kwargs):
    definition = MetricDefinition("m", "Metric", operation, tracks, **kwargs)
    return definition, calculate_metrics(result, (definition,))


@pytest.mark.parametrize(
    ("operation", "tracks", "kwargs", "expected", "unit"),
    [
        ("position", ("track/a",), {}, [280, 281, None, None], "eV"),
        ("area", ("track/a",), {}, [10, 20, None, None], "area"),
        ("fwhm", ("track/a",), {}, [1, 1, None, None], "eV"),
        ("sig", ("track/a",), {}, [0.5, 0.5, None, None], "eV"),
        ("position_difference", ("track/a", "track/b"), {}, [-10, -11, None, None], "eV"),
        ("position_difference", ("track/b", "track/a"), {}, [10, 11, None, None], "eV"),
        (
            "position_shift",
            ("track/a",),
            {"reference_spectrum_id": "s1"},
            [-1, 0, None, None],
            "eV",
        ),
        ("area_sum", ("track/a", "track/b", "track/a"), {}, [40, 40, None, None], "area"),
        ("area_ratio", ("track/a", "track/b"), {}, [1 / 3, 1, None, None], "1"),
        (
            "area_sum_ratio",
            ("track/a", "track/b"),
            {"denominator": ("track/b", "track/b")},
            [4 / 3, 2, None, None],
            "1",
        ),
        (
            "area_fraction",
            ("track/a",),
            {"denominator": ("track/a", "track/b", "track/a")},
            [25, 50, None, None],
            "%",
        ),
        ("spectrum_area_fraction", ("track/a",), {}, [25, 50, None, None], "%"),
    ],
)
def test_operations_units_and_gaps(result, operation, tracks, kwargs, expected, unit):
    definition, cells = values(result, operation, tracks, **kwargs)
    assert definition.unit == unit
    assert [cell.spectrum_id for cell in cells] == ["s0", "s1", "empty", "s3"]
    for cell, value in zip(cells, expected, strict=True):
        if value is None:
            assert cell.value is None
            assert cell.status == "missing"
            assert cell.reason == "missing_peak"
        else:
            assert cell.value == pytest.approx(value)
            assert cell.status == "ok"


def test_no_partial_sum_or_missing_reference_substitution(result):
    _, cells = values(result, "area_sum", ("track/a", "track/e"))
    assert all(cell.value is None and cell.reason == "missing_peak" for cell in cells)
    _, cells = values(result, "position_shift", reference_spectrum_id="empty")
    assert cells[0].value is None
    assert cells[0].reason == "missing_peak"
    _, cells = values(result, "position_shift", reference_spectrum_id="deleted")
    assert cells[0].reason == "reference_selection_required"
    _, cells = values(result, "area", ("track/deleted",))
    assert cells[0].reason == "operand_selection_required"


def test_zero_denominator_and_missing_width():
    result = match_series(
        MatchingSeries("s", (SeriesStep("one", (PeakDescriptor("a", "r", "asym", 284, 0),)),))
    )
    for operation in ("spectrum_area_fraction", "area_ratio"):
        tracks = ("track/a", "track/a") if operation == "area_ratio" else ("track/a",)
        _, cells = values(result, operation, tracks)
        assert cells[0].value is None
        assert cells[0].reason == "zero_denominator"
    _, cells = values(result, "fwhm")
    assert cells[0].reason == "missing_feature"


def test_ambiguity_propagates_and_definitions_are_independent_of_values():
    result = match_series(
        MatchingSeries(
            "s",
            (
                SeriesStep("one", (peak("a", 284, 10), peak("b", 284, 10))),
                SeriesStep("two", (peak("c", 284, 10), peak("d", 284, 10))),
            ),
        )
    )
    definition, cells = values(result, "position_shift", reference_spectrum_id="two")
    assert all(cell.status == "ambiguous" for cell in cells)
    assert definition.reference_spectrum_id == "two"
    _, cells = values(result, "area_sum", ("track/a", "track/b"))
    assert all(cell.status == "ambiguous" and cell.value == 20 for cell in cells)


def test_nonfinite_calculation_and_unavailable_step():
    result = match_series(
        MatchingSeries(
            "s",
            (
                SeriesStep("one", (peak("a", 284, 1e308),)),
                SeriesStep("gone", available=False),
            ),
        )
    )
    # Sum of finite source values can overflow; it must not leak infinity.
    _, cells = values(result, "area_sum_ratio", denominator=("track/a",))
    assert cells[1].reason == "unavailable_spectrum"
    huge = replace(
        result,
        series=replace(
            result.series,
            steps=(SeriesStep("one", (peak("a", 284, 1e308), peak("b", 285, 1e308))),),
        ),
    )
    _, cells = values(huge, "spectrum_area_fraction")
    assert cells[0].reason == "nonfinite_value"
    assert cells[0].value is None


@pytest.mark.parametrize(
    ("operation", "tracks", "kwargs"),
    [
        ("position_difference", ("a",), {}),
        ("position_shift", ("a",), {}),
        ("area_sum", (), {}),
        ("area_sum_ratio", ("a",), {}),
        ("area_fraction", ("a",), {"denominator": ("b",)}),
        ("area", ("a",), {"reference_spectrum_id": "s"}),
        ("area", ("a",), {"denominator": ("a",)}),
    ],
)
def test_invalid_definition(operation, tracks, kwargs):
    with pytest.raises(ValueError, match=r"requires|require|must belong|accepts|accept"):
        MetricDefinition("m", "Metric", operation, tracks, **kwargs)
