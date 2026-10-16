import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core.post_processing import (
    LinkConstraints,
    MatchingSeries,
    MatchingSettings,
    PeakDescriptor,
    SeriesStep,
    match_series,
)


def peak(cid, position=284.0, area=10.0, region="r"):
    return PeakDescriptor(cid, region, "pseudo-voigt", position, area, 1.2, 0.6, 0.5)


def series(*steps, settings=None, constraints=None):
    return MatchingSeries(
        "series",
        tuple(SeriesStep(f"s{i}", tuple(peaks)) for i, peaks in enumerate(steps)),
        settings or MatchingSettings(),
        constraints or LinkConstraints(),
    )


CASES = json.loads(Path(__file__).with_name("regression.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["input"]["id"])
def test_original_continuous_algorithm_regression(case):
    steps = tuple(
        SeriesStep(
            str(step["step_index"]),
            tuple(
                peak(c["id"], c["parameters"]["Position"], c["parameters"]["Area"])
                for c in step["components"]
            ),
        )
        for step in case["input"]["steps"]
    )
    settings = (
        MatchingSettings(start_cost=0.01, end_cost=0.01)
        if case["low_boundary_cost"]
        else MatchingSettings()
    )
    result = match_series(MatchingSeries("regression", steps, settings))
    expected = case["expected"]
    assert result.objective == pytest.approx(expected["objective"])
    assert result.candidate_links == expected["candidate_links"]
    assert [t.component_ids for t in result.tracks] == [
        tuple(p["component_id"] for p in t["points"]) for t in expected["tracks"]
    ]
    for actual, original in zip(result.tracks, expected["tracks"], strict=True):
        assert actual.id_ == original["id"]
        for link, old in zip(actual.links, original["links"], strict=True):
            assert (link.source_id, link.target_id) == (
                old["from_component_id"],
                old["to_component_id"],
            )
            assert link.cost == pytest.approx(old["cost"])
            assert link.cost_margin == pytest.approx(old["cost_margin"])
            assert link.ambiguous == old["ambiguous"]


def test_regions_are_provenance_not_match_gates():
    result = match_series(series([peak("a", region="old")], [peak("b", region="new")]))
    assert result.tracks[0].component_ids == ("a", "b")


def test_features_use_all_regions_and_tied_energy_ranks():
    result = match_series(
        series(
            [peak("a", 280, 10, "r1"), peak("b", 282, 30, "r2")],
            [peak("c", 280, 30, "r3"), peak("d", 282, 10, "r4")],
        )
    )
    first = dict(result.tracks[0].links[0].terms)
    assert first["area_fraction"] == pytest.approx(0.1 * (0.5 - (0.75**0.5)) ** 2)
    assert first["energy_rank"] == 0


def test_manual_links_bypass_gate_and_have_no_ambiguity():
    data = series(
        [peak("a", 280), peak("b", 281)],
        [peak("c", 290), peak("d", 281)],
        constraints=LinkConstraints(required=(("a", "c"),)),
    )
    result = match_series(data)
    assert [t.component_ids for t in result.tracks] == [("a", "c"), ("b", "d")]
    manual = result.tracks[0].links[0]
    assert manual.manual
    assert manual.cost_margin is None
    assert not manual.ambiguous
    assert not result.tracks[1].links[0].ambiguous


def test_forbidden_link_and_unavailable_step():
    result = match_series(
        series([peak("a")], [peak("b")], constraints=LinkConstraints(forbidden=(("a", "b"),)))
    )
    assert len(result.tracks) == 2
    result = match_series(
        MatchingSeries(
            "s",
            (
                SeriesStep("s0", (peak("a"),)),
                SeriesStep("s1", available=False),
                SeriesStep("s2", (peak("b"),)),
            ),
        )
    )
    assert len(result.tracks) == 2
    assert len(result.series.steps) == 3


@pytest.mark.parametrize(
    ("constraints", "message"),
    [
        (LinkConstraints(required=(("a", "c"), ("a", "d"))), "multiple successors"),
        (LinkConstraints(required=(("a", "c"), ("b", "c"))), "multiple predecessors"),
        (LinkConstraints(required=(("a", "c"),), forbidden=(("a", "c"),)), "also be forbidden"),
        (LinkConstraints(required=(("missing", "c"),)), "unavailable peak"),
        (LinkConstraints(required=(("a", "b"),)), "adjacent"),
    ],
)
def test_conflicting_constraints_are_rejected(constraints, message):
    with pytest.raises(ValueError, match=message):
        match_series(
            series([peak("a"), peak("b")], [peak("c"), peak("d")], constraints=constraints)
        )


def test_missing_width_feature_with_enabled_weights():
    result = match_series(
        series(
            [PeakDescriptor("a", "r1", "asym", 284)],
            [peak("b")],
            settings=MatchingSettings(width_weight=1, shape_weight=1),
        )
    )
    assert result.tracks[0].links[0].missing_features == ("fwhm", "shape_gl")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"energy_scale": 0},
        {"area_weight": -1},
        {"width_weight": float("nan")},
        {"energy_gate": float("inf")},
        {"end_cost": True},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError, match="must be"):
        MatchingSettings(**kwargs)


def test_immutable_unique_inputs():
    data = series([peak("a")])
    with pytest.raises(FrozenInstanceError):
        data.id_ = "changed"
    with pytest.raises(ValueError, match="Component IDs"):
        series([peak("a")], [peak("a")])
    with pytest.raises(ValueError, match="Spectrum IDs"):
        MatchingSeries("s", (SeriesStep("s0"), SeriesStep("s0")))
