"""Tests for core.fitting.fit_scope."""

from core.dto import ComponentDTO, ParameterDTO
from core.fitting.fit_scope import (
    collect_expression_problems,
    expand_fit_region_ids,
    format_expression_problems,
)
from core.math_models import PseudoVoigtPeakModel


def _param(name: str, value: float = 1.0, *, expr: str | None = None) -> ParameterDTO:
    return ParameterDTO(
        name=name,
        value=value,
        lower=0.0,
        upper=10.0,
        vary=True,
        expr=expr,
    )


def _peak(
    id_: str,
    region_id: str,
    *,
    amp_expr: str | None = None,
    cen_expr: str | None = None,
) -> ComponentDTO:
    return ComponentDTO(
        id_=id_,
        parent_id=region_id,
        normalized=False,
        parameters={
            "amp": _param("amp", 1.0, expr=amp_expr),
            "cen": _param("cen", 0.0, expr=cen_expr),
            "sig": _param("sig", 1.0),
            "frac": _param("frac", 0.0),
        },
        model=PseudoVoigtPeakModel(),
        kind="peak",
    )


class TestExpandFitRegionIds:
    def test_no_exprs_returns_selection(self):
        comps = (_peak("p1", "r1"), _peak("p2", "r2"))
        assert expand_fit_region_ids(["r1"], comps) == ("r1",)

    def test_cross_region_expr_pulls_dependency(self):
        comps = (
            _peak("peakAA", "r1"),
            _peak("peakBB", "r2", amp_expr="2 * peakAA"),
        )
        assert expand_fit_region_ids(["r2"], comps) == ("r2", "r1")
        assert expand_fit_region_ids(["r1"], comps) == ("r1", "r2")

    def test_transitive_chain(self):
        comps = (
            _peak("p1", "r1"),
            _peak("p2", "r2", amp_expr="p1"),
            _peak("p3", "r3", amp_expr="p2"),
        )
        assert expand_fit_region_ids(["r3"], comps) == ("r3", "r1", "r2")

    def test_same_region_expr_does_not_add_extras(self):
        comps = (
            _peak("peakAA", "r1"),
            _peak("peakBB", "r1", amp_expr="peakAA"),
            _peak("peakCC", "r2"),
        )
        assert expand_fit_region_ids(["r1"], comps) == ("r1",)

    def test_empty_selection(self):
        assert expand_fit_region_ids([], (_peak("p1", "r1"),)) == ()

    def test_invalid_expr_still_uses_resolved_refs(self):
        """Partial refs count for closure even when the whole expr is invalid."""
        comps = (
            _peak("peakAA", "r1"),
            _peak("peakBB", "r2", amp_expr="peakAA + missingThing"),
        )
        assert expand_fit_region_ids(["r2"], comps) == ("r2", "r1")


class TestCollectExpressionProblems:
    def test_unknown_component(self):
        comps = (_peak("peakBB", "r1", cen_expr="noSuchPeak"),)
        problems = collect_expression_problems(comps, region_ids=["r1"])
        assert len(problems) == 1
        assert problems[0].parameter_name == "cen"
        assert problems[0].issues[0].code == "unknown_component"
        assert "unknown component" in format_expression_problems(problems)

    def test_valid_expr_not_reported(self):
        comps = (
            _peak("peakAA", "r1"),
            _peak("peakBB", "r1", amp_expr="peakAA"),
        )
        assert collect_expression_problems(comps, region_ids=["r1"]) == ()
