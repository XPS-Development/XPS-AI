"""Tests for core.fitting.expressions."""

from core.fitting.expressions import (
    match_component_reference,
    parse_parameter_expression,
    resolve_component_reference,
)


class TestMatchComponentReference:
    def test_exact_and_prefix(self):
        ids = ("pfxAAA00xx", "pfxBBB00xx")
        exact = match_component_reference("pfxAAA00xx", ids)
        assert exact.status == "exact"
        assert exact.component_id == "pfxAAA00xx"

        prefix = match_component_reference("pfxAA", ids)
        assert prefix.status == "prefix"
        assert prefix.component_id == "pfxAAA00xx"

    def test_ambiguous_and_unknown(self):
        ids = ("pfxAAA00xx", "pfxBBB00xx")
        amb = match_component_reference("pfx", ids)
        assert amb.status == "ambiguous"
        assert amb.component_id is None
        assert set(amb.candidates) == set(ids)

        unknown = match_component_reference("zzz", ids)
        assert unknown.status == "unknown"
        assert resolve_component_reference("zzz", ids) is None


class TestParseParameterExpression:
    def test_simple_component_ref(self):
        parsed = parse_parameter_expression(
            "2 * peakAA",
            owner_component_id="peakBB",
            parameter_name="amp",
            known_component_ids=("peakAA", "peakBB"),
            parameter_names_by_component={
                "peakAA": {"amp", "cen"},
                "peakBB": {"amp", "cen"},
            },
        )
        assert parsed.issues == ()
        assert parsed.lmfit_expr == "2 * peakAA_amp"
        assert len(parsed.references) == 1
        assert parsed.references[0].component_id == "peakAA"
        assert parsed.references[0].parameter_name == "amp"

    def test_scientific_notation_does_not_fail(self):
        parsed = parse_parameter_expression(
            "peakAA * 1.5e3",
            owner_component_id="peakBB",
            parameter_name="amp",
            known_component_ids=("peakAA", "peakBB"),
            parameter_names_by_component={
                "peakAA": {"amp"},
                "peakBB": {"amp"},
            },
        )
        assert parsed.issues == ()
        assert parsed.lmfit_expr == "peakAA_amp * 1.5e3"

    def test_negative_exponent_and_float(self):
        parsed = parse_parameter_expression(
            "peakAA * 1e-3 + 0.5",
            owner_component_id="peakBB",
            parameter_name="amp",
            known_component_ids=("peakAA",),
            parameter_names_by_component={"peakAA": {"amp"}},
        )
        assert parsed.issues == ()
        assert parsed.lmfit_expr == "peakAA_amp * 1e-3 + 0.5"

    def test_lmfit_builtin_functions_are_kept(self):
        parsed = parse_parameter_expression(
            "sqrt(peakAA) + exp(0)",
            owner_component_id="peakBB",
            parameter_name="amp",
            known_component_ids=("peakAA",),
            parameter_names_by_component={"peakAA": {"amp"}},
        )
        assert parsed.issues == ()
        assert parsed.lmfit_expr == "sqrt(peakAA_amp) + exp(0)"

    def test_unknown_token_reports_issue(self):
        parsed = parse_parameter_expression(
            "2 * missingComp",
            owner_component_id="peakBB",
            parameter_name="amp",
            known_component_ids=("peakBB",),
        )
        assert parsed.lmfit_expr is None
        assert len(parsed.issues) == 1
        assert parsed.issues[0].code == "unknown_component"
        assert parsed.issues[0].token == "missingComp"

    def test_ambiguous_token_reports_issue(self):
        parsed = parse_parameter_expression(
            "pfx * 2",
            owner_component_id="other",
            parameter_name="amp",
            known_component_ids=("pfxAAA", "pfxBBB"),
            parameter_names_by_component={
                "pfxAAA": {"amp"},
                "pfxBBB": {"amp"},
            },
        )
        assert parsed.lmfit_expr is None
        assert parsed.issues[0].code == "ambiguous_component"

    def test_missing_parameter_reports_issue(self):
        parsed = parse_parameter_expression(
            "bg1",
            owner_component_id="peak1",
            parameter_name="amp",
            known_component_ids=("bg1", "peak1"),
            parameter_names_by_component={
                "bg1": {"const"},
                "peak1": {"amp", "cen"},
            },
        )
        assert parsed.lmfit_expr is None
        assert parsed.issues[0].code == "missing_parameter"
        assert "amp" in parsed.issues[0].message

    def test_prefix_resolution_in_lmfit_expr(self):
        parsed = parse_parameter_expression(
            "2 * peakAA",
            owner_component_id="peakBB00xx",
            parameter_name="amp",
            known_component_ids=("peakAA00xx", "peakBB00xx"),
            parameter_names_by_component={
                "peakAA00xx": {"amp"},
                "peakBB00xx": {"amp"},
            },
        )
        assert parsed.issues == ()
        assert parsed.lmfit_expr == "2 * peakAA00xx_amp"

    def test_builtin_does_not_mask_real_component(self):
        """A uniquely matching component id wins over the builtin name set."""
        parsed = parse_parameter_expression(
            "sqrt",
            owner_component_id="other",
            parameter_name="amp",
            known_component_ids=("sqrt", "other"),
            parameter_names_by_component={"sqrt": {"amp"}, "other": {"amp"}},
        )
        assert parsed.issues == ()
        assert parsed.lmfit_expr == "sqrt_amp"
