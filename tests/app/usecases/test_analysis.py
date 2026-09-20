"""Tests for AnalysisUseCases Change construction."""

from typing import Any

import pytest

from app.command.changes import CompositeChange, CreatePeak
from app.nn_service import NNService
from app.optimization import OptimizationService
from app.orchestration import AppParameters, QueryService
from app.usecases.analysis import AnalysisUseCases
from core.math_models import PseudoVoigtPeakModel
from core.objects import Peak, Region, Spectrum
from core.services import CoreContext


def _analysis(collection, nn: NNService | None = None) -> AnalysisUseCases:
    ctx = CoreContext.from_collection(collection)
    return AnalysisUseCases(
        QueryService(ctx),
        nn if nn is not None else NNService(model_path=None),
        OptimizationService(),
        AppParameters(),
    )


def test_run_segmenter_skips_spectra_that_already_have_regions(
    simple_collection, spectrum_id
) -> None:
    """run_segmenter returns None when every spectrum already has regions."""

    class _BoomNN(NNService):
        def run_segmenter(self, *args: Any, **kwargs: Any) -> CompositeChange:
            raise AssertionError("segmenter should not run")

    change = _analysis(simple_collection, nn=_BoomNN(model_path=None)).run_segmenter([spectrum_id])
    assert change is None


def test_optimize_regions_requires_ids(simple_collection) -> None:
    """optimize_regions raises when neither region_ids nor spectrum_ids is given."""
    with pytest.raises(ValueError, match="region_ids or spectrum_ids"):
        _analysis(simple_collection).optimize_regions()


def test_preview_fit_scope_no_extras(simple_collection, region_id) -> None:
    """Without cross-region exprs, preview reports no extras."""
    preview = _analysis(simple_collection).preview_fit_scope(region_ids=[region_id])
    assert preview.selected_region_ids == (region_id,)
    assert preview.expanded_region_ids == (region_id,)
    assert not preview.needs_confirmation
    assert preview.extra_region_count == 0
    assert preview.extra_spectrum_count == 0
    assert not preview.has_expression_errors


def test_preview_reports_unknown_component_in_expr(empty_collection, simple_gauss_spectrum) -> None:
    """Unknown peak id in expr is reported on the fit-scope preview."""
    x, y = simple_gauss_spectrum
    collection = empty_collection
    s1 = Spectrum(x, y, id_="s1")
    r1 = Region(slice(20, len(x) - 20), parent_id=s1.id_, id_="r1")
    p1 = Peak(
        model=PseudoVoigtPeakModel(),
        region_id=r1.id_,
        component_id="peakBB",
        amp=1.0,
        cen=0.0,
        sig=1.0,
        frac=0.0,
    )
    p1.parameters["cen"].set(expr="pa3cf67e3aec942db988d3d93e635b014")
    for obj in (s1, r1, p1):
        collection.add(obj)

    preview = _analysis(collection).preview_fit_scope(region_ids=["r1"])
    assert preview.has_expression_errors
    assert preview.expression_problems[0].parameter_name == "cen"
    assert preview.expression_problems[0].issues[0].code == "unknown_component"


def test_optimize_regions_rejects_unknown_component_expr(
    empty_collection, simple_gauss_spectrum
) -> None:
    """optimize_regions fails loudly instead of silently dropping a bad expr."""
    x, y = simple_gauss_spectrum
    collection = empty_collection
    s1 = Spectrum(x, y, id_="s1")
    r1 = Region(slice(20, len(x) - 20), parent_id=s1.id_, id_="r1")
    p1 = Peak(
        model=PseudoVoigtPeakModel(),
        region_id=r1.id_,
        component_id="peakBB",
        amp=1.0,
        cen=0.0,
        sig=1.0,
        frac=0.0,
    )
    p1.parameters["cen"].set(expr="pa3cf67e3aec942db988d3d93e635b014")
    for obj in (s1, r1, p1):
        collection.add(obj)

    with pytest.raises(ValueError, match="unknown component reference"):
        _analysis(collection).optimize_regions(region_ids=["r1"], method="least_squares")


def test_preview_fit_scope_cross_spectrum_expr(empty_collection, simple_gauss_spectrum) -> None:
    """Expr linking peaks across spectra pulls the dependent region into the preview."""
    x, y = simple_gauss_spectrum
    collection = empty_collection
    s1 = Spectrum(x, y, id_="s1")
    s2 = Spectrum(x.copy(), y.copy(), id_="s2")
    r1 = Region(slice(20, len(x) - 20), parent_id=s1.id_, id_="r1")
    r2 = Region(slice(20, len(x) - 20), parent_id=s2.id_, id_="r2")
    p1 = Peak(
        model=PseudoVoigtPeakModel(),
        region_id=r1.id_,
        component_id="peakAA",
        amp=1.0,
        cen=0.0,
        sig=1.0,
        frac=0.0,
    )
    p2 = Peak(
        model=PseudoVoigtPeakModel(),
        region_id=r2.id_,
        component_id="peakBB",
        amp=0.5,
        cen=0.0,
        sig=1.0,
        frac=0.0,
    )
    p2.parameters["amp"].set(expr="2 * peakAA")
    for obj in (s1, s2, r1, r2, p1, p2):
        collection.add(obj)

    preview = _analysis(collection).preview_fit_scope(region_ids=["r2"])
    assert preview.needs_confirmation
    assert preview.extra_region_count == 1
    assert preview.extra_spectrum_count == 1
    assert preview.expanded_region_ids == ("r2", "r1")


def test_optimize_regions_expands_linked_regions(empty_collection, simple_gauss_spectrum) -> None:
    """optimize_regions fits the expression closure, not only the requested region."""
    x, y = simple_gauss_spectrum
    collection = empty_collection
    s1 = Spectrum(x, y, id_="s1")
    s2 = Spectrum(x.copy(), y.copy(), id_="s2")
    r1 = Region(slice(20, len(x) - 20), parent_id=s1.id_, id_="r1")
    r2 = Region(slice(20, len(x) - 20), parent_id=s2.id_, id_="r2")
    p1 = Peak(
        model=PseudoVoigtPeakModel(),
        region_id=r1.id_,
        component_id="peakAA",
        amp=1.0,
        cen=0.0,
        sig=1.0,
        frac=0.0,
    )
    p2 = Peak(
        model=PseudoVoigtPeakModel(),
        region_id=r2.id_,
        component_id="peakBB",
        amp=0.5,
        cen=0.0,
        sig=1.0,
        frac=0.0,
    )
    p2.parameters["amp"].set(expr="peakAA * 0.25", vary=True)
    for obj in (s1, s2, r1, r2, p1, p2):
        collection.add(obj)

    change = _analysis(collection).optimize_regions(region_ids=["r2"], method="least_squares")
    assert isinstance(change, CompositeChange)
    updated_ids = {c.component_id for c in change.changes[0].changes}
    assert updated_ids == {"peakAA", "peakBB"}


def test_optimize_regions_selection_only_skips_linked(
    empty_collection, simple_gauss_spectrum
) -> None:
    """expand_linked=False fits only the selection; cross-region exprs stay inactive."""
    x, y = simple_gauss_spectrum
    collection = empty_collection
    s1 = Spectrum(x, y, id_="s1")
    s2 = Spectrum(x.copy(), y.copy(), id_="s2")
    r1 = Region(slice(20, len(x) - 20), parent_id=s1.id_, id_="r1")
    r2 = Region(slice(20, len(x) - 20), parent_id=s2.id_, id_="r2")
    p1 = Peak(
        model=PseudoVoigtPeakModel(),
        region_id=r1.id_,
        component_id="peakAA",
        amp=1.0,
        cen=0.0,
        sig=1.0,
        frac=0.0,
    )
    p2 = Peak(
        model=PseudoVoigtPeakModel(),
        region_id=r2.id_,
        component_id="peakBB",
        amp=0.5,
        cen=0.0,
        sig=1.0,
        frac=0.0,
    )
    p2.parameters["amp"].set(expr="peakAA * 0.25", vary=True)
    for obj in (s1, s2, r1, r2, p1, p2):
        collection.add(obj)

    change = _analysis(collection).optimize_regions(
        region_ids=["r2"], expand_linked=False, method="least_squares"
    )
    updated_ids = {c.component_id for c in change.changes[0].changes}
    assert updated_ids == {"peakBB"}


def test_run_segmenter_returns_composite_for_empty_spectrum(
    empty_collection, simple_gauss_spectrum, monkeypatch
) -> None:
    """run_segmenter wraps NN CreatePeak/region changes in a CompositeChange."""
    from app.orchestration import AppOrchestrator

    x, y = simple_gauss_spectrum
    orch = AppOrchestrator(empty_collection, AppParameters())
    orch.create_spectrum(x, y, spectrum_id="s1")

    fake = CompositeChange(changes=[CreatePeak(region_id="r-new", model_name="pseudo-voigt")])
    monkeypatch.setattr(
        orch._nn,
        "run_segmenter",
        lambda spectrum_id, normalized_spectrum, original_spectrum: fake,
    )

    change = orch._analysis.run_segmenter(["s1"])
    assert isinstance(change, CompositeChange)
    assert change.changes == [fake]
