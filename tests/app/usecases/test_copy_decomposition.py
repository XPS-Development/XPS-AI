"""Tests for CopyDecompositionUseCases."""

from __future__ import annotations

import numpy as np
import pytest

from app.command.changes import (
    BaseChange,
    CompositeChange,
    CreatePeak,
    FullRemoveObject,
    UpdateParameter,
)
from app.query_service import QueryService
from app.usecases.copy_decomposition import CopyDecompositionUseCases
from core.collection import CoreCollection
from core.math_models import ConstantBackgroundModel, PseudoVoigtPeakModel
from core.math_models.normalization import NormalizationContext
from core.objects import Background, Peak, Region, Spectrum
from core.services import CoreContext


def _usecase(collection: CoreCollection) -> CopyDecompositionUseCases:
    return CopyDecompositionUseCases(QueryService(CoreContext.from_collection(collection)))


def _two_peak_source_and_targets() -> CoreCollection:
    """
    Source s1 with two linked peaks; empty s2; fitted s3.

    Peak ``p2.amp`` expr references ``p1`` (same-parameter component link).
    """
    collection = CoreCollection()
    x = np.linspace(-10.0, 10.0, 200)
    y_src = np.linspace(0.0, 10.0, 200)  # scale=10, offset=0
    y_tgt = np.linspace(5.0, 25.0, 200)  # scale=20, offset=5
    y_fitted = np.linspace(0.0, 5.0, 200)

    collection.add(Spectrum(x, y_src, id_="s1"))
    collection.add(Spectrum(x, y_tgt, id_="s2"))
    collection.add(Spectrum(x, y_fitted, id_="s3"))

    r1 = Region(slice(20, 180), parent_id="s1", id_="r1")
    collection.add(r1)
    p1 = Peak(
        model=PseudoVoigtPeakModel(),
        region_id="r1",
        component_id="p1",
        amp=2.0,
        cen=0.0,
        sig=1.0,
        frac=0.5,
    )
    p2 = Peak(
        model=PseudoVoigtPeakModel(),
        region_id="r1",
        component_id="p2",
        amp=1.0,
        cen=1.0,
        sig=1.0,
        frac=0.5,
    )
    p2.parameters["amp"].set(expr="p1")
    collection.add(p1)
    collection.add(p2)
    collection.add(
        Background(model=ConstantBackgroundModel(), region_id="r1", component_id="b1", const=1.0)
    )

    r3 = Region(slice(20, 180), parent_id="s3", id_="r3")
    collection.add(r3)
    collection.add(
        Peak(
            model=PseudoVoigtPeakModel(),
            region_id="r3",
            component_id="p3",
            amp=0.5,
            cen=0.0,
            sig=1.0,
            frac=0.0,
        )
    )
    return collection


def _composite(change: BaseChange) -> CompositeChange:
    assert isinstance(change, CompositeChange)
    return change


def _peaks(change: BaseChange) -> list[CreatePeak]:
    return [item for item in _composite(change).changes if isinstance(item, CreatePeak)]


def _peak_id_at_cen(change: BaseChange, cen: float) -> str:
    for peak in _peaks(change):
        params = peak.parameters
        peak_id = peak.peak_id
        if params is not None and peak_id is not None and params["cen"] == cen:
            return peak_id
    raise AssertionError(f"no copied peak at cen={cen}")


def _peak_params_at_cen(change: BaseChange, cen: float) -> dict[str, float]:
    for peak in _peaks(change):
        params = peak.parameters
        if params is not None and params["cen"] == cen:
            return params
    raise AssertionError(f"no copied peak at cen={cen}")


def _expr_map(change: BaseChange) -> dict[tuple[str, str], str | bool | float | None]:
    updates = [
        item
        for item in _composite(change).changes
        if isinstance(item, UpdateParameter) and item.parameter_field == "expr"
    ]
    return {(item.component_id, item.name): item.new_value for item in updates}


def test_link_off_rewrites_sibling_expr_onto_new_ids() -> None:
    """With link off, copied amp expr maps p1→new peak id within the copy set."""
    collection = _two_peak_source_and_targets()
    results = _usecase(collection).copy_decomposition(
        "s1",
        ["s2"],
        link_flags={},
        rescale_intensities=False,
    )
    assert len(results) == 1
    change = results[0][1]
    assert len(_peaks(change)) == 2
    new_p1 = _peak_id_at_cen(change, 0.0)
    new_p2 = _peak_id_at_cen(change, 1.0)

    exprs = _expr_map(change)
    assert exprs[(new_p2, "amp")] == new_p1
    assert (new_p1, "amp") not in exprs


def test_link_on_points_expr_at_source_component() -> None:
    """With link on, copied parameter expr is the source component id."""
    collection = _two_peak_source_and_targets()
    results = _usecase(collection).copy_decomposition(
        "s1",
        ["s2"],
        link_flags={("p2", "amp"): True},
        rescale_intensities=False,
    )
    assert len(results) == 1
    change = results[0][1]
    new_p2 = _peak_id_at_cen(change, 1.0)

    # Linked amp tracks the source peak itself (not the sibling rewrite).
    assert _expr_map(change)[(new_p2, "amp")] == "p2"


def test_rescale_amp_by_norm_scale_ratio() -> None:
    """Intensity amp is rescaled by tgt.scale / src.scale when rescale is on."""
    collection = _two_peak_source_and_targets()
    src_ctx = NormalizationContext.from_array(
        QueryService(CoreContext.from_collection(collection))
        .get_spectrum_dto("s1", normalized=False)
        .y
    )
    tgt_ctx = NormalizationContext.from_array(
        QueryService(CoreContext.from_collection(collection))
        .get_spectrum_dto("s2", normalized=False)
        .y
    )
    expected_amp = 2.0 * (tgt_ctx.scale / src_ctx.scale)

    results = _usecase(collection).copy_decomposition(
        "s1",
        ["s2"],
        link_flags={},
        rescale_intensities=True,
    )
    params = _peak_params_at_cen(results[0][1], 0.0)
    assert params["amp"] == pytest.approx(expected_amp)
    assert params["cen"] == pytest.approx(0.0)
    assert params["sig"] == pytest.approx(1.0)


def test_rescale_off_keeps_raw_amp() -> None:
    """With rescale off, amp values are copied unchanged."""
    collection = _two_peak_source_and_targets()
    results = _usecase(collection).copy_decomposition(
        "s1",
        ["s2"],
        link_flags={},
        rescale_intensities=False,
    )
    params = _peak_params_at_cen(results[0][1], 0.0)
    assert params["amp"] == pytest.approx(2.0)


def test_skip_targets_with_regions_unless_overwrite() -> None:
    """Fitted targets are skipped unless listed in overwrite_targets."""
    collection = _two_peak_source_and_targets()
    uc = _usecase(collection)

    skipped = uc.copy_decomposition("s1", ["s3"], link_flags={}, overwrite_targets=set())
    assert skipped == []

    overwritten = uc.copy_decomposition(
        "s1", ["s3"], link_flags={}, overwrite_targets={"s3"}, rescale_intensities=False
    )
    assert len(overwritten) == 1
    target_id, change = overwritten[0]
    assert target_id == "s3"
    removed = [item for item in _composite(change).changes if isinstance(item, FullRemoveObject)]
    assert any(item.obj_id == "r3" for item in removed)
    assert _peaks(change)


def test_empty_source_returns_no_changes() -> None:
    """Source without regions yields an empty change list."""
    collection = CoreCollection()
    x = np.linspace(-10.0, 10.0, 50)
    y = np.linspace(0.0, 1.0, 50)
    collection.add(Spectrum(x, y, id_="s1"))
    collection.add(Spectrum(x, y, id_="s2"))
    assert _usecase(collection).copy_decomposition("s1", ["s2"], link_flags={}) == []


def test_orchestrator_copy_cross_region_cen_expr() -> None:
    """
    Execute copy when two regions share a cen expr (regression for Create+Update).

    Previously CompositeChange built all UpdateParameter commands before applying
    Create*, which raised KeyError on the new component ids.
    """
    from app.orchestration import AppOrchestrator, AppParameters

    collection = CoreCollection()
    x = np.linspace(-10.0, 10.0, 200)
    y = np.linspace(0.0, 10.0, 200)
    collection.add(Spectrum(x, y, id_="s1"))
    collection.add(Spectrum(x, y, id_="s2"))

    collection.add(Region(slice(10, 80), parent_id="s1", id_="r1"))
    collection.add(Region(slice(100, 170), parent_id="s1", id_="r2"))
    collection.add(
        Peak(
            model=PseudoVoigtPeakModel(),
            region_id="r1",
            component_id="pLeft",
            amp=1.0,
            cen=-2.0,
            sig=1.0,
            frac=0.5,
        )
    )
    p_right = Peak(
        model=PseudoVoigtPeakModel(),
        region_id="r2",
        component_id="pRight",
        amp=1.0,
        cen=-2.0,
        sig=1.0,
        frac=0.5,
    )
    p_right.parameters["cen"].set(expr="pLeft")
    collection.add(p_right)
    collection.add(
        Background(model=ConstantBackgroundModel(), region_id="r1", component_id="b1", const=1.0)
    )
    collection.add(
        Background(model=ConstantBackgroundModel(), region_id="r2", component_id="b2", const=1.0)
    )

    orch = AppOrchestrator(collection, AppParameters())
    applied = orch.copy_decomposition(
        "s1",
        ["s2"],
        link_flags={},
        rescale_intensities=False,
        optimize_after=False,
    )
    assert applied == ["s2"]

    region_ids = orch.query.get_regions_ids("s2")
    assert len(region_ids) == 2
    peaks = [peak_id for rid in region_ids for peak_id in orch.query.get_peaks_ids(rid)]
    assert len(peaks) == 2
    exprs = {pid: orch.ctx.component.get_parameter(pid, "cen")["expr"] for pid in peaks}
    # One peak has rewritten expr pointing at the other copy; the other has None.
    assert None in exprs.values()
    linked = [e for e in exprs.values() if e is not None]
    assert len(linked) == 1
    assert linked[0] in peaks
