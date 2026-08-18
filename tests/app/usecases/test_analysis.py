"""Tests for AnalysisUseCases Change construction."""

from typing import Any

import pytest

from app.command.changes import CompositeChange, CreatePeak
from app.nn_service import NNService
from app.optimization import OptimizationService
from app.orchestration import AppParameters, QueryService
from app.usecases.analysis import AnalysisUseCases
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

    change = _analysis(simple_collection, nn=_BoomNN(model_path=None)).run_segmenter(
        [spectrum_id]
    )
    assert change is None


def test_optimize_regions_requires_ids(simple_collection) -> None:
    """optimize_regions raises when neither region_ids nor spectrum_ids is given."""
    with pytest.raises(ValueError, match="region_ids or spectrum_ids"):
        _analysis(simple_collection).optimize_regions()


def test_run_segmenter_returns_composite_for_empty_spectrum(
    empty_collection, simple_gauss_spectrum, monkeypatch
) -> None:
    """run_segmenter wraps NN CreatePeak/region changes in a CompositeChange."""
    from app.orchestration import AppOrchestrator

    x, y = simple_gauss_spectrum
    orch = AppOrchestrator(empty_collection, AppParameters())
    orch.create_spectrum(x, y, spectrum_id="s1")

    fake = CompositeChange(
        changes=[CreatePeak(region_id="r-new", model_name="pseudo-voigt")]
    )
    monkeypatch.setattr(
        orch._nn,
        "run_segmenter",
        lambda spectrum_id, normalized_spectrum, original_spectrum: fake,
    )

    change = orch._analysis.run_segmenter(["s1"])
    assert isinstance(change, CompositeChange)
    assert change.changes == [fake]
