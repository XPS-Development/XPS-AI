"""Tests for NN app service and segmenter adapter."""

from uuid import uuid4

import numpy as np
import pytest

from tools.dto import DTOService

from app.command.changes import (
    CompositeChange,
    CreateBackground,
    CreatePeak,
    CreateRegion,
    CreateSpectrum,
)
from app.command.core import CommandExecutor, UndoRedoStack, create_default_registry
from app.nn_adapter import segmenter_results_to_changes
from app.nn_service import NNService
from core.services import CoreContext
from tools.nn.segmenter import SegmenterResult
from tools.nn.types import (
    BackgroundDetectionResult,
    PeakDetectionResult,
    RegionDetectionResult,
)


def _make_result(
    start: int = 10,
    stop: int = 100,
    peaks: tuple[PeakDetectionResult, ...] | None = None,
    include_background: bool = True,
) -> SegmenterResult:
    if peaks is None:
        peaks = (
            PeakDetectionResult(
                model_name="pseudo-voigt",
                parameters={"amp": 1.0, "cen": 50.0, "sig": 2.0, "frac": 0.0},
            ),
        )
    background: BackgroundDetectionResult | None
    if include_background:
        background = BackgroundDetectionResult(
            model_name="shirley",
            parameters={"i1": 0.1, "i2": 0.2},
        )
    else:
        background = None
    return SegmenterResult(
        region=RegionDetectionResult(start=start, stop=stop),
        peaks=peaks,
        background=background,
    )


def test_segmenter_results_to_changes_returns_composite_change():
    """segmenter_results_to_changes returns CompositeChange."""
    results = [_make_result()]
    change = segmenter_results_to_changes("spectrum-1", results)
    assert isinstance(change, CompositeChange)
    assert len(change.changes) == 3


def test_segmenter_results_to_changes_creates_region_background_peaks():
    """segmenter_results_to_changes produces CreateRegion, CreateBackground, CreatePeak."""
    results = [_make_result()]
    change = segmenter_results_to_changes("spectrum-1", results)

    create_regions = [c for c in change.changes if isinstance(c, CreateRegion)]
    create_backgrounds = [c for c in change.changes if isinstance(c, CreateBackground)]
    create_peaks = [c for c in change.changes if isinstance(c, CreatePeak)]

    assert len(create_regions) == 1
    assert len(create_backgrounds) == 1
    assert len(create_peaks) == 1

    region_id = create_regions[0].region_id
    assert region_id is not None
    assert region_id.startswith("r")
    assert create_backgrounds[0].region_id == region_id
    assert create_peaks[0].region_id == region_id

    assert create_regions[0].spectrum_id == "spectrum-1"
    assert create_regions[0].start == 10
    assert create_regions[0].stop == 100
    assert create_peaks[0].model_name == "pseudo-voigt"
    assert create_backgrounds[0].model_name == "shirley"


def test_segmenter_results_to_changes_multiple_regions():
    """segmenter_results_to_changes produces separate region/background/peak sets per result."""
    results = [_make_result(10, 50), _make_result(60, 120)]
    change = segmenter_results_to_changes("s1", results)

    assert len(change.changes) == 6
    create_regions = [c for c in change.changes if isinstance(c, CreateRegion)]
    assert len(create_regions) == 2
    assert create_regions[0].region_id != create_regions[1].region_id
    assert create_regions[0].start == 10 and create_regions[0].stop == 50
    assert create_regions[1].start == 60 and create_regions[1].stop == 120


def test_segmenter_results_to_changes_no_background():
    """segmenter_results_to_changes omits CreateBackground when background is None."""
    result = _make_result(include_background=False)
    change = segmenter_results_to_changes("s1", [result])

    create_backgrounds = [c for c in change.changes if isinstance(c, CreateBackground)]
    assert len(create_backgrounds) == 0
    assert len(change.changes) == 2  # region + 1 peak


def test_segmenter_results_to_changes_multiple_peaks():
    """segmenter_results_to_changes produces CreatePeak per peak in result."""
    peaks = (
        PeakDetectionResult("pseudo-voigt", {"amp": 1.0, "cen": 30.0, "sig": 2.0, "frac": 0.0}),
        PeakDetectionResult("pseudo-voigt", {"amp": 2.0, "cen": 70.0, "sig": 3.0, "frac": 0.1}),
    )
    result = _make_result(peaks=peaks)
    change = segmenter_results_to_changes("s1", [result])

    create_peaks = [c for c in change.changes if isinstance(c, CreatePeak)]
    assert len(create_peaks) == 2
    assert create_peaks[0].parameters["cen"] == 30.0
    assert create_peaks[1].parameters["cen"] == 70.0


def test_segmenter_changes_execute_via_command_executor(empty_collection, simple_gauss_spectrum):
    """Execute adapter output via CommandExecutor and verify regions/peaks/backgrounds."""
    x, y = simple_gauss_spectrum
    sid = f"s{uuid4().hex}"
    import_changes = CompositeChange(changes=[CreateSpectrum(x=x, y=y, spectrum_id=sid)])

    ctx = CoreContext.from_collection(empty_collection)
    stack = UndoRedoStack()
    executor = CommandExecutor(ctx, stack, create_default_registry())
    executor.execute(import_changes)

    # Build segmenter results and convert to changes
    results = [
        _make_result(start=20, stop=180),
    ]
    segmenter_changes = segmenter_results_to_changes(sid, results)
    executor.execute(segmenter_changes)

    regions = [obj for obj in empty_collection.objects_index.values() if obj.__class__.__name__ == "Region"]
    peaks = [obj for obj in empty_collection.objects_index.values() if obj.__class__.__name__ == "Peak"]
    backgrounds = [
        obj for obj in empty_collection.objects_index.values() if obj.__class__.__name__ == "Background"
    ]

    assert len(regions) == 1
    assert len(peaks) == 1
    assert len(backgrounds) == 1
    assert regions[0].parent_id == sid
    assert peaks[0].parent_id == regions[0].id_
    assert backgrounds[0].parent_id == regions[0].id_


@pytest.fixture
def dto_service(simple_collection):
    """DTO service for spectrum/region DTOs."""
    return DTOService(simple_collection)


def test_nn_service_run_segmenter_returns_composite_change(simple_collection, spectrum_id, dto_service):
    """NnService.run_segmenter returns CompositeChange (with patched pipeline)."""
    service = NNService(model_path=None)
    norm_spec = dto_service.get_spectrum(spectrum_id, normalized=True)
    orig_spec = dto_service.get_spectrum(spectrum_id, normalized=False)

    # Patch pipeline to return mock results without loading ONNX
    service._pipeline.run = lambda n, o: [_make_result(20, 180)]

    change = service.run_segmenter(spectrum_id, norm_spec, orig_spec)

    assert isinstance(change, CompositeChange)
    assert len(change.changes) >= 1
    create_regions = [c for c in change.changes if isinstance(c, CreateRegion)]
    assert len(create_regions) == 1
    assert create_regions[0].spectrum_id == spectrum_id
