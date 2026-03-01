"""Tests for AppOrchestrator: aggregation and orchestration of app services and commands."""

from uuid import uuid4

import numpy as np
import pytest

from app.orchestration import AppOrchestrator
from core.metadata import SpectrumMetadata
from core.objects import Background, Peak, Region, Spectrum
from tools.nn.segmenter import SegmenterResult
from tools.nn.types import (
    BackgroundDetectionResult,
    PeakDetectionResult,
    RegionDetectionResult,
)


def _make_segmenter_result(
    start: int = 20,
    stop: int = 180,
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
    background = (
        BackgroundDetectionResult(
            model_name="shirley",
            parameters={"i1": 0.1, "i2": 0.2},
        )
        if include_background
        else None
    )
    return SegmenterResult(
        region=RegionDetectionResult(start=start, stop=stop),
        peaks=peaks,
        background=background,
    )


@pytest.fixture
def orchestrator(empty_collection):
    """AppOrchestrator with empty collection."""
    return AppOrchestrator(empty_collection)


@pytest.fixture
def orchestrator_with_data(simple_collection):
    """AppOrchestrator with simple_collection (one spectrum, region, peak, background)."""
    return AppOrchestrator(simple_collection)


# ---- Construction and properties ----


def test_orchestrator_undo_redo_initially_false(orchestrator):
    """Initially can_undo and can_redo are False."""
    assert orchestrator.can_undo is False
    assert orchestrator.can_redo is False


# ---- Run app services ----


def test_orchestrator_import_spectra(empty_collection):
    """import_spectra parses file and executes CreateSpectrum + SetMetadata."""
    orch = AppOrchestrator(empty_collection)
    orch.import_spectra("tests/data/test_1_spec.txt")

    assert len(empty_collection.objects_index) >= 1
    spectrum_ids = [oid for oid, obj in empty_collection.objects_index.items() if isinstance(obj, Spectrum)]
    assert len(spectrum_ids) == 1
    meta = orch.ctx.metadata.get_metadata(spectrum_ids[0])
    assert meta is not None
    assert meta.name == "Ag3d"
    assert "test_1_spec.txt" in meta.file


def test_orchestrator_run_segmenter(empty_collection, simple_gauss_spectrum):
    """run_segmenter executes CreateRegion, CreateBackground, CreatePeak."""
    orch = AppOrchestrator(empty_collection)
    x, y = simple_gauss_spectrum
    sid = f"s{uuid4().hex}"
    orch.create_spectrum(x, y, spectrum_id=sid)

    norm_spec = orch.dto_service.get_spectrum(sid, normalized=True)
    orig_spec = orch.dto_service.get_spectrum(sid, normalized=False)
    orch._nn._pipeline.run = lambda n, o: [_make_segmenter_result(20, 180)]

    orch.run_segmenter(sid, norm_spec, orig_spec)

    regions = [o for o in empty_collection.objects_index.values() if isinstance(o, Region)]
    peaks = [o for o in empty_collection.objects_index.values() if isinstance(o, Peak)]
    backgrounds = [o for o in empty_collection.objects_index.values() if isinstance(o, Background)]
    assert len(regions) == 1
    assert len(peaks) == 1
    assert len(backgrounds) == 1
    assert regions[0].parent_id == sid


def test_orchestrator_optimize_regions(orchestrator_with_data, region_id, peak_id):
    """optimize_regions executes UpdateMultipleParameterValues."""
    orch = orchestrator_with_data
    region_reprs = [orch.dto_service.get_region_repr(region_id, normalized=True)]
    orch.optimize_regions(region_reprs, method="least_squares")

    params = orch.ctx.component.get_parameters(peak_id, normalized=True)
    assert len(params) > 0
    for p in params.values():
        assert np.isfinite(p["value"])


# ---- Create ----


def test_orchestrator_create_spectrum(orchestrator):
    """create_spectrum adds a spectrum to the collection."""
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 1, 100)  # non-constant so NormalizationContext accepts it
    orch = orchestrator
    orch.create_spectrum(x, y)

    spectra = [o for o in orch.core_collection.objects_index.values() if isinstance(o, Spectrum)]
    assert len(spectra) == 1
    assert spectra[0].x.shape == (100,)
    assert spectra[0].y.shape == (100,)


def test_orchestrator_create_region(orchestrator_with_data, spectrum_id):
    """create_region adds a region under a spectrum."""
    orch = orchestrator_with_data
    orch.create_region(spectrum_id, start=50, stop=150)

    regions = [o for o in orch.core_collection.objects_index.values() if isinstance(o, Region)]
    assert len(regions) >= 2
    new_regions = [
        r for r in regions if r.parent_id == spectrum_id and r.slice_.start == 50 and r.slice_.stop == 150
    ]
    assert len(new_regions) == 1


def test_orchestrator_create_peak(orchestrator_with_data, region_id):
    """create_peak adds a peak under a region."""
    orch = orchestrator_with_data
    orch.create_peak(region_id, "pseudo-voigt", parameters={"amp": 2.0, "cen": 0.0, "sig": 1.0, "frac": 0.0})

    peaks = [o for o in orch.core_collection.objects_index.values() if isinstance(o, Peak)]
    assert len(peaks) >= 2
    new_peaks = [p for p in peaks if p.parent_id == region_id]
    assert len(new_peaks) >= 1


def test_orchestrator_create_background(orchestrator_with_data, region_id):
    """create_background adds or replaces a background under a region."""
    orch = orchestrator_with_data
    orch.create_background(region_id, "shirley", parameters={"i1": 0.1, "i2": 0.2})

    backgrounds = [o for o in orch.core_collection.objects_index.values() if isinstance(o, Background)]
    assert len(backgrounds) >= 1


# ---- Metadata ----


def test_orchestrator_set_metadata(orchestrator_with_data, spectrum_id):
    """set_metadata stores metadata for an object."""
    orch = orchestrator_with_data
    meta = SpectrumMetadata(name="NewName", group="G1", file="/path/to/file")
    orch.set_metadata(spectrum_id, meta)

    assert orch.ctx.metadata.get_metadata(spectrum_id) == meta


# ---- Remove ----


def test_orchestrator_remove_object(orchestrator_with_data, peak_id):
    """remove_object detaches the object (and children) from the collection."""
    orch = orchestrator_with_data
    assert orch.ctx.query.check_object_exists(peak_id)
    orch.remove_object(peak_id)
    assert not orch.ctx.query.check_object_exists(peak_id)


def test_orchestrator_remove_metadata(orchestrator_with_data, spectrum_id):
    """remove_metadata clears metadata for an object."""
    orch = orchestrator_with_data
    meta = SpectrumMetadata(name="X", group="Y", file="Z")
    orch.set_metadata(spectrum_id, meta)
    assert orch.ctx.metadata.get_metadata(spectrum_id) is not None
    orch.remove_metadata(spectrum_id)
    assert orch.ctx.metadata.get_metadata(spectrum_id) is None


def test_orchestrator_full_remove_object(orchestrator_with_data, spectrum_id):
    """full_remove_object removes object and all metadata in subtree."""
    orch = orchestrator_with_data
    meta = SpectrumMetadata(name="X", group="Y", file="Z")
    orch.set_metadata(spectrum_id, meta)
    region_ids = list(orch.ctx.query.get_regions(spectrum_id))
    assert len(region_ids) >= 1
    orch.full_remove_object(spectrum_id)
    assert not orch.ctx.query.check_object_exists(spectrum_id)
    assert orch.ctx.metadata.get_metadata(spectrum_id) is None


# ---- Parameters and models ----


def test_orchestrator_update_parameter(orchestrator_with_data, peak_id):
    """update_parameter changes a single parameter field."""
    orch = orchestrator_with_data
    orch.update_parameter(peak_id, "cen", "value", 5.0)
    param = orch.ctx.component.get_parameter(peak_id, "cen")
    assert param["value"] == 5.0


def test_orchestrator_update_parameters(orchestrator_with_data, peak_id):
    """update_parameters changes multiple parameter values."""
    orch = orchestrator_with_data
    orch.update_parameters(peak_id, {"cen": 3.0, "amp": 10.0})
    params = orch.ctx.component.get_parameters(peak_id, normalized=False)
    assert params["cen"]["value"] == 3.0
    assert params["amp"]["value"] == 10.0


def test_orchestrator_update_region_slice(orchestrator_with_data, region_id):
    """update_region_slice changes region bounds."""
    orch = orchestrator_with_data
    orch.update_region_slice(region_id, 25, 175)
    sl = orch.ctx.region.get_slice(region_id)
    assert sl.start == 25
    assert sl.stop == 175


def test_orchestrator_replace_peak_model(orchestrator_with_data: AppOrchestrator, peak_id: str):
    """replace_peak_model swaps peak model preserving ID."""
    orch = orchestrator_with_data
    orch.replace_peak_model(
        peak_id, "pseudo-voigt", parameters={"amp": 1.0, "cen": 0.0, "sig": 1.0, "frac": 0.0}
    )
    assert orch.ctx.query.check_object_exists(peak_id)
    comp = orch.ctx.query._get(peak_id)
    assert isinstance(comp, Peak)


def test_orchestrator_replace_background_model(orchestrator_with_data, region_id):
    """replace_background_model swaps background model for region."""
    orch = orchestrator_with_data
    orch.replace_background_model(region_id, "shirley", parameters={"i1": 0.1, "i2": 0.2})
    bg_id = orch.ctx.query.get_background(region_id)
    assert bg_id is not None


# ---- Undo/redo ----


def test_orchestrator_undo_redo_after_import(empty_collection):
    """Undo after import removes spectra; redo restores them."""
    orch = AppOrchestrator(empty_collection)
    orch.import_spectra("tests/data/test_1_spec.txt")
    assert len(empty_collection.objects_index) >= 1
    assert orch.can_undo

    orch.undo()
    assert len(empty_collection.objects_index) == 0
    assert orch.can_redo

    orch.redo()
    assert len(empty_collection.objects_index) >= 1


def test_orchestrator_undo_redo_after_update_parameter(orchestrator_with_data, peak_id):
    """Undo after update_parameter restores old value; redo reapplies."""
    orch = orchestrator_with_data
    orch.update_parameter(peak_id, "cen", "value", 7.0)
    assert orch.ctx.component.get_parameter(peak_id, "cen")["value"] == 7.0

    orch.undo()
    assert orch.ctx.component.get_parameter(peak_id, "cen")["value"] == 0.0

    orch.redo()
    assert orch.ctx.component.get_parameter(peak_id, "cen")["value"] == 7.0


def test_orchestrator_execute_raw_change(orchestrator_with_data, peak_id):
    """execute() accepts any BaseChange (e.g. UpdateParameter)."""
    from app.command.changes import UpdateParameter

    orch = orchestrator_with_data
    orch.execute(UpdateParameter(peak_id, "cen", "value", 12.0))
    assert orch.ctx.component.get_parameter(peak_id, "cen")["value"] == 12.0


# ---- Serialization ----


def test_orchestrator_execute_marks_dirty(orchestrator_with_data, peak_id, tmp_path):
    """After execute(change), document is marked dirty (is_saved False after save)."""
    orch = orchestrator_with_data
    fp = tmp_path / "saved.json"
    orch.dump_collection(path=fp)
    assert orch.serialization.is_saved is True
    orch.update_parameter(peak_id, "cen", "value", 1.0)
    assert orch.serialization.is_saved is False


def test_orchestrator_dump_load_replace_clears_undo_stack(orchestrator_with_data, peak_id, tmp_path):
    """load_collection with mode=replace clears undo/redo stack."""
    orch = orchestrator_with_data
    fp = tmp_path / "coll.json"
    orch.dump_collection(path=fp)
    orch.update_parameter(peak_id, "cen", "value", 99.0)
    assert orch.can_undo
    orch.load_collection(fp, mode="replace")
    assert orch.can_undo is False
    assert orch.can_redo is False


def test_orchestrator_load_new_replaces_state(orchestrator_with_data, tmp_path):
    """load_collection with mode=new replaces collection/context and clears stack."""
    orch = orchestrator_with_data
    fp = tmp_path / "coll.json"
    orch.dump_collection(path=fp)
    old_collection = orch.core_collection
    old_ctx = orch.ctx
    orch.load_collection(fp, mode="new")
    assert orch.core_collection is not old_collection
    assert orch.ctx is not old_ctx
    assert orch.core_collection is orch.serialization._collection
    assert orch.can_undo is False
    assert len(orch.core_collection.objects_index) == len(old_collection.objects_index)


def test_orchestrator_set_get_default_save_path(orchestrator_with_data, tmp_path):
    """set_default_save_path and get_default_save_path delegate to service."""
    orch = orchestrator_with_data
    assert orch.get_default_save_path() is None
    p = tmp_path / "default.json"
    orch.set_default_save_path(p)
    assert orch.get_default_save_path() == p
    orch.dump_collection()
    assert p.exists()
