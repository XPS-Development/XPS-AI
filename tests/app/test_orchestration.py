"""Tests for AppOrchestrator: aggregation and orchestration of app services and commands."""

from uuid import uuid4

import numpy as np
import pytest

from app.orchestration import AppOrchestrator, AppParameters
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
    return AppOrchestrator(empty_collection, AppParameters())


@pytest.fixture
def orchestrator_with_data(simple_collection):
    """AppOrchestrator with simple_collection (one spectrum, region, peak, background)."""
    return AppOrchestrator(simple_collection, AppParameters())


# ---- Construction and properties ----


def test_orchestrator_undo_redo_initially_false(orchestrator):
    """Initially can_undo and can_redo are False."""
    assert orchestrator.can_undo is False
    assert orchestrator.can_redo is False


# ---- Run app services ----


def test_orchestrator_import_spectra(empty_collection):
    """import_spectra parses file and executes CreateSpectrum + SetMetadata."""
    orch = AppOrchestrator(empty_collection, AppParameters())
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
    orch = AppOrchestrator(empty_collection, AppParameters())
    x, y = simple_gauss_spectrum
    sid = f"s{uuid4().hex}"
    orch.create_spectrum(x, y, spectrum_id=sid)

    orch._nn._pipeline.run = lambda n, o: [_make_segmenter_result(20, 180)]

    orch.run_segmenter(spectrum_ids=[sid])

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
    orch.optimize_regions(region_ids=[region_id], method="least_squares")

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
    start, stop = orch.ctx.region.get_slice(region_id, mode="index")
    assert start == 25
    assert stop == 175


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
    orch = AppOrchestrator(empty_collection, AppParameters())
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
    assert orch.is_dirty is False
    orch.update_parameter(peak_id, "cen", "value", 1.0)
    assert orch.is_dirty is True


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


def test_orchestrator_set_get_default_save_path(orchestrator_with_data, tmp_path):
    """set_default_save_path and get_default_save_path update AppParameters."""
    orch = orchestrator_with_data
    assert orch.get_default_save_path() is None
    p = tmp_path / "default.json"
    orch.set_default_save_path(p)
    assert orch.get_default_save_path() == p
    orch.dump_collection()
    assert p.exists()


def test_orchestrator_import_uses_app_params(empty_collection, tmp_path):
    """import_spectra respects AppParameters.import_use_binding_energy and import_use_cps."""
    from app.orchestration import AppParameters

    params = AppParameters(import_use_binding_energy=True, import_use_cps=True)
    orch = AppOrchestrator(empty_collection, params)
    orch.import_spectra("tests/data/test_1_spec.txt")
    spectrum_ids = [oid for oid, obj in empty_collection.objects_index.items() if isinstance(obj, Spectrum)]
    assert len(spectrum_ids) == 1


def test_orchestrator_nn_service_initialized_from_params(empty_collection):
    """NNService is constructed with AppParameters.nn_* values."""
    from app.orchestration import AppParameters

    params = AppParameters(
        nn_model_path="/some/model.onnx",
        nn_pred_threshold=0.7,
        nn_smooth=False,
        nn_interp_num=128,
    )
    orch = AppOrchestrator(empty_collection, params)
    pl = orch._nn._pipeline
    assert str(pl.adapter._model_path) == "/some/model.onnx"
    assert pl.postprocessor._threshold == 0.7
    assert pl.postprocessor._smooth is False
    assert pl.preprocessor._num == 128


def test_orchestrator_optimize_regions_merges_default_kwargs(simple_collection, region_id):
    """optimize_regions merges AppParameters.optimization_kwargs with explicit kwargs."""
    from app.orchestration import AppParameters

    params = AppParameters(optimization_kwargs={"method": "leastsq", "xtol": 1e-8})
    orch = AppOrchestrator(simple_collection, params)
    orch.optimize_regions(region_ids=[region_id], xtol=1e-6)

    peak_id = next(pid for pid in orch.ctx.query.get_peaks(region_id))
    params_result = orch.ctx.component.get_parameters(peak_id, normalized=True)
    assert len(params_result) > 0
    for p in params_result.values():
        assert np.isfinite(p["value"])


def test_orchestrator_dump_raises_without_path(orchestrator_with_data):
    """dump_collection raises ValueError when path is None and no default is set."""
    orch = orchestrator_with_data
    with pytest.raises(ValueError, match="path is required"):
        orch.dump_collection()


# ---- Automatic methods (AutomatizationAdapter) ----


def test_create_peak_auto_params_when_automatic_methods(empty_collection, simple_gauss_spectrum):
    """create_peak with parameters=None uses AutomatizationAdapter when automatic_methods=True."""
    x, y = simple_gauss_spectrum
    orch = AppOrchestrator(empty_collection, AppParameters(automatic_methods=True))
    orch.create_spectrum(x, y, spectrum_id="s1")
    orch.create_region("s1", 20, 180, region_id="r1")
    orch.create_background("r1", "shirley", parameters=None)

    orch.create_peak("r1", "pseudo-voigt", parameters=None)

    peaks = [p for p in empty_collection.objects_index.values() if isinstance(p, Peak)]
    assert len(peaks) == 1
    params_dict = orch.ctx.component.get_parameters(peaks[0].id_, normalized=False)
    assert "amp" in params_dict
    assert "cen" in params_dict
    assert "sig" in params_dict
    assert "frac" in params_dict


def test_create_background_auto_params_when_automatic_methods(empty_collection, simple_gauss_spectrum):
    """create_background with parameters=None uses AutomatizationAdapter when automatic_methods=True."""
    x, y = simple_gauss_spectrum
    orch = AppOrchestrator(empty_collection, AppParameters(automatic_methods=True))
    orch.create_spectrum(x, y, spectrum_id="s1")
    orch.create_region("s1", 20, 180, region_id="r1")
    assert orch.ctx.query.get_background("r1") is None

    orch.create_background("r1", "shirley", parameters=None)

    bg_id = orch.ctx.query.get_background("r1")
    assert bg_id is not None
    params_dict = orch.ctx.component.get_parameters(bg_id, normalized=False)
    assert "i1" in params_dict
    assert "i2" in params_dict


def test_update_region_slice_updates_background_intensities(simple_collection, region_id, background_id):
    """update_region_slice with automatic_methods=True updates slice and background params when region has bg."""
    params = AppParameters(automatic_methods=True)
    orch = AppOrchestrator(simple_collection, params)

    orch.update_region_slice(region_id, 30, 170)
    start, stop = orch.ctx.region.get_slice(region_id, mode="index")
    assert start == 30
    assert stop == 170

    bg_params = orch.ctx.component.get_parameters(background_id, normalized=False)
    assert "const" in bg_params


def test_create_peak_explicit_params_when_automatic_methods_false(simple_collection, region_id):
    """create_peak with automatic_methods=False uses explicit parameters path."""
    params = AppParameters(automatic_methods=False)
    orch = AppOrchestrator(simple_collection, params)
    orch.create_peak(region_id, "pseudo-voigt", parameters={"amp": 5.0, "cen": 0.0, "sig": 1.5, "frac": 0.5})

    peaks = [p for p in simple_collection.objects_index.values() if isinstance(p, Peak)]
    new_peak = next(p for p in peaks if p.parent_id == region_id and p.id_ != "p1")
    assert orch.ctx.component.get_parameter(new_peak.id_, "amp")["value"] == 5.0


def test_update_region_slice_plain_when_no_background(empty_collection, simple_gauss_spectrum):
    """update_region_slice with no background uses plain UpdateRegionSlice only."""
    x, y = simple_gauss_spectrum
    orch = AppOrchestrator(empty_collection, AppParameters(automatic_methods=True))
    orch.create_spectrum(x, y, spectrum_id="s1")
    orch.create_region("s1", 20, 180, region_id="r1")
    assert orch.ctx.query.get_background("r1") is None

    orch.update_region_slice("r1", 25, 175)
    start, stop = orch.ctx.region.get_slice("r1", mode="index")
    assert start == 25
    assert stop == 175
