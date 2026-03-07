"""
Tests for app.automatization module (AutomatizationService and Change outputs).

App-level tests: service uses CoreContext and returns Command-layer Change objects
(CompositeChange, CreatePeak, CreateBackground) for the command executor.
"""

import pytest

from app.automatization import AutomatizationService
from app.command.changes import (
    CompositeChange,
    UpdateRegionSlice,
    UpdateMultipleParameterValues,
    CreatePeak,
    CreateBackground,
)


@pytest.fixture
def service(ctx, dto_service) -> AutomatizationService:
    return AutomatizationService(ctx, dto=dto_service)


def test_update_slice_with_intensities_returns_composite_change(
    service: AutomatizationService,
    region_id: str,
) -> None:
    """update_slice_with_intensities returns CompositeChange with slice and bg updates."""
    change = service.update_slice_with_intensities(region_id, start=25, stop=175, mode="index", avg_on=3)
    assert isinstance(change, CompositeChange)
    assert len(change.changes) == 2
    assert isinstance(change.changes[0], UpdateRegionSlice)
    assert isinstance(change.changes[1], UpdateMultipleParameterValues)


def test_update_slice_with_intensities_slice_change_content(
    service: AutomatizationService,
    region_id: str,
) -> None:
    """UpdateRegionSlice has correct region_id, start, stop, mode."""
    change = service.update_slice_with_intensities(region_id, start=30, stop=170, mode="index")
    slice_change = change.changes[0]
    assert isinstance(slice_change, UpdateRegionSlice)
    assert slice_change.region_id == region_id
    assert slice_change.start == 30
    assert slice_change.stop == 170
    assert slice_change.mode == "index"


def test_update_slice_with_intensities_background_parameters(
    service: AutomatizationService,
    region_id: str,
) -> None:
    """UpdateMultipleParameterValues has component_id and parameters for constant bg."""
    change = service.update_slice_with_intensities(region_id, start=25, stop=175, mode="index", avg_on=3)
    bg_change = change.changes[1]
    assert isinstance(bg_change, UpdateMultipleParameterValues)
    assert bg_change.component_id is not None
    assert "const" in bg_change.parameters


def test_create_pseudo_voigt_peak_returns_create_peak(
    service: AutomatizationService,
    region_id: str,
) -> None:
    """create_pseudo_voigt_peak returns CreatePeak with model_name and parameters."""
    change = service.create_pseudo_voigt_peak(region_id)
    assert isinstance(change, CreatePeak)
    assert change.region_id == region_id
    assert change.model_name == "pseudo-voigt"
    assert change.parameters is not None
    assert "amp" in change.parameters
    assert "cen" in change.parameters
    assert "sig" in change.parameters
    assert "frac" in change.parameters


def test_create_background_shirley_returns_create_background(
    service: AutomatizationService,
    region_id: str,
) -> None:
    """create_background with shirley returns CreateBackground with i1, i2."""
    change = service.create_background(region_id, model_name="shirley", avg_on=3)
    assert isinstance(change, CreateBackground)
    assert change.region_id == region_id
    assert change.model_name == "shirley"
    assert change.parameters is not None
    assert "i1" in change.parameters
    assert "i2" in change.parameters


def test_create_background_constant_returns_const_parameter(
    service: AutomatizationService,
    region_id: str,
) -> None:
    """create_background with constant returns CreateBackground with const."""
    change = service.create_background(region_id, model_name="constant", avg_on=3)
    assert isinstance(change, CreateBackground)
    assert change.region_id == region_id
    assert change.model_name == "constant"
    assert "const" in change.parameters
