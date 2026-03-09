"""
Tests for app.automatization module (AutomatizationService and Change outputs).

App-level tests: service uses CoreContext and returns Command-layer Change objects
(CompositeChange, CreatePeak, CreateBackground) for the command executor.
"""

import numpy as np
import pytest

from app.automatization import AutomatizationAdapter
from app.command.changes import (
    CompositeChange,
    CreateBackground,
    CreatePeak,
    UpdateMultipleParameterValues,
    UpdateRegionSlice,
)


@pytest.fixture
def adapter() -> AutomatizationAdapter:
    return AutomatizationAdapter()


def _dummy_spectrum() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 10.0, 201)
    y = np.linspace(1.0, 2.0, 201)
    return x, y


def test_update_slice_with_intensities_returns_composite_change(
    adapter: AutomatizationAdapter,
) -> None:
    """update_slice_with_intensities returns CompositeChange with slice and bg updates."""
    region_id = "region-1"
    background_id = "bg-1"
    x, y = _dummy_spectrum()
    change = adapter.update_slice_with_intensities(
        region_id=region_id,
        background_id=background_id,
        background_model_name="shirley",
        spectrum_x=x,
        spectrum_y=y,
        start=25,
        stop=175,
        mode="index",
        avg_on=3,
    )
    assert isinstance(change, CompositeChange)
    assert len(change.changes) == 2
    assert isinstance(change.changes[0], UpdateRegionSlice)
    assert isinstance(change.changes[1], UpdateMultipleParameterValues)


def test_update_slice_with_intensities_slice_change_content(
    adapter: AutomatizationAdapter,
) -> None:
    """UpdateRegionSlice has correct region_id, start, stop, mode."""
    region_id = "region-1"
    background_id = "bg-1"
    x, y = _dummy_spectrum()
    change = adapter.update_slice_with_intensities(
        region_id=region_id,
        background_id=background_id,
        background_model_name="shirley",
        spectrum_x=x,
        spectrum_y=y,
        start=30,
        stop=170,
        mode="index",
        avg_on=3,
    )
    slice_change = change.changes[0]
    assert isinstance(slice_change, UpdateRegionSlice)
    assert slice_change.region_id == region_id
    assert slice_change.start == 30
    assert slice_change.stop == 170
    assert slice_change.mode == "index"


def test_update_slice_with_intensities_background_parameters_constant(
    adapter: AutomatizationAdapter,
) -> None:
    """UpdateMultipleParameterValues has component_id and parameters for constant bg."""
    region_id = "region-1"
    background_id = "bg-1"
    x, y = _dummy_spectrum()
    change = adapter.update_slice_with_intensities(
        region_id=region_id,
        background_id=background_id,
        background_model_name="constant",
        spectrum_x=x,
        spectrum_y=y,
        start=25,
        stop=175,
        mode="index",
        avg_on=3,
    )
    bg_change = change.changes[1]
    assert isinstance(bg_change, UpdateMultipleParameterValues)
    assert bg_change.component_id == background_id
    assert "const" in bg_change.parameters


class _DummyRegion:
    def __init__(self, region_id: str) -> None:
        self.id_ = region_id


def test_create_pseudo_voigt_peak_returns_create_peak(
    adapter: AutomatizationAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """create_pseudo_voigt_peak returns CreatePeak with model_name and parameters."""
    region = _DummyRegion("region-1")
    components = ("c1", "c2")

    captured: dict[str, object] = {}

    def _fake_create_params(region_arg, components_arg):  # type: ignore[no-untyped-def]
        captured["region"] = region_arg
        captured["components"] = components_arg
        return {"amp": 1.0, "cen": 0.0, "sig": 1.0, "frac": 0.5}

    monkeypatch.setattr(
        "app.automatization.create_pseudo_voigt_peak_parameters",
        _fake_create_params,
    )

    change = adapter.create_pseudo_voigt_peak(region, components)  # type: ignore[arg-type]
    assert isinstance(change, CreatePeak)
    assert change.region_id == region.id_
    assert change.model_name == "pseudo-voigt"
    assert change.parameters is not None
    assert set(change.parameters.keys()) == {"amp", "cen", "sig", "frac"}
    assert captured["region"] is region
    assert captured["components"] == components


def test_create_background_shirley_returns_create_background(
    adapter: AutomatizationAdapter,
) -> None:
    """create_background with shirley returns CreateBackground with i1, i2."""
    region_id = "region-1"
    x, y = _dummy_spectrum()
    change = adapter.create_background(
        region_id=region_id,
        model_name="shirley",
        spectrum_x=x,
        spectrum_y=y,
        start=25,
        stop=175,
        mode="index",
        avg_on=3,
    )
    assert isinstance(change, CreateBackground)
    assert change.region_id == region_id
    assert change.model_name == "shirley"
    assert change.parameters is not None
    assert "i1" in change.parameters
    assert "i2" in change.parameters


def test_create_background_constant_returns_const_parameter(
    adapter: AutomatizationAdapter,
) -> None:
    """create_background with constant returns CreateBackground with const."""
    region_id = "region-1"
    x, y = _dummy_spectrum()
    change = adapter.create_background(
        region_id=region_id,
        model_name="constant",
        spectrum_x=x,
        spectrum_y=y,
        start=25,
        stop=175,
        mode="index",
        avg_on=3,
    )
    assert isinstance(change, CreateBackground)
    assert change.region_id == region_id
    assert change.model_name == "constant"
    assert change.parameters is not None
    assert "const" in change.parameters
