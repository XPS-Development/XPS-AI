"""
Tests for app.automatization module (AutomatizationService and Change outputs).

App-level tests: service uses CoreContext and returns Command-layer Change objects
(CompositeChange, CreatePeak, CreateBackground) for the command executor.
"""

import numpy as np
import pytest

from app.automatization import AutomatizationAdapter
from app.command.changes import CreateBackground, CreatePeak, UpdateMultipleParameterValues
from tools.dto import ComponentDTO, RegionDTO, SpectrumDTO


@pytest.fixture
def adapter() -> AutomatizationAdapter:
    return AutomatizationAdapter()


def _dummy_spectrum() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 10.0, 201)
    y = np.linspace(1.0, 2.0, 201)
    return x, y


def _dummy_spectrum_dto() -> SpectrumDTO:
    x, y = _dummy_spectrum()
    return SpectrumDTO(id_="spec-1", parent_id="root", normalized=False, x=x, y=y)


class _DummyModel:
    def __init__(self, name: str) -> None:
        self.name = name


def _dummy_background_dto(model_name: str = "shirley") -> ComponentDTO:
    return ComponentDTO(
        id_="bg-1",
        parent_id="region-1",
        normalized=False,
        parameters={},
        model=_DummyModel(model_name),
        kind="background",
    )


def test_update_intensities_returns_update_change_with_parameters(
    adapter: AutomatizationAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """update_intensities returns UpdateMultipleParameterValues with calculated parameters."""
    background_dto = _dummy_background_dto("shirley")
    spectrum_dto = _dummy_spectrum_dto()

    captured: dict[str, object] = {}

    def _fake_calc_bg_intensities(  # type: ignore[no-untyped-def]
        x,
        y,
        start,
        stop,
        mode,
        avg_on,
    ):
        captured["x"] = x
        captured["y"] = y
        captured["start"] = start
        captured["stop"] = stop
        captured["mode"] = mode
        captured["avg_on"] = avg_on
        return {"i1": 1.0, "i2": 2.0}

    monkeypatch.setattr("app.automatization.calculate_background_intensities", _fake_calc_bg_intensities)

    new_slice = (25, 175)
    change = adapter.update_intensities(
        background_dto=background_dto,
        spectrum_dto=spectrum_dto,
        new_slice=new_slice,
        slice_mode="index",
        avg_on=3,
    )

    assert isinstance(change, UpdateMultipleParameterValues)
    assert change.component_id == background_dto.id_
    assert change.parameters == {"i1": 1.0, "i2": 2.0}

    assert captured["x"] is spectrum_dto.x
    assert captured["y"] is spectrum_dto.y
    assert captured["start"] == new_slice[0]
    assert captured["stop"] == new_slice[1]
    assert captured["mode"] == "index"
    assert captured["avg_on"] == 3


def test_get_bg_parameters_constant_returns_const(
    adapter: AutomatizationAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_bg_parameters with constant model returns const based on i1, i2."""
    spectrum_dto = _dummy_spectrum_dto()

    def _fake_calc_bg_intensities(  # type: ignore[no-untyped-def]
        x,
        y,
        start,
        stop,
        mode,
        avg_on,
    ):
        return {"i1": 10.0, "i2": 20.0}

    monkeypatch.setattr("app.automatization.calculate_background_intensities", _fake_calc_bg_intensities)

    params = adapter.get_bg_parameters(
        model_name="constant",
        spectrum_dto=spectrum_dto,
        reg_slice=(25, 175),
        slice_mode="index",
        avg_on=3,
    )

    assert params == {"const": 10.0}


def test_create_pseudo_voigt_peak_returns_create_peak(
    adapter: AutomatizationAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """create_pseudo_voigt_peak returns CreatePeak with model_name and parameters."""
    region = RegionDTO(
        id_="region-1",
        parent_id="spec-1",
        normalized=False,
        x=np.linspace(0.0, 10.0, 201),
        y=np.linspace(1.0, 2.0, 201),
    )
    components = (
        ComponentDTO(
            id_="c1",
            parent_id=region.id_,
            normalized=False,
            parameters={},
            model=_DummyModel("background-1"),
            kind="background",
        ),
        ComponentDTO(
            id_="c2",
            parent_id=region.id_,
            normalized=False,
            parameters={},
            model=_DummyModel("background-2"),
            kind="background",
        ),
    )

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
    spectrum_dto = _dummy_spectrum_dto()
    change = adapter.create_background(
        region_id=region_id,
        spectrum_dto=spectrum_dto,
        new_slice=(25, 175),
        slice_mode="index",
        model_name="shirley",
        background_id="bg-1",
        avg_on=3,
    )
    assert isinstance(change, CreateBackground)
    assert change.region_id == region_id
    assert change.model_name == "shirley"
    assert change.background_id == "bg-1"
    assert change.parameters is not None
    assert "i1" in change.parameters
    assert "i2" in change.parameters


def test_create_background_constant_returns_const_parameter(
    adapter: AutomatizationAdapter,
) -> None:
    """create_background with constant returns CreateBackground with const."""
    region_id = "region-1"
    spectrum_dto = _dummy_spectrum_dto()
    change = adapter.create_background(
        region_id=region_id,
        spectrum_dto=spectrum_dto,
        new_slice=(25, 175),
        slice_mode="index",
        model_name="constant",
        background_id="bg-1",
        avg_on=3,
    )
    assert isinstance(change, CreateBackground)
    assert change.region_id == region_id
    assert change.model_name == "constant"
    assert change.background_id == "bg-1"
    assert change.parameters is not None
    assert "const" in change.parameters
