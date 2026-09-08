"""Tests for EditingUseCases Change construction (automatic_methods on/off)."""

from typing import Any, cast

import numpy as np
import pytest

from app.command.changes import (
    CompositeChange,
    CreateBackground,
    CreatePeak,
    ReplaceBackgroundModel,
    ReplacePeakModel,
    UpdateMultipleParameterValues,
    UpdateRegionSlice,
)
from app.parameters import AppParameters
from app.query_service import QueryService
from app.usecases.editing import EditingUseCases
from core.dto import ComponentDTO, SpectrumDTO
from core.math_models import ParametricModelLike
from core.services import CoreContext


def _editing(
    collection,
    *,
    automatic_methods: bool,
) -> EditingUseCases:
    ctx = CoreContext.from_collection(collection)
    return EditingUseCases(
        QueryService(ctx),
        AppParameters(automatic_methods=automatic_methods),
    )


def test_create_peak_auto_returns_guessed_pseudo_voigt(simple_collection, region_id) -> None:
    """automatic_methods and parameters=None yield CreatePeak with guessed PV params."""
    change = _editing(simple_collection, automatic_methods=True).create_peak(
        region_id, "pseudo-voigt", parameters=None
    )

    assert isinstance(change, CreatePeak)
    assert change.region_id == region_id
    assert change.model_name == "pseudo-voigt"
    assert change.parameters is not None
    assert set(change.parameters) == {"amp", "cen", "sig", "frac"}


def test_create_peak_explicit_when_automatic_methods_false(simple_collection, region_id) -> None:
    """automatic_methods=False keeps explicit CreatePeak parameters."""
    params = {"amp": 5.0, "cen": 0.0, "sig": 1.5, "frac": 0.5}
    change = _editing(simple_collection, automatic_methods=False).create_peak(
        region_id, "pseudo-voigt", parameters=params, peak_id="p-new"
    )

    assert isinstance(change, CreatePeak)
    assert change.parameters == params
    assert change.peak_id == "p-new"


def test_create_background_auto_guesses_intensities(simple_collection, region_id) -> None:
    """automatic_methods and parameters=None yield CreateBackground with i1/i2."""
    change = _editing(simple_collection, automatic_methods=True).create_background(
        region_id, "shirley", parameters=None
    )

    assert isinstance(change, CreateBackground)
    assert change.region_id == region_id
    assert change.model_name == "shirley"
    assert change.parameters is not None
    assert "i1" in change.parameters
    assert "i2" in change.parameters


def test_create_background_auto_guesses_constant(simple_collection, region_id) -> None:
    """automatic_methods guesses a finite const for the constant background model."""
    change = _editing(simple_collection, automatic_methods=True).create_background(
        region_id, "constant", parameters=None
    )

    assert isinstance(change, CreateBackground)
    assert change.model_name == "constant"
    assert change.parameters is not None
    assert set(change.parameters) == {"const"}
    assert np.isfinite(change.parameters["const"])


def test_create_background_explicit_when_automatic_methods_false(
    simple_collection, region_id
) -> None:
    """automatic_methods=False keeps explicit CreateBackground parameters."""
    params = {"i1": 0.1, "i2": 0.2}
    change = _editing(simple_collection, automatic_methods=False).create_background(
        region_id, "shirley", parameters=params, background_id="bg-new"
    )

    assert isinstance(change, CreateBackground)
    assert change.parameters == params
    assert change.background_id == "bg-new"


def test_update_region_slice_composites_background_when_auto(
    simple_collection, region_id, background_id
) -> None:
    """automatic_methods with an existing background returns CompositeChange."""
    change = _editing(simple_collection, automatic_methods=True).update_region_slice(
        region_id, 30, 170
    )

    assert isinstance(change, CompositeChange)
    assert len(change.changes) == 2
    slice_change, bg_change = change.changes
    assert isinstance(slice_change, UpdateRegionSlice)
    assert slice_change.region_id == region_id
    assert slice_change.start == 30
    assert slice_change.stop == 170
    assert isinstance(bg_change, UpdateMultipleParameterValues)
    assert bg_change.component_id == background_id


def test_update_region_slice_guesses_via_model_registry(
    simple_collection, region_id, background_id, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slice sync passes the new bounds into model guess_initial."""
    captured: dict[str, object] = {}

    class _FakeModel:
        def guess_initial(self, x, y, **kwargs):  # type: ignore[no-untyped-def]
            captured["x"] = x
            captured["y"] = y
            captured.update(kwargs)
            return {"i1": 1.0, "i2": 2.0}

    monkeypatch.setattr(
        "app.usecases.editing.ModelRegistry.get",
        lambda name: _FakeModel(),
    )

    change = _editing(simple_collection, automatic_methods=True).update_region_slice(
        region_id, 25, 175
    )

    assert isinstance(change, CompositeChange)
    bg_change = change.changes[1]
    assert isinstance(bg_change, UpdateMultipleParameterValues)
    assert bg_change.component_id == background_id
    assert bg_change.parameters == {"i1": 1.0, "i2": 2.0}
    assert captured["start"] == 25
    assert captured["stop"] == 175
    assert captured["mode"] == "index"
    assert captured["avg_on"] == 3


def test_update_region_slice_plain_when_automatic_methods_false(
    simple_collection, region_id
) -> None:
    """automatic_methods=False returns only UpdateRegionSlice."""
    change = _editing(simple_collection, automatic_methods=False).update_region_slice(
        region_id, 30, 170
    )

    assert isinstance(change, UpdateRegionSlice)
    assert change.start == 30
    assert change.stop == 170


def test_replace_background_model_auto_fills_parameters(simple_collection, region_id) -> None:
    """automatic_methods and parameters=None fill ReplaceBackgroundModel params."""
    change = _editing(simple_collection, automatic_methods=True).replace_background_model(
        region_id, "shirley", parameters=None
    )

    assert isinstance(change, ReplaceBackgroundModel)
    assert change.region_id == region_id
    assert change.new_model_name == "shirley"
    assert change.parameters is not None
    assert "i1" in change.parameters
    assert "i2" in change.parameters


def test_replace_background_model_explicit_when_automatic_methods_false(
    simple_collection, region_id
) -> None:
    """automatic_methods=False keeps explicit ReplaceBackgroundModel parameters."""
    params = {"i1": 1.0, "i2": 2.0}
    change = _editing(simple_collection, automatic_methods=False).replace_background_model(
        region_id, "shirley", parameters=params
    )

    assert isinstance(change, ReplaceBackgroundModel)
    assert change.parameters == params


def test_replace_peak_model_transfers_same_name_params(simple_collection, peak_id) -> None:
    """automatic_methods=False copies overlapping parameter values from the old peak."""
    change = _editing(simple_collection, automatic_methods=False).replace_peak_model(
        peak_id, "pseudo-voigt", parameters=None
    )

    assert isinstance(change, ReplacePeakModel)
    assert change.peak_id == peak_id
    assert change.new_model_name == "pseudo-voigt"
    assert change.parameters == {"amp": 1.0, "cen": 0.0, "sig": 1.0, "frac": 0.0}


def test_replace_peak_model_auto_guesses_and_transfers(simple_collection, peak_id) -> None:
    """automatic_methods=True guesses params and overlays same-name values from the old peak."""
    change = _editing(simple_collection, automatic_methods=True).replace_peak_model(
        peak_id, "pseudo-voigt", parameters=None
    )

    assert isinstance(change, ReplacePeakModel)
    assert change.parameters is not None
    assert set(change.parameters) == {"amp", "cen", "sig", "frac"}
    assert change.parameters["frac"] == 0.0


def test_replace_peak_model_explicit_params_skip_transfer(simple_collection, peak_id) -> None:
    """Explicit parameters bypass transfer and guessing."""
    params = {"amp": 9.0, "cen": 3.0, "sig": 2.0, "frac": 0.25}
    change = _editing(simple_collection, automatic_methods=True).replace_peak_model(
        peak_id, "pseudo-voigt", parameters=params
    )

    assert isinstance(change, ReplacePeakModel)
    assert change.parameters == params


def test_create_peak_and_return_id_embeds_peak_id(simple_collection, region_id) -> None:
    """create_peak_and_return_id returns a change with a known peak identifier."""
    editing = _editing(simple_collection, automatic_methods=False)
    change, peak_id = editing.create_peak_and_return_id(
        region_id, "pseudo-voigt", parameters={"amp": 1.0, "cen": 0.0, "sig": 1.0, "frac": 0.5}
    )

    assert isinstance(change, CreatePeak)
    assert change.peak_id == peak_id
    assert peak_id.startswith("p")


def test_guess_background_parameters_forwards_slice_bounds() -> None:
    """_guess_background_parameters calls model guess_initial with slice kwargs."""
    x = np.linspace(0.0, 10.0, 201)
    y = np.linspace(1.0, 2.0, 201)
    spectrum = SpectrumDTO(id_="spec-1", parent_id="root", normalized=False, x=x, y=y)

    params = EditingUseCases._guess_background_parameters(
        "constant",
        spectrum,
        (25, 175),
        slice_mode="index",
        avg_on=3,
    )

    assert set(params.keys()) == {"const"}
    assert np.isfinite(params["const"])


def test_guess_peak_parameters_uses_residuals(monkeypatch: pytest.MonkeyPatch) -> None:
    """_guess_peak_parameters passes peak_index from residuals into guess_initial."""
    from core.dto import RegionDTO

    region = RegionDTO(
        id_="region-1",
        parent_id="spec-1",
        normalized=False,
        x=np.linspace(0.0, 10.0, 201),
        y=np.linspace(1.0, 2.0, 201),
    )

    class _DummyModel:
        name = "shirley"

        def evaluate(self, x, y=None, **params):  # type: ignore[no-untyped-def]
            return np.zeros_like(x)

    components = (
        ComponentDTO(
            id_="c1",
            parent_id=region.id_,
            normalized=False,
            parameters={},
            model=cast(ParametricModelLike[Any], _DummyModel()),
            kind="background",
        ),
    )
    captured: dict[str, object] = {}

    class _FakeModel:
        def guess_initial(self, x, y, **kwargs):  # type: ignore[no-untyped-def]
            captured["peak_index"] = kwargs.get("peak_index")
            return {"amp": 1.0, "cen": 0.0, "sig": 1.0, "frac": 0.5}

    monkeypatch.setattr(
        "app.usecases.editing.ModelRegistry.get",
        lambda name: _FakeModel(),
    )

    params = EditingUseCases._guess_peak_parameters(region, components, "pseudo-voigt")

    assert set(params) == {"amp", "cen", "sig", "frac"}
    assert captured["peak_index"] is not None
