import pytest

from core import Peak, Background, Component, RuntimeParameter
from core.math_models import ModelRegistry


def test_create_peak_registers_peak(component_service, region_id):
    peak_id = component_service.create_peak(
        region_id=region_id,
        model_name="pseudo-voigt",
        parameters={"cen": 1.0, "amp": 10.0},
    )

    peak = component_service.collection.get(peak_id)

    assert isinstance(peak, Peak)
    assert peak.parent_id == region_id


def test_create_peak_with_explicit_id(component_service, region_id):
    peak_id = component_service.create_peak(
        region_id=region_id,
        model_name="pseudo-voigt",
        peak_id="p2",
    )

    assert peak_id == "p2"


def test_create_peak_rejects_background_model(component_service, region_id):
    with pytest.raises(ValueError):
        component_service.create_peak(
            region_id=region_id,
            model_name="linear",  # background model
        )


def test_replace_background_creates_background(component_service, region_id):
    bg_id = component_service.replace_background(
        region_id=region_id,
        model_name="linear",
    )

    bg = component_service.collection.get(bg_id)
    assert isinstance(bg, Background)
    assert bg.parent_id == region_id


def test_replace_background_replaces_existing(component_service, region_id):
    bg1 = component_service.replace_background(region_id, "linear")
    bg2 = component_service.replace_background(region_id, "linear")

    assert bg1 != bg2
    assert bg1 not in component_service.collection.objects_index


def test_replace_background_fails_if_multiple_backgrounds(component_service, region_id):
    # руками создаём неконсистентное состояние
    bg1 = component_service.replace_background(region_id, "linear")
    bg2 = component_service._create_component_obj(region_id, "linear")

    component_service.collection.add(bg2)

    with pytest.raises(RuntimeError):
        component_service.replace_background(region_id, "linear")


def test_remove_component(component_service, region_id):
    peak_id = component_service.create_peak(region_id, "pseudo-voigt")

    component_service.remove_component(peak_id)

    assert peak_id not in component_service.collection.objects_index


def test_get_parameter_returns_runtime_parameter(component_service, region_id):
    peak_id = component_service.create_peak(
        region_id,
        "pseudo-voigt",
        parameters={"cen": 5.0},
    )

    param = component_service.get_parameter(peak_id, "cen")

    assert isinstance(param, RuntimeParameter)
    assert param.value == 5.0


def test_get_parameters_returns_mapping(component_service, region_id):
    peak_id = component_service.create_peak(region_id, "pseudo-voigt")

    params = component_service.get_parameters(peak_id)

    assert isinstance(params, dict)
    assert all(isinstance(p, RuntimeParameter) for p in params.values())


def test_set_parameter_updates_value(component_service, region_id):
    peak_id = component_service.create_peak(
        region_id,
        "pseudo-voigt",
        parameters={"cen": 1.0},
    )

    component_service.set_parameter(peak_id, "cen", value=2.5)

    param = component_service.get_parameter(peak_id, "cen")
    assert param.value == 2.5


def test_set_parameter_updates_bounds(component_service, region_id):
    peak_id = component_service.create_peak(region_id, "pseudo-voigt")

    component_service.set_parameter(peak_id, "cen", lower=0.0, upper=10.0)

    p = component_service.get_parameter(peak_id, "cen")
    assert p.lower == 0.0
    assert p.upper == 10.0


def test_set_values_updates_multiple_parameters(component_service, region_id):
    peak_id = component_service.create_peak(
        region_id,
        "pseudo-voigt",
        parameters={"cen": 1.0, "amp": 5.0},
    )

    component_service.set_values(
        peak_id,
        {"cen": 2.0, "amp": 10.0},
    )

    params = component_service.get_parameters(peak_id)

    assert params["cen"].value == 2.0
    assert params["amp"].value == 10.0


def test_get_model_returns_model_instance(component_service, region_id):
    peak_id = component_service.create_peak(region_id, "pseudo-voigt")

    model = component_service.get_model(peak_id)

    assert isinstance(model, type(ModelRegistry.get("pseudo-voigt")))
