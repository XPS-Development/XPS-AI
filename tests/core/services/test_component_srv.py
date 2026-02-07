import pytest

from core.objects import Peak, Background, RuntimeParameter
from core.math_models import ModelRegistry
from core.services import ComponentService


@pytest.fixture
def srv(simple_collection):
    return ComponentService(simple_collection)


def test_create_peak_registers_peak(srv, region_id):
    peak_id = srv.create_peak(
        region_id=region_id,
        model_name="pseudo-voigt",
        parameters={"cen": 1.0, "amp": 10.0},
    )

    peak = srv.collection.get(peak_id)

    assert isinstance(peak, Peak)
    assert peak.parent_id == region_id


def test_create_peak_with_explicit_id(srv, region_id):
    peak_id = srv.create_peak(
        region_id=region_id,
        model_name="pseudo-voigt",
        peak_id="p2",
    )

    assert peak_id == "p2"


def test_create_peak_rejects_background_model(srv, region_id):
    with pytest.raises(ValueError):
        srv.create_peak(
            region_id=region_id,
            model_name="linear",  # background model
        )


def test_replace_background_creates_background(srv, region_id):
    bg_id = srv.replace_background(
        region_id=region_id,
        model_name="linear",
    )

    bg = srv.collection.get(bg_id)
    assert isinstance(bg, Background)
    assert bg.parent_id == region_id


def test_replace_background_replaces_existing(srv, region_id):
    bg1 = srv.replace_background(region_id, "linear")
    bg2 = srv.replace_background(region_id, "linear")

    assert bg1 != bg2
    assert bg1 not in srv.collection.objects_index


def test_replace_background_fails_if_multiple_backgrounds(srv, region_id):
    # руками создаём неконсистентное состояние
    bg1 = srv.replace_background(region_id, "linear")
    bg2 = srv._create_component_obj(region_id, "linear")

    srv.collection.add(bg2)

    with pytest.raises(RuntimeError):
        srv.replace_background(region_id, "linear")


def test_remove_component(srv, region_id):
    peak_id = srv.create_peak(region_id, "pseudo-voigt")

    srv.remove_component(peak_id)

    assert peak_id not in srv.collection.objects_index


def test_get_parameter_returns_runtime_parameter(srv, region_id):
    peak_id = srv.create_peak(
        region_id,
        "pseudo-voigt",
        parameters={"cen": 5.0},
    )

    param = srv.get_parameter(peak_id, "cen")

    assert isinstance(param, RuntimeParameter)
    assert param.value == 5.0


def test_get_parameters_returns_mapping(srv, region_id):
    peak_id = srv.create_peak(region_id, "pseudo-voigt")

    params = srv.get_parameters(peak_id)

    assert isinstance(params, dict)
    assert all(isinstance(p, RuntimeParameter) for p in params.values())


def test_set_parameter_updates_value(srv, region_id):
    peak_id = srv.create_peak(
        region_id,
        "pseudo-voigt",
        parameters={"cen": 1.0},
    )

    srv.set_parameter(peak_id, "cen", value=2.5)

    param = srv.get_parameter(peak_id, "cen")
    assert param.value == 2.5


def test_set_parameter_updates_bounds(srv, region_id):
    peak_id = srv.create_peak(region_id, "pseudo-voigt")

    srv.set_parameter(peak_id, "cen", lower=0.0, upper=10.0)

    p = srv.get_parameter(peak_id, "cen")
    assert p.lower == 0.0
    assert p.upper == 10.0


def test_set_values_updates_multiple_parameters(srv, region_id):
    peak_id = srv.create_peak(
        region_id,
        "pseudo-voigt",
        parameters={"cen": 1.0, "amp": 5.0},
    )

    srv.set_values(
        peak_id,
        {"cen": 2.0, "amp": 10.0},
    )

    params = srv.get_parameters(peak_id)

    assert params["cen"].value == 2.0
    assert params["amp"].value == 10.0


def test_get_model_returns_model_instance(srv, region_id):
    peak_id = srv.create_peak(region_id, "pseudo-voigt")

    model = srv.get_model(peak_id)

    assert isinstance(model, type(ModelRegistry.get("pseudo-voigt")))
