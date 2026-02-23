import pytest

from core.objects import Peak, Background, Spectrum, Region
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
    with pytest.raises(TypeError):
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


def test_detach(srv, region_id):
    peak_id = srv.create_peak(region_id, "pseudo-voigt")

    peak = srv.detach(peak_id)[0]

    assert isinstance(peak, Peak)
    assert peak.id_ == peak_id
    assert peak_id not in srv.collection.objects_index


def test_get_parameter_returns_parameter_dict(srv, region_id):
    peak_id = srv.create_peak(
        region_id,
        "pseudo-voigt",
        parameters={"cen": 5.0},
    )

    param = srv.get_parameter(peak_id, "cen")

    assert isinstance(param, dict)
    assert param["value"] == 5.0
    assert "lower" in param
    assert "upper" in param
    assert "vary" in param


def test_get_parameters_returns_mapping(srv, region_id):
    peak_id = srv.create_peak(region_id, "pseudo-voigt")

    params = srv.get_parameters(peak_id)

    assert isinstance(params, dict)
    assert all(isinstance(p, dict) and "value" in p for p in params.values())


def test_set_parameter_updates_value(srv, region_id):
    peak_id = srv.create_peak(
        region_id,
        "pseudo-voigt",
        parameters={"cen": 1.0},
    )

    srv.set_parameter(peak_id, "cen", value=2.5)

    param = srv.get_parameter(peak_id, "cen")
    assert param["value"] == 2.5


def test_set_parameter_updates_bounds(srv, region_id):
    peak_id = srv.create_peak(region_id, "pseudo-voigt")

    srv.set_parameter(peak_id, "cen", lower=0.0, upper=10.0)

    p = srv.get_parameter(peak_id, "cen")
    assert p["lower"] == 0.0
    assert p["upper"] == 10.0


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

    assert params["cen"]["value"] == 2.0
    assert params["amp"]["value"] == 10.0


def _norm_ctx(srv, region_id):
    """Get normalization context for the spectrum that owns the region."""
    region = srv._get_typed(region_id, Region)
    spectrum = srv._get_typed(region.parent_id, Spectrum)
    return spectrum.norm_ctx


def test_get_parameter_normalized_returns_normalized_value_for_amp(srv, region_id):
    """For pseudo-voigt, amp is in normalization_target_parameters and uses scale only."""
    peak_id = srv.create_peak(
        region_id,
        "pseudo-voigt",
        parameters={"amp": 10.0, "cen": 1.0},
    )
    ctx = _norm_ctx(srv, region_id)

    raw = srv.get_parameter(peak_id, "amp", normalized=False)
    norm = srv.get_parameter(peak_id, "amp", normalized=True)

    assert raw["value"] == 10.0
    expected_norm = 10.0 / ctx.scale  # PseudoVoigtPeakModel uses use_offset=False, use_scale=True
    assert norm["value"] == pytest.approx(expected_norm)


def test_get_parameter_normalized_leaves_non_target_unchanged(srv, region_id):
    """Parameters not in normalization_target_parameters are unchanged when normalized=True."""
    peak_id = srv.create_peak(
        region_id,
        "pseudo-voigt",
        parameters={"cen": 5.0},
    )

    raw = srv.get_parameter(peak_id, "cen", normalized=False)
    norm = srv.get_parameter(peak_id, "cen", normalized=True)

    assert raw["value"] == norm["value"] == 5.0


def test_get_parameters_normalized_applies_to_target_params_only(srv, region_id):
    peak_id = srv.create_peak(
        region_id,
        "pseudo-voigt",
        parameters={"amp": 4.0, "cen": 2.0},
    )
    ctx = _norm_ctx(srv, region_id)

    raw = srv.get_parameters(peak_id, normalized=False)
    norm = srv.get_parameters(peak_id, normalized=True)

    assert raw["amp"]["value"] == 4.0
    assert norm["amp"]["value"] == pytest.approx(4.0 / ctx.scale)
    assert raw["cen"]["value"] == norm["cen"]["value"] == 2.0


def test_set_parameter_normalized_denormalizes_value(srv, region_id):
    peak_id = srv.create_peak(
        region_id,
        "pseudo-voigt",
        parameters={"amp": 1.0},
    )
    ctx = _norm_ctx(srv, region_id)
    norm_value = 0.5  # normalized amp

    srv.set_parameter(
        peak_id,
        "amp",
        value=norm_value,
        lower=0.0,
        upper=1.0,
        normalized=True,
    )

    raw = srv.get_parameter(peak_id, "amp", normalized=False)
    expected_raw = norm_value * ctx.scale
    assert raw["value"] == pytest.approx(expected_raw)


def test_set_values_normalized_denormalizes_values(srv, region_id):
    peak_id = srv.create_peak(
        region_id,
        "pseudo-voigt",
        parameters={"amp": 1.0, "cen": 0.0},
    )
    ctx = _norm_ctx(srv, region_id)

    srv.set_values(peak_id, {"amp": 0.5}, normalized=True)

    raw_amp = srv.get_parameter(peak_id, "amp", normalized=False)
    assert raw_amp["value"] == pytest.approx(0.5 * ctx.scale)
    # cen unchanged
    raw_cen = srv.get_parameter(peak_id, "cen", normalized=False)
    assert raw_cen["value"] == 0.0


def test_get_model_returns_model_instance(srv, region_id):
    peak_id = srv.create_peak(region_id, "pseudo-voigt")

    model = srv.get_model(peak_id)

    assert isinstance(model, type(ModelRegistry.get("pseudo-voigt")))
