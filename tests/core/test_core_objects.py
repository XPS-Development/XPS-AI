import numpy as np
import pytest

from core.core_objects import RuntimeParameter, Component, Peak, Background, Region, Spectrum
from core.math_models import PseudoVoigtPeakModel, ConstantBackgroundModel, LinearBackgroundModel


# === RuntimeParameter tests ===
def test_runtime_parameter_clips_on_init():
    p = RuntimeParameter(name="a", value=10, lower=0, upper=5)
    assert p.value == 5


def test_runtime_parameter_set_value_and_bounds():
    p = RuntimeParameter(name="a", value=1, lower=0, upper=10)

    p.set(value=20)
    assert p.value == 10

    p.set(value=-5)
    assert p.value == 0

    p.set(lower=5)
    assert p.value == 5

    p.set(upper=-3)
    assert p.value == 5
    assert p.upper == 5


def test_runtime_parameter_set_vary_and_expr():
    p = RuntimeParameter(name="a", value=1)

    p.set(vary=False, expr="b*2")
    assert p.vary is False
    assert p.expr == "b*2"


def test_runtime_parameter_clone_creates_new():
    p1 = RuntimeParameter(name="a", value=2, lower=0, upper=10)
    p2 = p1.clone(value=5)

    assert p1 is not p2
    assert p2.value == 5
    assert p1.value == 2


def test_runtime_parameter_copy_to():
    p1 = RuntimeParameter(name="a", value=2)
    p2 = RuntimeParameter(name="a", value=0)

    out = p1.copy_to(p2)

    assert out is None or out is p2
    assert p2.value == 2
    assert p1.value == 2


# === Component tests ===
def test_component_initializes_parameters_from_schema():
    model = PseudoVoigtPeakModel()
    c = Component(model=model, parent_id="r1")

    assert set(c.parameters) == {"amp", "cen", "sig", "frac"}
    assert isinstance(c.parameters["amp"], RuntimeParameter)
    assert c.parameters["amp"].value == 1
    assert c.parameters["sig"].lower == 0


def test_component_overrides_default_parameters():
    model = PseudoVoigtPeakModel()
    c = Component(model=model, parent_id="r1", amp=10, cen=5)

    assert c.parameters["amp"].value == 10
    assert c.parameters["cen"].value == 5
    assert c.parameters["sig"].value == 1  # default


def test_component_unknown_parameter_raises():
    model = PseudoVoigtPeakModel()
    with pytest.raises(ValueError, match="Unknown parameters"):
        Component(model=model, parent_id="r1", foo=123)


def test_component_set_param_updates_runtime_parameter():
    model = PseudoVoigtPeakModel()
    c = Component(model=model, parent_id="r1")

    c.set_param("amp", value=5, lower=1)

    p = c.get_param("amp")
    assert p.value == 5
    assert p.lower == 1


def test_component_set_param_unknown_name():
    model = PseudoVoigtPeakModel()
    c = Component(model=model, parent_id="r1")

    with pytest.raises(KeyError):
        c.set_param("unknown", value=1)


def test_component_get_param_unknown_name():
    model = PseudoVoigtPeakModel()
    c = Component(model=model, parent_id="r1")

    with pytest.raises(KeyError):
        c.get_param("unknown")


def test_peak_has_correct_parent_and_prefix():
    model = PseudoVoigtPeakModel()
    peak = Peak(model=model, region_id="region-123")

    assert peak.parent_id == "region-123"
    assert peak.id_.startswith("p")


def test_background_has_correct_prefix():
    model = ConstantBackgroundModel()
    bg = Background(model=model, region_id="r1")

    assert bg.id_.startswith("b")


# === Region tests ===
def test_region_creates_id_if_missing():
    r = Region(slice_=slice(0, 10), parent_id="s1")

    assert r.id_ is not None
    assert r.id_.startswith("r")


def test_region_rejects_non_slice():
    with pytest.raises(TypeError):
        Region(slice_=(0, 10), parent_id="s1")  # type: ignore


def test_region_rejects_invalid_step():
    with pytest.raises(ValueError):
        Region(slice_=slice(0, 10, 2), parent_id="s1")


def test_region_keeps_parent_id():
    r = Region(slice_=slice(5, 15), parent_id="spectrum-1")
    assert r.parent_id == "spectrum-1"


# === Spectrum tests ===
def test_spectrum_creates_norm_context():
    x = np.linspace(0, 10, 5)
    y = np.array([1, 2, 3, 4, 5])

    s = Spectrum(x=x, y=y)

    assert s.norm_ctx.offset == 1
    assert s.norm_ctx.scale == 4


def test_spectrum_creates_id():
    s = Spectrum(x=np.arange(3), y=np.arange(3))
    assert s.id_.startswith("s")


def test_spectrum_validates_shape():
    with pytest.raises(ValueError):
        Spectrum(x=np.arange(3), y=np.arange(4))


def test_spectrum_validates_ndim():
    with pytest.raises(ValueError):
        Spectrum(x=np.ones((3, 1)), y=np.ones((3,)))


def test_spectrum_parent_id_is_none():
    s = Spectrum(x=np.arange(3), y=np.arange(3))
    assert s.parent_id is None
