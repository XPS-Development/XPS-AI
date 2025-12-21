import numpy as np
import pytest

from lib.domain import Component, Peak, Background
from lib.parametrics import (
    RuntimeParameter,
    PseudoVoigtPeakModel,
    ConstantBackgroundModel,
)


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
