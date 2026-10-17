from dataclasses import replace

import pytest

from core.dto import ComponentDTO, ParameterDTO
from core.math_models import (
    AsymPseudoVoigtPeakModel,
    ConstantBackgroundModel,
    PseudoVoigtPeakModel,
    TailPseudoVoigtPeakModel,
)
from core.post_processing import step_from_components


def component(cid="p", region="r", model=None, **params):
    model = model or PseudoVoigtPeakModel()
    values = {spec.name: spec.default for spec in model.parameter_schema}
    values.update(params)
    return ComponentDTO(
        cid,
        region,
        False,
        {
            name: ParameterDTO(name, value, -float("inf"), float("inf"), True, None)
            for name, value in values.items()
        },
        model,
        "background" if isinstance(model, ConstantBackgroundModel) else "peak",
    )


def test_physical_area_width_and_multiple_regions():
    components = [component("a", "r1", amp=120, sig=0.7), component("b", "r2", amp=0)]
    step = step_from_components("s", components)
    assert [p.region_id for p in step.peaks] == ["r1", "r2"]
    assert [p.area for p in step.peaks] == [120, 0]
    assert step.peaks[0].sig == pytest.approx(0.7)
    assert step.peaks[0].fwhm is None
    components[0].parameters["amp"] = replace(components[0].parameters["amp"], value=999)
    assert step.peaks[0].area == 120


@pytest.mark.parametrize("model", [AsymPseudoVoigtPeakModel(), TailPseudoVoigtPeakModel()])
def test_non_symmetric_models_keep_sig_without_fwhm(model):
    dto = component(model=model, amp=100, sig=0.8)
    step = step_from_components("s", [dto])
    assert step.peaks[0].fwhm is None
    assert step.peaks[0].sig == 0.8
    assert step.peaks[0].area == pytest.approx(
        model.area({k: v.value for k, v in dto.parameters.items()})
    )


def test_exclusions_and_empty_steps():
    invalid = component("missing")
    del invalid.parameters["cen"]
    step = step_from_components(
        "s",
        [
            component("bg", model=ConstantBackgroundModel()),
            invalid,
            component("nan", cen=float("nan")),
            component("zero", sig=0),
            component("inf", amp=float("inf")),
        ],
    )
    assert not step.peaks
    assert [e.reason for e in step.excluded] == [
        "background",
        "missing_or_invalid_position",
        "missing_or_invalid_position",
        "invalid_sig",
        "nonfinite_model_parameter",
    ]
    assert step_from_components("empty", []).peaks == ()


def test_normalized_and_duplicate_inputs_rejected():
    with pytest.raises(ValueError, match="unnormalized"):
        step_from_components("s", [replace(component(), normalized=True)])
    with pytest.raises(ValueError, match="Duplicate"):
        step_from_components("s", [component(), component()])


def test_adapter_accepts_physical_dtos_from_existing_service(dto_service):
    spectrum, regions = dto_service.get_spectrum_repr("s1", normalized=False)
    step = step_from_components(spectrum.id_, (c for _, components in regions for c in components))
    assert step.spectrum_id == "s1"
    assert len(step.peaks) == 1
    assert step.peaks[0].component_id == "p1"
    assert step.peaks[0].area == 1
    assert step.peaks[0].sig == 1
    assert step.peaks[0].fwhm is None
    assert step.excluded[0].component_id == "b1"
