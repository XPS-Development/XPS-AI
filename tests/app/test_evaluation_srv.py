import numpy as np
import pytest

from app.dto import ComponentDTO, RegionDTO
from app.evaluation import EvaluationService


@pytest.fixture
def srv():
    return EvaluationService()


@pytest.fixture
def simple_component_dto(dto_service) -> ComponentDTO:
    return dto_service.get_component("p1")


@pytest.fixture
def simple_region_bundle(dto_service) -> tuple[RegionDTO, tuple[ComponentDTO, ...]]:
    return dto_service.get_region_repr("r1")


@pytest.fixture
def simple_spectrum_bundle(dto_service):
    return dto_service.get_spectrum_repr("s1")


def test_component_y(
    srv,
    simple_component_dto,
    x_axis,
    simple_gauss,
):
    srv = srv

    y = srv.component_y(simple_component_dto, x_axis, simple_gauss)

    assert np.allclose(y, simple_gauss)


def test_component_result_wraps_correctly(
    srv,
    simple_component_dto,
    x_axis,
    simple_gauss,
):
    srv = srv

    res = srv.component_result(simple_component_dto, x_axis, simple_gauss)

    assert res.id_ == simple_component_dto.id_
    assert res.parent_id == simple_component_dto.parent_id
    assert res.kind == "peak"
    assert np.allclose(res.y, simple_gauss)


def test_region_bundle(
    srv,
    simple_region_bundle,
    x_axis,
    simple_gauss,
):
    srv = srv

    res = srv.region_bundle(*simple_region_bundle)
    rs = slice(20, len(x_axis) + 1 - 20)

    assert len(res.peaks) == 1
    assert res.background is not None

    assert np.allclose(res.model, simple_gauss[rs], atol=1)


def test_spectrum_bundle(
    srv,
    simple_spectrum_bundle,
):
    srv = srv

    result = srv.spectrum_bundle(*simple_spectrum_bundle)

    assert len(result.regions) == 1
