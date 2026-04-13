"""
Tests for tools.evaluation module (stateless evaluation functions).

The evaluation module operates on Protocol-typed objects (ComponentLike, RegionLike,
SpectrumLike). These tests use DTOs from DTOService as concrete implementations
of those protocols.
"""

import numpy as np
import pytest

from core.types import ComponentLike, RegionLike, SpectrumLike
from tools.evaluation import component_y, component_result, region_bundle, spectrum_bundle


@pytest.fixture
def simple_component(dto_service) -> ComponentLike:
    return dto_service.get_component("p1")


@pytest.fixture
def simple_region_bundle(dto_service) -> tuple[RegionLike, tuple[ComponentLike, ...]]:
    return dto_service.get_region_repr("r1")


@pytest.fixture
def simple_spectrum_bundle(dto_service) -> tuple[SpectrumLike, tuple[tuple[RegionLike, tuple[ComponentLike, ...]], ...]]:
    return dto_service.get_spectrum_repr("s1")


def test_component_y(
    simple_component: ComponentLike,
    x_axis: np.ndarray,
    simple_gauss: np.ndarray,
) -> None:
    y = component_y(simple_component, x_axis, simple_gauss)
    assert np.allclose(y, simple_gauss)


def test_component_result_wraps_correctly(
    simple_component: ComponentLike,
    x_axis: np.ndarray,
    simple_gauss: np.ndarray,
) -> None:
    res = component_result(simple_component, x_axis, simple_gauss)
    assert res.id_ == simple_component.id_
    assert res.parent_id == simple_component.parent_id
    assert res.kind == "peak"
    assert np.allclose(res.y, simple_gauss)


def test_region_bundle(
    simple_region_bundle: tuple[RegionLike, tuple[ComponentLike, ...]],
    x_axis: np.ndarray,
    simple_gauss: np.ndarray,
) -> None:
    res = region_bundle(*simple_region_bundle)
    rs = slice(20, len(x_axis) + 1 - 20)
    assert len(res.peaks) == 1
    assert res.background is not None
    assert np.allclose(res.model, simple_gauss[rs], atol=1)


def test_spectrum_bundle(
    simple_spectrum_bundle: tuple[SpectrumLike, tuple[tuple[RegionLike, tuple[ComponentLike, ...]], ...]],
) -> None:
    result = spectrum_bundle(*simple_spectrum_bundle)
    assert len(result.regions) == 1
