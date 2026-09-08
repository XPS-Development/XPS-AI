"""
Tests for core.evaluation module (stateless evaluation functions).

These tests use DTOs from DTOService as inputs to the evaluation helpers.
"""

import numpy as np
import pytest

from core.dto import ComponentDTO, RegionDTO, SpectrumDTO
from core.evaluation import (
    component_result,
    component_y,
    plot_data_from_evaluation,
    region_bundle,
    spectrum_bundle,
)


@pytest.fixture
def simple_component(dto_service) -> ComponentDTO:
    return dto_service.get_component("p1")


@pytest.fixture
def simple_region_bundle(dto_service) -> tuple[RegionDTO, tuple[ComponentDTO, ...]]:
    return dto_service.get_region_repr("r1")


@pytest.fixture
def simple_spectrum_bundle(
    dto_service,
) -> tuple[SpectrumDTO, tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...]]:
    return dto_service.get_spectrum_repr("s1")


def test_component_y(
    simple_component: ComponentDTO,
    x_axis: np.ndarray,
    simple_gauss: np.ndarray,
) -> None:
    y = component_y(simple_component, x_axis, simple_gauss)
    assert np.allclose(y, simple_gauss)


def test_component_result_wraps_correctly(
    simple_component: ComponentDTO,
    x_axis: np.ndarray,
    simple_gauss: np.ndarray,
) -> None:
    res = component_result(simple_component, x_axis, simple_gauss)
    assert res.id_ == simple_component.id_
    assert res.parent_id == simple_component.parent_id
    assert res.kind == "peak"
    assert np.allclose(res.y, simple_gauss)


def test_region_bundle(
    simple_region_bundle: tuple[RegionDTO, tuple[ComponentDTO, ...]],
    x_axis: np.ndarray,
    simple_gauss: np.ndarray,
) -> None:
    res = region_bundle(*simple_region_bundle)
    rs = slice(20, len(x_axis) + 1 - 20)
    assert len(res.peaks) == 1
    assert res.background is not None
    assert np.allclose(res.model, simple_gauss[rs], atol=1)


def test_spectrum_bundle(
    simple_spectrum_bundle: tuple[
        SpectrumDTO, tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...]
    ],
) -> None:
    result = spectrum_bundle(*simple_spectrum_bundle)
    assert len(result.regions) == 1


def test_plot_data_from_evaluation_builds_curves(
    simple_spectrum_bundle: tuple[
        SpectrumDTO, tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...]
    ],
) -> None:
    """plot_data_from_evaluation returns raw, fit, and residual curves."""
    evaluated = spectrum_bundle(*simple_spectrum_bundle)
    plot_data = plot_data_from_evaluation(evaluated)

    kinds = {curve.kind for curve in plot_data.curves}
    assert "raw" in kinds
    assert "model" in kinds
    assert plot_data.residual_y_range is not None
