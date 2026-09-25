"""
Tests for core.evaluation module (stateless evaluation functions).

These tests use DTOs from DTOService as inputs to the evaluation helpers.
"""

import numpy as np
import pytest

from core.dto import ComponentDTO, RegionDTO, SpectrumDTO
from core.evaluation import (
    FitChiSquare,
    chi_square_contributions,
    component_result,
    component_y,
    peak_area,
    plot_data_from_evaluation,
    region_area,
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


def test_peak_and_region_area(
    simple_component: ComponentDTO,
    simple_region_bundle: tuple[RegionDTO, tuple[ComponentDTO, ...]],
) -> None:
    """Peak area is ``amp``; region area sums peaks and ignores background."""
    assert peak_area(simple_component) == pytest.approx(1.0)
    _region, components = simple_region_bundle
    backgrounds = [c for c in components if c.kind == "background"]
    assert backgrounds
    assert peak_area(backgrounds[0]) is None
    assert region_area(components) == pytest.approx(1.0)


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


def test_reduced_chi_square_is_undefined_without_degrees_of_freedom() -> None:
    """Reduced chi-square is omitted when there are no leftover degrees of freedom."""
    stat = FitChiSquare(chi_square=10.0, n_points=3, n_free_parameters=3)
    assert stat.reduced_chi_square is None


def test_chi_square_contributions_use_poisson_variance() -> None:
    """Each channel contributes (y - model)² / max(|y|, 1)."""
    y = np.array([0.0, 4.0, -9.0])
    model = np.array([1.0, 2.0, -12.0])
    assert np.allclose(chi_square_contributions(y, model), [1.0, 1.0, 1.0])


def test_plot_data_from_evaluation_builds_curves(
    simple_spectrum_bundle: tuple[
        SpectrumDTO, tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...]
    ],
) -> None:
    """plot_data_from_evaluation returns raw, fit, and chi-squared curves."""
    evaluated = spectrum_bundle(*simple_spectrum_bundle)
    plot_data = plot_data_from_evaluation(evaluated)

    kinds = {curve.kind for curve in plot_data.curves}
    assert "raw" in kinds
    assert "model" in kinds
    assert plot_data.residual_y_range is not None
    assert plot_data.residual_y_range[0] == 0.0

    region = evaluated.regions[0]
    residual_curves = [curve for curve in plot_data.curves if curve.kind == "residual"]
    assert len(residual_curves) == 1
    expected = chi_square_contributions(region.y, region.model)
    assert np.allclose(residual_curves[0].y, expected)
    assert not np.allclose(residual_curves[0].y, region.residuals)

    assert plot_data.chi_square is not None
    assert plot_data.chi_square.chi_square == pytest.approx(float(np.sum(expected)))
    assert plot_data.chi_square.n_points == expected.size
    assert plot_data.chi_square.n_free_parameters == 4
    assert plot_data.chi_square.reduced_chi_square == pytest.approx(
        float(np.sum(expected)) / (expected.size - 4)
    )

    peaks = [c for c in plot_data.curves if c.kind == "peak"]
    backgrounds = [c for c in plot_data.curves if c.kind == "background"]
    assert peaks
    assert all(c.component_id is not None for c in peaks)
    assert all(c.component_id is not None for c in backgrounds)
