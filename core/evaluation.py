"""
Stateless numerical evaluation of spectral models.

Provides module-level functions that operate on DTO projections
(ComponentDTO, RegionDTO, SpectrumDTO) for model evaluation without domain state.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from core.dto import BaseDTO, ComponentDTO, RegionDTO, SpectrumDTO
from core.math_models.base_models import EvaluationLikeFn


@dataclass(frozen=True)
class ComponentEvaluationResult(BaseDTO):
    """Evaluated intensity of a single component over a region."""

    y: NDArray
    kind: Literal["peak", "background"]


@dataclass(frozen=True)
class RegionEvaluationResult(RegionDTO):
    """Evaluated region: peaks, background, model, and residuals."""

    peaks: tuple[ComponentEvaluationResult, ...]
    background: ComponentEvaluationResult | None
    model: NDArray
    residuals: NDArray


@dataclass(frozen=True)
class SpectrumEvaluationResult(SpectrumDTO):
    """Evaluated spectrum containing per-region evaluation results."""

    regions: tuple[RegionEvaluationResult, ...]


PlotCurveKind = Literal["raw", "background", "peak", "model", "residual"]


@dataclass(frozen=True)
class PlotCurve:
    """Single curve ready for plotting."""

    x: NDArray
    y: NDArray
    kind: PlotCurveKind
    peak_index: int | None = None


@dataclass(frozen=True)
class SpectrumPlotData:
    """Display-ready plot series derived from a spectrum evaluation."""

    curves: tuple[PlotCurve, ...]
    residual_y_range: tuple[float, float] | None


def get_eval_fn(component: ComponentDTO) -> EvaluationLikeFn:
    """
    Return the evaluation function for a component's model.

    Parameters
    ----------
    component : ComponentDTO
        Component DTO containing the model.

    Returns
    -------
    EvaluationLikeFn
        Model evaluation function.
    """
    return component.model.evaluate


def component_y(
    component: ComponentDTO,
    x: NDArray,
    y: NDArray | None = None,
) -> NDArray:
    """
    Evaluate a single component model.

    Parameters
    ----------
    component : ComponentDTO
        Component DTO containing model and parameters.
    x : NDArray
        X-axis values for evaluation.
    y : NDArray, optional
        Reference signal (passed to model if required).

    Returns
    -------
    NDArray
        Model contribution evaluated on x.
    """
    eval_fn = get_eval_fn(component)
    params = {name: p.value for name, p in component.parameters.items()}
    return eval_fn(x, y, **params)


def component_result(
    component: ComponentDTO,
    x: NDArray,
    y: NDArray | None = None,
) -> ComponentEvaluationResult:
    """
    Evaluate component and wrap result.

    Parameters
    ----------
    component : ComponentDTO
        Component DTO containing model and parameters.
    x : NDArray
        X-axis values for evaluation.
    y : NDArray, optional
        Reference signal (passed to model if required).

    Returns
    -------
    ComponentEvaluationResult
        Evaluated component result.
    """
    return ComponentEvaluationResult(
        id_=component.id_,
        parent_id=component.parent_id,
        normalized=component.normalized,
        y=component_y(component, x, y),
        kind=component.kind,
    )


def region_bundle(
    region: RegionDTO,
    components: tuple[ComponentDTO, ...],
    *,
    include_background: bool = True,
) -> RegionEvaluationResult:
    """
    Evaluate all numerical signals for a region.

    Parameters
    ----------
    region : RegionDTO
        Region DTO with numerical data.
    components : tuple[ComponentDTO, ...]
        Associated component DTOs.
    include_background : bool, optional
        If True, include background component in the model and residuals.

    Returns
    -------
    RegionEvaluationResult
        Evaluated region result.
    """
    x = region.x
    y = region.y

    peak_results: list[ComponentEvaluationResult] = []
    background_result: ComponentEvaluationResult | None = None

    for c in components:
        res = component_result(c, x, y)
        if res.kind == "peak":
            peak_results.append(res)
        elif include_background:
            background_result = res

    # model signal
    model = np.zeros_like(x)

    for p in peak_results:
        model += p.y

    if background_result is not None:
        model += background_result.y

    residuals = y - model

    return RegionEvaluationResult(
        id_=region.id_,
        parent_id=region.parent_id,
        normalized=region.normalized,
        x=x,
        y=y,
        peaks=tuple(peak_results),
        background=background_result,
        model=model,
        residuals=residuals,
    )


def spectrum_bundle(
    spectrum: SpectrumDTO,
    regions: tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...],
    *,
    include_background: bool = True,
) -> SpectrumEvaluationResult:
    """
    Evaluate numerical representations for an entire spectrum.

    Parameters
    ----------
    spectrum : SpectrumDTO
        Spectrum DTO with numerical data.
    regions : tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...]
        Tuples of (region, components) for each region.
    include_background : bool, optional
        If True, include background components in the model and residuals.

    Returns
    -------
    SpectrumEvaluationResult
        Evaluated spectrum result.
    """
    region_results = tuple(
        region_bundle(
            region,
            components,
            include_background=include_background,
        )
        for region, components in regions
    )

    return SpectrumEvaluationResult(
        id_=spectrum.id_,
        parent_id=spectrum.parent_id,
        normalized=spectrum.normalized,
        x=spectrum.x,
        y=spectrum.y,
        regions=region_results,
    )


def plot_data_from_evaluation(result: SpectrumEvaluationResult) -> SpectrumPlotData:
    """
    Build display-ready plot curves from an evaluated spectrum.

    Parameters
    ----------
    result : SpectrumEvaluationResult
        Evaluated spectrum with per-region fit components.

    Returns
    -------
    SpectrumPlotData
        Curves for the main and residuals plots plus optional residuals y-range.
    """
    curves: list[PlotCurve] = [
        PlotCurve(x=result.x, y=result.y, kind="raw"),
    ]
    residual_arrays: list[NDArray] = []

    for region in result.regions:
        x = region.x
        bg_y = np.zeros_like(x) if region.background is None else region.background.y

        if region.background is not None:
            curves.append(PlotCurve(x=x, y=bg_y, kind="background"))

        for peak_index, peak in enumerate(region.peaks):
            curves.append(
                PlotCurve(
                    x=x,
                    y=bg_y + peak.y,
                    kind="peak",
                    peak_index=peak_index,
                )
            )

        curves.append(PlotCurve(x=x, y=region.model, kind="model"))

        if region.residuals.size > 0:
            curves.append(PlotCurve(x=x, y=region.residuals, kind="residual"))
            residual_arrays.append(region.residuals)

    residual_y_range: tuple[float, float] | None = None
    if residual_arrays:
        concat = np.concatenate(residual_arrays)
        r_min, r_max = float(np.min(concat)), float(np.max(concat))
        margin = max((r_max - r_min) * 0.1, 1e-12)
        residual_y_range = (r_min - margin, r_max + margin)

    return SpectrumPlotData(curves=tuple(curves), residual_y_range=residual_y_range)
