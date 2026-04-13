"""
Stateless numerical evaluation of spectral models.

Provides module-level functions that operate on Protocol-typed objects
(ComponentLike, RegionLike, SpectrumLike) for model evaluation without domain state.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from core.math_models.base_models import EvaluationLikeFn
from core.types import ComponentLike, RegionLike, SpectrumLike

from .dto import BaseDTO, RegionDTO, SpectrumDTO


@dataclass(frozen=True)
class ComponentEvaluationResult(BaseDTO):
    y: NDArray
    kind: Literal["peak", "background"]


@dataclass(frozen=True)
class RegionEvaluationResult(RegionDTO):
    peaks: tuple[ComponentEvaluationResult, ...]
    background: ComponentEvaluationResult | None
    model: NDArray
    residuals: NDArray


@dataclass(frozen=True)
class SpectrumEvaluationResult(SpectrumDTO):
    regions: tuple[RegionEvaluationResult, ...]


def get_eval_fn(component: ComponentLike) -> EvaluationLikeFn:
    """
    Return the evaluation function for a component's model.

    Parameters
    ----------
    component : ComponentLike
        Component-like object containing the model.

    Returns
    -------
    EvaluationLikeFn
        Model evaluation function.
    """
    return component.model.evaluate


def component_y(
    component: ComponentLike,
    x: NDArray,
    y: NDArray | None = None,
) -> NDArray:
    """
    Evaluate a single component model.

    Parameters
    ----------
    component : ComponentLike
        Component-like object containing model and parameters.
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
    component: ComponentLike,
    x: NDArray,
    y: NDArray | None = None,
) -> ComponentEvaluationResult:
    """
    Evaluate component and wrap result.

    Parameters
    ----------
    component : ComponentLike
        Component-like object containing model and parameters.
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
    region: RegionLike,
    components: tuple[ComponentLike, ...],
    *,
    include_background: bool = True,
) -> RegionEvaluationResult:
    """
    Evaluate all numerical signals for a region.

    Parameters
    ----------
    region : RegionLike
        Region-like object with numerical data.
    components : tuple[ComponentLike, ...]
        Associated component-like objects.
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
    spectrum: SpectrumLike,
    regions: tuple[tuple[RegionLike, tuple[ComponentLike, ...]], ...],
    *,
    include_background: bool = True,
) -> SpectrumEvaluationResult:
    """
    Evaluate numerical representations for an entire spectrum.

    Parameters
    ----------
    spectrum : SpectrumLike
        Spectrum-like object with numerical data.
    regions : tuple[tuple[RegionLike, tuple[ComponentLike, ...]], ...]
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
