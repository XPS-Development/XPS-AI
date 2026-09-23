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
    n_free_parameters: int


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
    component_id: str | None = None


@dataclass(frozen=True)
class FitChiSquare:
    """
    Poisson chi-squared criterion for the plotted fit.

    Attributes
    ----------
    chi_square : float
        Sum of per-channel contributions ``(y - model)² / max(|y|, 1)``.
    n_points : int
        Number of channels included in the sum.
    n_free_parameters : int
        Varied parameters that are not bound by an expression.
    """

    chi_square: float
    n_points: int
    n_free_parameters: int

    @property
    def reduced_chi_square(self) -> float | None:
        """Return reduced chi-square, or None when degrees of freedom are not positive."""
        dof = self.n_points - self.n_free_parameters
        if dof <= 0:
            return None
        return self.chi_square / dof


@dataclass(frozen=True)
class SpectrumPlotData:
    """Display-ready plot series derived from a spectrum evaluation."""

    curves: tuple[PlotCurve, ...]
    residual_y_range: tuple[float, float] | None
    chi_square: FitChiSquare | None = None


def free_parameter_count(components: tuple[ComponentDTO, ...]) -> int:
    """
    Count parameters that the fitter is free to change.

    Parameters
    ----------
    components : tuple of ComponentDTO
        Peak and background components in one region.

    Returns
    -------
    int
        Parameters with ``vary`` set and no expression.
    """
    return sum(
        1
        for component in components
        for param in component.parameters.values()
        if param.vary and not param.expr
    )


def poisson_variance(measured: NDArray) -> NDArray:
    """
    Return per-channel Poisson variance ``max(|measured|, 1)``.

    The floor of 1 count keeps empty and negative channels finite.

    Parameters
    ----------
    measured : NDArray
        Measured intensity that sets the variance. This is the original
        spectrum, not a target with fixed components removed.

    Returns
    -------
    NDArray
        Variance at each channel.
    """
    return np.maximum(np.abs(measured), 1.0)


def chi_square_residual(difference: NDArray, measured: NDArray) -> NDArray:
    """
    Return the residual whose squares sum to the Poisson chi-squared.

    Parameters
    ----------
    difference : NDArray
        ``target - model`` on the fit grid. Equals ``y - model`` when the
        target is the measured spectrum.
    measured : NDArray
        Intensities that set the Poisson variance.

    Returns
    -------
    NDArray
        ``difference / sqrt(max(|measured|, 1))``.
    """
    return difference / np.sqrt(poisson_variance(measured))


def chi_square_contributions(y: NDArray, model: NDArray) -> NDArray:
    """
    Return per-channel Poisson chi-squared contributions.

    Variance is estimated as ``max(|y|, 1)`` so empty and negative channels
    stay finite. The chi-squared criterion is the sum of this array.

    Parameters
    ----------
    y : NDArray
        Measured intensity.
    model : NDArray
        Fitted model intensity on the same grid.

    Returns
    -------
    NDArray
        ``(y - model)² / max(|y|, 1)`` at each channel.
    """
    return chi_square_residual(y - model, y) ** 2


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
        n_free_parameters=free_parameter_count(components),
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
        Curves for the main plot and the chi-squared subplot, plus the
        summed χ² criterion when at least one region is evaluated.
    """
    curves: list[PlotCurve] = [
        PlotCurve(x=result.x, y=result.y, kind="raw"),
    ]
    residual_arrays: list[NDArray] = []
    total_chi_square = 0.0
    total_points = 0
    total_free_parameters = 0

    for region in result.regions:
        x = region.x
        bg_y = np.zeros_like(x) if region.background is None else region.background.y

        if region.background is not None:
            curves.append(
                PlotCurve(
                    x=x,
                    y=bg_y,
                    kind="background",
                    component_id=region.background.id_,
                )
            )

        for peak_index, peak in enumerate(region.peaks):
            curves.append(
                PlotCurve(
                    x=x,
                    y=bg_y + peak.y,
                    kind="peak",
                    peak_index=peak_index,
                    component_id=peak.id_,
                )
            )

        curves.append(PlotCurve(x=x, y=region.model, kind="model"))

        contributions = chi_square_contributions(region.y, region.model)
        if contributions.size > 0:
            curves.append(PlotCurve(x=x, y=contributions, kind="residual"))
            residual_arrays.append(contributions)
            total_chi_square += float(np.sum(contributions))
            total_points += int(contributions.size)
            total_free_parameters += region.n_free_parameters

    residual_y_range: tuple[float, float] | None = None
    if residual_arrays:
        c_max = float(np.max(np.concatenate(residual_arrays)))
        top = c_max * 1.1 if c_max > 0.0 else 1.0
        residual_y_range = (0.0, top)

    chi_square: FitChiSquare | None = None
    if total_points > 0:
        chi_square = FitChiSquare(
            chi_square=total_chi_square,
            n_points=total_points,
            n_free_parameters=total_free_parameters,
        )

    return SpectrumPlotData(
        curves=tuple(curves),
        residual_y_range=residual_y_range,
        chi_square=chi_square,
    )
