from dataclasses import dataclass
import numpy as np

from core.math_models.base_models import EvaluationLikeFn
from .dto import BaseDTO, ComponentDTO, RegionDTO, SpectrumDTO

from typing import Literal
from numpy.typing import NDArray


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


class EvaluationService:
    """
    Stateless service for numerical evaluation of DTO-based spectral models.

    Operates exclusively on immutable DTO objects and performs
    numerical model evaluation without accessing domain state.
    """

    def get_eval_fn(self, component: ComponentDTO) -> EvaluationLikeFn:
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
        self,
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
        eval_fn = self.get_eval_fn(component)
        params = {name: p.value for name, p in component.parameters.items()}
        return eval_fn(x, y, **params)

    def component_result(
        self,
        component: ComponentDTO,
        x: NDArray,
        y: NDArray | None = None,
    ) -> ComponentEvaluationResult:
        """
        Evaluate component and wrap result into DTO.

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
            DTO containing evaluated component.
        """
        return ComponentEvaluationResult(
            id_=component.id_,
            parent_id=component.parent_id,
            normalized=component.normalized,
            y=self.component_y(component, x, y),
            kind=component.kind,
        )

    def region_bundle(
        self,
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
            Region numerical data.
        components : tuple[ComponentDTO, ...]
            Associated component DTOs.
        include_background : bool, optional
            If True, include background component in the model and residuals.

        Returns
        -------
        RegionEvaluationResult
            DTO containing evaluated region.
        """
        x = region.x
        y = region.y

        peak_results: list[ComponentEvaluationResult] = []
        background_result: ComponentEvaluationResult | None = None

        for c in components:
            res = self.component_result(c, x, y)
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
        self,
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
            Spectrum numerical data.
        regions : tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...]
            Tuples of (RegionDTO, component DTOs) for each region.
        include_background : bool, optional
            If True, include background components in the model and residuals.

        Returns
        -------
        SpectrumEvaluationResult
            DTO containing spectrum data and evaluated regions.
        """
        region_results = tuple(
            self.region_bundle(
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
