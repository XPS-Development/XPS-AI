"""
Automatizations for app layer.

Thin adapter: model ``guess_initial`` → Change objects for CommandExecutor.
"""

from typing import Literal

from core.dto import ComponentDTO, RegionDTO, SpectrumDTO
from core.evaluation import region_bundle
from core.math_models import ModelRegistry
from core.math_models.guess_helpers import peak_index_from_residuals

from .command.changes import (
    CreateBackground,
    CreatePeak,
    UpdateMultipleParameterValues,
)


class AutomatizationAdapter:
    """
    Service for automatization.

    Returns Change objects for CommandExecutor.
    """

    def update_intensities(
        self,
        background_dto: ComponentDTO,
        spectrum_dto: SpectrumDTO,
        new_slice: tuple[int | float, int | float],
        slice_mode: Literal["value", "index"] = "index",
        avg_on: int = 3,
    ) -> UpdateMultipleParameterValues:
        """Return a change that updates background intensities for a new slice."""
        bg_model_name = background_dto.model.name
        params = self.get_bg_parameters(bg_model_name, spectrum_dto, new_slice, slice_mode, avg_on)
        return UpdateMultipleParameterValues(component_id=background_dto.id_, parameters=params)

    def create_background(
        self,
        region_id: str,
        spectrum_dto: SpectrumDTO,
        new_slice: tuple[int | float, int | float],
        slice_mode: Literal["value", "index"] = "index",
        model_name: str = "shirley",
        background_id: str | None = None,
        avg_on: int = 3,
    ) -> CreateBackground:
        """Create background parameters for a region via model ``guess_initial``."""
        params = self.get_bg_parameters(model_name, spectrum_dto, new_slice, slice_mode, avg_on)

        return CreateBackground(
            region_id=region_id,
            model_name=model_name,
            parameters=params,
            background_id=background_id,
        )

    def create_peak(
        self,
        region: RegionDTO,
        components: tuple[ComponentDTO, ...],
        model_name: str,
    ) -> CreatePeak:
        """Create peak parameters from residuals via model ``guess_initial``."""
        region_eval = region_bundle(region, components)
        peak_index = peak_index_from_residuals(region_eval.residuals)
        parameters = ModelRegistry.get(model_name).guess_initial(
            region_eval.x,
            region_eval.y,
            peak_index=peak_index,
        )
        return CreatePeak(
            region_id=region.id_,
            model_name=model_name,
            parameters=parameters,
        )

    def get_bg_parameters(
        self,
        model_name: str,
        spectrum_dto: SpectrumDTO,
        reg_slice: tuple[int | float, int | float],
        slice_mode: Literal["value", "index"] = "index",
        avg_on: int = 3,
    ) -> dict[str, float]:
        """Get parameters for background creation or replacement."""
        return ModelRegistry.get(model_name).guess_initial(
            spectrum_dto.x,
            spectrum_dto.y,
            start=reg_slice[0],
            stop=reg_slice[1],
            mode=slice_mode,
            avg_on=avg_on,
        )
