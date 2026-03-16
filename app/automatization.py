"""
Automatizations for app layer.

Provides services for routine tasks.
"""

from typing import Literal

from tools.automatization import (
    calculate_background_intensities,
    create_pseudo_voigt_peak_parameters,
)
from tools.dto import ComponentDTO, RegionDTO, SpectrumDTO

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

    @staticmethod
    def _i1_i2_to_const(params: dict[str, float]) -> dict[str, float]:
        return {"const": min(params["i1"], params["i2"])}

    def update_intensities(
        self,
        background_dto: ComponentDTO,
        spectrum_dto: SpectrumDTO,
        new_slice: tuple[int | float, int | float],
        slice_mode: Literal["value", "index"] = "index",
        avg_on: int = 3,
    ):
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
        """Create linear background parameters for a region."""

        params = self.get_bg_parameters(model_name, spectrum_dto, new_slice, slice_mode, avg_on)

        return CreateBackground(
            region_id=region_id,
            model_name=model_name,
            parameters=params,
            background_id=background_id,
        )

    def create_pseudo_voigt_peak(
        self, region: RegionDTO, components: tuple[ComponentDTO, ...]
    ) -> CreatePeak:
        """Create pseudo-voigt peak parameters for a region."""
        return CreatePeak(
            region_id=region.id_,
            model_name="pseudo-voigt",
            parameters=create_pseudo_voigt_peak_parameters(region, components),
        )

    def get_bg_parameters(
        self,
        model_name: str,
        spectrum_dto: SpectrumDTO,
        reg_slice: tuple[int | float, int | float],
        slice_mode: Literal["value", "index"] = "index",
        avg_on: int = 3,
    ) -> dict[str, float]:
        """Get parameters for background replacement."""
        params = calculate_background_intensities(
            spectrum_dto.x,
            spectrum_dto.y,
            reg_slice[0],
            reg_slice[1],
            slice_mode,
            avg_on,
        )

        if model_name == "constant":
            return self._i1_i2_to_const(params)

        return params
