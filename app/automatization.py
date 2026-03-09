"""
Automatizations for app layer.

Provides services for routine tasks.
"""

from typing import Literal

from numpy.typing import NDArray

from tools.automatization import (
    calculate_background_intensities,
    create_pseudo_voigt_peak_parameters,
)
from tools.dto import ComponentDTO, RegionDTO

from .command.changes import (
    CompositeChange,
    CreateBackground,
    CreatePeak,
    UpdateMultipleParameterValues,
    UpdateRegionSlice,
)


class AutomatizationAdapter:
    """
    Service for automatization.

    Returns Change objects for CommandExecutor.
    """

    @staticmethod
    def _bg_parameters_adapter(model_name: str, params: dict[str, float]) -> dict[str, float]:
        if model_name == "constant":
            return {"const": min(params["i1"], params["i2"])}
        return params

    def update_slice_with_intensities(
        self,
        region_id: str,
        background_id: str,
        background_model_name: str,
        spectrum_x: NDArray,
        spectrum_y: NDArray,
        start: int | float,
        stop: int | float,
        mode: Literal["value", "index"] = "index",
        avg_on: int = 3,
    ) -> CompositeChange:

        slice_change = UpdateRegionSlice(region_id, start, stop, mode)
        bg_params = calculate_background_intensities(
            spectrum_x,
            spectrum_y,
            start,
            stop,
            mode,
            avg_on,
        )
        bg_change = UpdateMultipleParameterValues(
            component_id=background_id,
            parameters=self._bg_parameters_adapter(background_model_name, bg_params),
        )
        return CompositeChange(changes=[slice_change, bg_change])

    def create_pseudo_voigt_peak(
        self, region: RegionDTO, components: tuple[ComponentDTO, ...]
    ) -> CreatePeak:
        """Create pseudo-voigt peak parameters for a region."""
        return CreatePeak(
            region_id=region.id_,
            model_name="pseudo-voigt",
            parameters=create_pseudo_voigt_peak_parameters(region, components),
        )

    def create_background(
        self,
        region_id: str,
        spectrum_x: NDArray,
        spectrum_y: NDArray,
        start: int | float,
        stop: int | float,
        mode: Literal["value", "index"] = "index",
        model_name: str = "shirley",
        avg_on: int = 3,
    ) -> CreateBackground:
        """Create linear background parameters for a region."""

        parameters = calculate_background_intensities(
            spectrum_x,
            spectrum_y,
            start,
            stop,
            mode,
            avg_on,
        )
        parameters = self._bg_parameters_adapter(model_name, parameters)

        return CreateBackground(
            region_id=region_id,
            model_name=model_name,
            parameters=parameters,
        )
