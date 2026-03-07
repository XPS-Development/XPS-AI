"""
Automatizations for app layer.

Provides services for routine tasks.
"""

from typing import Literal

from core.services import CoreContext
from tools.automatization import AutomaticEvaluator
from tools.dto import DTOService

from .command.changes import (
    CompositeChange,
    CreateBackground,
    CreatePeak,
    UpdateMultipleParameterValues,
    UpdateRegionSlice,
)


class AutomatizationService:
    """
    Service for automatization.

    Returns Change objects for CommandExecutor.
    """

    def __init__(self, ctx: CoreContext, *, dto: DTOService | None = None) -> None:
        self._ctx = ctx
        self._evaluator = AutomaticEvaluator(ctx, dto=dto)

    def update_slice_with_intensities(
        self,
        region_id: str,
        start: int | float,
        stop: int | float,
        mode: Literal["value", "index"] = "index",
        avg_on: int = 3,
    ) -> CompositeChange:

        slice_change = UpdateRegionSlice(region_id, start, stop)

        bg_change = UpdateMultipleParameterValues(
            component_id=self._ctx.query.get_background(region_id),
            parameters=self._evaluator.calculate_background_parameters(region_id, start, stop, mode, avg_on),
        )
        return CompositeChange(changes=[slice_change, bg_change])

    def create_pseudo_voigt_peak(self, region_id: str) -> CreatePeak:
        """Create pseudo-voigt peak parameters for a region."""
        return CreatePeak(
            region_id=region_id,
            model_name="pseudo-voigt",
            parameters=self._evaluator.create_pseudo_voigt_peak_parameters(region_id),
        )

    def create_background(
        self,
        region_id: str,
        model_name: str = "shirley",
        avg_on: int = 3,
    ) -> CreateBackground:
        """Create linear background parameters for a region."""

        parameters = self._evaluator.create_i1_i2_parameters(region_id, avg_on=avg_on)

        if model_name == "constant":
            parameters = {"const": min(parameters["i1"], parameters["i2"])}

        return CreateBackground(
            region_id=region_id,
            model_name=model_name,
            parameters=parameters,
        )
