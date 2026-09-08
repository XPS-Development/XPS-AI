"""
Editing use-cases: smart defaults around region, peak, and background mutations.

Builds Change objects from query state and AppParameters.automatic_methods.
Parameter guessing uses model ``guess_initial`` in core. Does not execute commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from app.command.changes import (
    BaseChange,
    CompositeChange,
    CreateBackground,
    CreatePeak,
    ReplaceBackgroundModel,
    ReplacePeakModel,
    UpdateMultipleParameterValues,
    UpdateRegionSlice,
)
from core.evaluation import region_bundle
from core.math_models import ModelRegistry
from core.math_models.guess_helpers import peak_index_from_residuals

if TYPE_CHECKING:
    from app.parameters import AppParameters
    from app.query_service import QueryService
    from core.dto import ComponentDTO, RegionDTO, SpectrumDTO


class EditingUseCases:
    """
    Build Change objects for region/peak/background edits.

    Applies automatic parameter guessing when ``AppParameters.automatic_methods``
    is enabled and the caller omitted explicit parameters.
    """

    def __init__(
        self,
        query: QueryService,
        params: AppParameters,
    ) -> None:
        """
        Initialize editing use-cases.

        Parameters
        ----------
        query
            Read-only query façade for collection and DTO access.
        params
            Application parameters; ``automatic_methods`` is read live.
        """
        self._query = query
        self._params = params

    def create_peak(
        self,
        region_id: str,
        model_name: str,
        parameters: dict[str, float] | None = None,
        peak_id: str | None = None,
    ) -> BaseChange:
        """
        Build a change that creates a peak, optionally guessing parameters.

        Parameters
        ----------
        region_id
            Parent region identifier.
        model_name
            Registered peak model name.
        parameters
            Explicit parameter values. If None and automatic methods are on,
            parameters are guessed from residuals via the model.
        peak_id
            Optional explicit peak identifier (ignored on the automatic path).

        Returns
        -------
        BaseChange
            ``CreatePeak`` with guessed or explicit parameters.
        """
        if self._params.automatic_methods and parameters is None:
            region, components = self._query.get_region_dto_repr(region_id, normalized=False)
            parameters = self._guess_peak_parameters(region, components, model_name)
        return CreatePeak(
            region_id=region_id,
            model_name=model_name,
            parameters=parameters,
            peak_id=peak_id,
        )

    def create_peak_and_return_id(
        self,
        region_id: str,
        model_name: str,
        parameters: dict[str, float] | None = None,
        peak_id: str | None = None,
    ) -> tuple[BaseChange, str]:
        """
        Build a peak-creation change with a known component identifier.

        Parameters
        ----------
        region_id
            Parent region identifier.
        model_name
            Registered peak model name.
        parameters
            Explicit parameter values. If None and automatic methods are on,
            parameters are guessed from residuals via the model.
        peak_id
            Optional explicit peak identifier. When omitted, a new id is
            generated and embedded in the returned change.

        Returns
        -------
        tuple[BaseChange, str]
            Change to execute and the peak identifier that will be created.
        """
        resolved_id = peak_id or f"p{uuid4().hex}"
        change = self.create_peak(
            region_id,
            model_name,
            parameters=parameters,
            peak_id=resolved_id,
        )
        return change, resolved_id

    def create_background(
        self,
        region_id: str,
        model_name: str,
        parameters: dict[str, float] | None = None,
        background_id: str | None = None,
    ) -> BaseChange:
        """
        Build a change that creates a background, optionally guessing intensities.

        Parameters
        ----------
        region_id
            Parent region identifier.
        model_name
            Registered background model name.
        parameters
            Explicit parameter values. If None and automatic methods are on,
            intensities are guessed from the parent spectrum slice.
        background_id
            Optional explicit background identifier.

        Returns
        -------
        BaseChange
            ``CreateBackground`` with guessed or explicit parameters.
        """
        if self._params.automatic_methods and parameters is None:
            spectrum_id = self._query.get_parent_id(region_id)
            spectrum = self._query.get_spectrum_dto(spectrum_id, normalized=False)
            start, stop = self._query.get_region_slice(region_id, mode="index")
            parameters = self._guess_background_parameters(
                model_name,
                spectrum,
                (start, stop),
                slice_mode="index",
            )
        return CreateBackground(
            region_id=region_id,
            model_name=model_name,
            parameters=parameters,
            background_id=background_id,
        )

    def update_region_slice(
        self,
        region_id: str,
        start: float | None = None,
        stop: float | None = None,
        mode: Literal["value", "index"] = "index",
    ) -> BaseChange:
        """
        Build a change that updates a region slice, optionally syncing background.

        Parameters
        ----------
        region_id
            Region identifier.
        start
            New start index or x value.
        stop
            New stop index or x value.
        mode
            Whether start/stop are indices or axis values.

        Returns
        -------
        BaseChange
            ``UpdateRegionSlice``, or a ``CompositeChange`` that also updates
            background intensities when automatic methods are on and a
            background exists.
        """
        change = UpdateRegionSlice(region_id=region_id, start=start, stop=stop, mode=mode)

        if not self._params.automatic_methods:
            return change

        background_id = self._query.get_background_id(region_id)
        if background_id is None:
            return change

        background_dto = self._query.get_component_dto(background_id)
        spectrum_id = self._query.get_parent_id(region_id)
        spectrum = self._query.get_spectrum_dto(spectrum_id, normalized=False)

        # If start/stop are omitted, `UpdateRegionSlice` will keep the current
        # slice boundaries. For automatic background syncing we must therefore
        # feed intensity guessing the effective slice, not `(None, None)`.
        if start is None or stop is None:
            eff_start, eff_stop = self._query.get_region_slice(region_id, mode=mode)
        else:
            eff_start, eff_stop = start, stop

        params = self._guess_background_parameters(
            background_dto.model.name,
            spectrum,
            (eff_start, eff_stop),
            slice_mode=mode,
        )
        bg_change = UpdateMultipleParameterValues(
            component_id=background_dto.id_,
            parameters=params,
        )
        return CompositeChange(changes=[change, bg_change])

    def replace_background_model(
        self,
        region_id: str,
        new_model_name: str,
        parameters: dict[str, float] | None = None,
        background_id: str | None = None,
    ) -> BaseChange:
        """
        Build a change that replaces a background model, optionally filling params.

        Parameters
        ----------
        region_id
            Region whose background is replaced.
        new_model_name
            Registered background model name.
        parameters
            Explicit parameter values. If None and automatic methods are on,
            intensities are guessed from the current region slice.
        background_id
            Optional background identifier to replace.

        Returns
        -------
        BaseChange
            ``ReplaceBackgroundModel`` with guessed or explicit parameters.
        """
        if self._params.automatic_methods and parameters is None:
            spectrum_dto = self._query.get_spectrum_dto(
                self._query.get_parent_id(region_id),
                normalized=False,
            )
            reg_slice = self._query.get_region_slice(region_id, mode="index")
            parameters = self._guess_background_parameters(
                new_model_name,
                spectrum_dto,
                reg_slice,
                slice_mode="index",
            )
        return ReplaceBackgroundModel(
            region_id=region_id,
            new_model_name=new_model_name,
            parameters=parameters,
            background_id=background_id,
        )

    def replace_peak_model(
        self,
        peak_id: str,
        new_model_name: str,
        parameters: dict[str, float] | None = None,
    ) -> BaseChange:
        """
        Build a change that replaces a peak model, transferring same-name parameters.

        Parameters
        ----------
        peak_id
            Peak whose model is replaced.
        new_model_name
            Registered peak model name.
        parameters
            Explicit parameter values. When omitted, parameters with the same
            names in the old and new model schemas are copied from the peak.
            If automatic methods are on, remaining parameters are guessed from
            residuals (excluding the peak being replaced).

        Returns
        -------
        BaseChange
            ``ReplacePeakModel`` with transferred, guessed, or explicit parameters.
        """
        if parameters is not None:
            return ReplacePeakModel(
                peak_id=peak_id,
                new_model_name=new_model_name,
                parameters=parameters,
            )

        old_dto = self._query.get_component_dto(peak_id, normalized=False)
        transferred = self._same_name_parameters(old_dto, new_model_name)

        if self._params.automatic_methods:
            region_id = self._query.get_parent_id(peak_id)
            region, components = self._query.get_region_dto_repr(region_id, normalized=False)
            other_components = tuple(c for c in components if c.id_ != peak_id)
            guessed = self._guess_peak_parameters(region, other_components, new_model_name)
            parameters = {**guessed, **transferred}
        else:
            parameters = transferred or None

        return ReplacePeakModel(
            peak_id=peak_id,
            new_model_name=new_model_name,
            parameters=parameters,
        )

    @staticmethod
    def _guess_peak_parameters(
        region: RegionDTO,
        components: tuple[ComponentDTO, ...],
        model_name: str,
    ) -> dict[str, float]:
        """Guess peak parameters from residuals via model ``guess_initial``."""
        region_eval = region_bundle(region, components)
        peak_index = peak_index_from_residuals(region_eval.residuals)
        return ModelRegistry.get(model_name).guess_initial(
            region_eval.x,
            region_eval.y,
            peak_index=peak_index,
        )

    @staticmethod
    def _guess_background_parameters(
        model_name: str,
        spectrum_dto: SpectrumDTO,
        reg_slice: tuple[int | float, int | float],
        *,
        slice_mode: Literal["value", "index"] = "index",
        avg_on: int = 3,
    ) -> dict[str, float]:
        """Guess background parameters via model ``guess_initial``."""
        return ModelRegistry.get(model_name).guess_initial(
            spectrum_dto.x,
            spectrum_dto.y,
            start=reg_slice[0],
            stop=reg_slice[1],
            mode=slice_mode,
            avg_on=avg_on,
        )

    @staticmethod
    def _same_name_parameters(
        component_dto: ComponentDTO,
        new_model_name: str,
    ) -> dict[str, float]:
        """Copy parameter values whose names exist in the new model schema."""
        new_names = {spec.name for spec in ModelRegistry.get(new_model_name).parameter_schema}
        return {
            name: param.value
            for name, param in component_dto.parameters.items()
            if name in new_names
        }
