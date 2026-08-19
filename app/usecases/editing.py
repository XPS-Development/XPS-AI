"""
Editing use-cases: smart defaults around region, peak, and background mutations.

Builds Change objects from query state, AppParameters.automatic_methods, and
AutomatizationAdapter. Does not execute commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from app.command.changes import (
    BaseChange,
    CompositeChange,
    CreateBackground,
    CreatePeak,
    ReplaceBackgroundModel,
    UpdateRegionSlice,
)

if TYPE_CHECKING:
    from app.automatization import AutomatizationAdapter
    from app.orchestration import AppParameters, QueryService


class EditingUseCases:
    """
    Build Change objects for region/peak/background edits.

    Applies automatic parameter guessing when ``AppParameters.automatic_methods``
    is enabled and the caller omitted explicit parameters.
    """

    def __init__(
        self,
        query: QueryService,
        automatization: AutomatizationAdapter,
        params: AppParameters,
    ) -> None:
        """
        Initialize editing use-cases.

        Parameters
        ----------
        query
            Read-only query façade for collection and DTO access.
        automatization
            Adapter that turns guessed parameters into Change objects.
        params
            Application parameters; ``automatic_methods`` is read live.
        """
        self._query = query
        self._automatization = automatization
        self._params = params

    def create_peak(
        self,
        region_id: str,
        model_name: str,
        parameters: dict[str, float] | None = None,
        peak_id: str | None = None,
    ) -> BaseChange:
        """
        Build a change that creates a peak, optionally guessing pseudo-voigt params.

        Parameters
        ----------
        region_id
            Parent region identifier.
        model_name
            Registered peak model name.
        parameters
            Explicit parameter values. If None and automatic methods are on
            for ``pseudo-voigt``, parameters are guessed from residuals.
        peak_id
            Optional explicit peak identifier (ignored on the automatic path).

        Returns
        -------
        BaseChange
            ``CreatePeak`` with guessed or explicit parameters.
        """
        if self._params.automatic_methods and model_name == "pseudo-voigt" and parameters is None:
            region_repr = self._query.get_region_dto_repr(region_id, normalized=False)
            return self._automatization.create_pseudo_voigt_peak(region_repr[0], region_repr[1])
        return CreatePeak(
            region_id=region_id,
            model_name=model_name,
            parameters=parameters,
            peak_id=peak_id,
        )

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
            return self._automatization.create_background(
                region_id=region_id,
                spectrum_dto=spectrum,
                new_slice=(start, stop),
                slice_mode="index",
                model_name=model_name,
                background_id=background_id,
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
        bg_change = self._automatization.update_intensities(
            background_dto=background_dto,
            spectrum_dto=spectrum,
            new_slice=(start, stop),
            slice_mode=mode,
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
            parameters = self._automatization.get_bg_parameters(
                new_model_name,
                spectrum_dto,
                reg_slice,
                "index",
            )
        return ReplaceBackgroundModel(
            region_id=region_id,
            new_model_name=new_model_name,
            parameters=parameters,
            background_id=background_id,
        )
