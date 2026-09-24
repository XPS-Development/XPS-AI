"""
Editing use-cases: smart defaults around region, peak, and background mutations.

Builds Change objects from query state and AppParameters.automatic_methods.
Parameter guessing uses model ``guess_initial`` in core. Does not execute commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import numpy as np

from app.command.changes import (
    BaseChange,
    CompositeChange,
    CreateBackground,
    CreatePeak,
    CreateRegion,
    RemoveObject,
    RenameComponent,
    ReplaceBackgroundModel,
    ReplacePeakModel,
    UpdateMultipleParameterValues,
    UpdateRegionSlice,
)
from core.evaluation import component_y, region_bundle
from core.math_models import ModelRegistry
from core.math_models.guess_helpers import amplitude_from_peak_height, peak_index_from_residuals
from core.numerics import find_closest_index

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
        peak_index: int | None = None,
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
            Optional explicit peak identifier.
        peak_index
            Optional region-local channel index for ``guess_initial``. When
            omitted, the residual maximum is used.

        Returns
        -------
        BaseChange
            ``CreatePeak`` with guessed or explicit parameters.
        """
        if self._params.automatic_methods and parameters is None:
            region, components = self._query.get_region_dto_repr(region_id, normalized=False)
            parameters = self._guess_peak_parameters(
                region,
                components,
                model_name,
                peak_index=peak_index,
            )
        return CreatePeak(
            region_id=region_id,
            model_name=model_name,
            parameters=parameters,
            peak_id=peak_id,
        )

    def create_cursor_peak(self, region_id: str, cen: float, height: float) -> BaseChange:
        """
        Build a pseudo-Voigt peak placed at a plot click.

        ``cen`` is the click energy. ``height`` is the click intensity; the
        background at ``cen`` is subtracted so the drawn peak (background plus
        component) meets that intensity. Amplitude is the pseudo-Voigt value
        that produces the remaining height. Width is fixed at 1.

        Parameters
        ----------
        region_id
            Parent region identifier.
        cen
            Peak center in axis units.
        height
            Desired intensity of the peak top, including background.

        Returns
        -------
        CreatePeak
            Peak with ``sig=1``, ``frac=0.5``, and amplitude from ``height``.
        """
        sig = 1.0
        frac = 0.5
        baseline = self._background_at(region_id, cen)
        peak_height = max(float(height) - baseline, 0.0)
        return CreatePeak(
            region_id=region_id,
            model_name="pseudo-voigt",
            parameters={
                "amp": amplitude_from_peak_height(peak_height, sig, frac),
                "cen": float(cen),
                "sig": sig,
                "frac": frac,
            },
        )

    def _background_at(self, region_id: str, x: float) -> float:
        """Background intensity at ``x``, or 0 when the region has none."""
        if self._query.get_background_id(region_id) is None:
            return 0.0
        region, components = self._query.get_region_dto_repr(region_id, normalized=False)
        background = next((c for c in components if c.kind == "background"), None)
        if background is None or len(region.x) == 0:
            return 0.0
        y_bg = component_y(background, region.x, region.y)
        order = np.argsort(region.x)
        return float(np.interp(float(x), region.x[order], y_bg[order]))

    def split_region(self, region_id: str, x: float) -> BaseChange | None:
        """
        Build a change that splits a region at the nearest interior channel.

        The left piece keeps ``region_id`` with slice ``[start, index)``. A new
        region owns ``[index, stop)``. Peaks with ``cen < x[index]`` stay on the
        left; the rest are re-created under the right region with the same ids.
        An existing background is re-guessed on the left (when automatic methods
        are on) and duplicated onto the right with fresh intensities.

        Parameters
        ----------
        region_id
            Region to split.
        x
            Split position in spectrum axis units.

        Returns
        -------
        BaseChange or None
            ``CompositeChange`` for a valid split, or ``None`` when the region
            is too narrow or ``x`` does not map inside it.
        """
        spectrum_id = self._query.get_parent_id(region_id)
        spectrum = self._query.get_spectrum_dto(spectrum_id, normalized=False)
        start, stop = self._query.get_region_slice(region_id, mode="index")
        if stop - start < 2:
            return None

        index = find_closest_index(x, spectrum.x)
        # Both sides need at least one channel: start < index < stop.
        if index <= start or index >= stop:
            return None

        split_x = float(spectrum.x[index])
        changes: list[BaseChange] = []

        left_slice = self.update_region_slice(region_id, start, index, mode="index")
        if isinstance(left_slice, CompositeChange):
            changes.extend(left_slice.changes)
        else:
            changes.append(left_slice)

        right_region_id = f"r{uuid4().hex}"
        changes.append(
            CreateRegion(
                spectrum_id=spectrum_id,
                start=index,
                stop=stop,
                region_id=right_region_id,
                mode="index",
            )
        )

        background_id = self._query.get_background_id(region_id)
        if background_id is not None:
            background_dto = self._query.get_component_dto(background_id, normalized=False)
            right_bg_params = self._guess_background_parameters(
                background_dto.model.name,
                spectrum,
                (index, stop),
                slice_mode="index",
            )
            changes.append(
                CreateBackground(
                    region_id=right_region_id,
                    model_name=background_dto.model.name,
                    parameters=right_bg_params,
                )
            )

        for peak_id in self._query.get_peaks_ids(region_id):
            peak_dto = self._query.get_component_dto(peak_id, normalized=False)
            cen = peak_dto.parameters["cen"].value
            if cen < split_x:
                continue
            parameters = {name: param.value for name, param in peak_dto.parameters.items()}
            changes.append(RemoveObject(obj_id=peak_id))
            changes.append(
                CreatePeak(
                    region_id=right_region_id,
                    model_name=peak_dto.model.name,
                    parameters=parameters,
                    peak_id=peak_id,
                    name=peak_dto.name,
                )
            )

        return CompositeChange(changes=changes)

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
        peak_index: int | None = None,
    ) -> dict[str, float]:
        """Guess peak parameters via model ``guess_initial`` at ``peak_index``."""
        region_eval = region_bundle(region, components)
        if peak_index is None:
            resolved_index = peak_index_from_residuals(region_eval.residuals)
        else:
            n = len(region_eval.x)
            if n == 0:
                raise ValueError("Cannot guess peak parameters on an empty region")
            resolved_index = max(0, min(int(peak_index), n - 1))
        return ModelRegistry.get(model_name).guess_initial(
            region_eval.x,
            region_eval.y,
            peak_index=resolved_index,
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

    def rename_component(self, component_id: str, new_name: str | None) -> RenameComponent:
        """
        Build a change that sets a component display name.

        Parameters
        ----------
        component_id
            Peak or background identifier.
        new_name
            New label, or ``None``/blank to clear.

        Returns
        -------
        RenameComponent
            Undoable rename change.
        """
        return RenameComponent(component_id=component_id, new_name=new_name)

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
