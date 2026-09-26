"""
Copy a spectrum's region/peak/background decomposition onto other spectra.

Builds CompositeChange objects (one per target) without executing them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING
from uuid import uuid4

from app.command.changes import (
    BaseChange,
    CompositeChange,
    CreateBackground,
    CreatePeak,
    CreateRegion,
    FullRemoveObject,
    UpdateParameter,
)
from core.fitting.expressions import rewrite_expression_component_ids, shortest_unique_prefix
from core.math_models import ModelRegistry
from core.math_models.normalization import NormalizationContext

if TYPE_CHECKING:
    from app.query_service import QueryService
    from core.dto import ComponentDTO, ParameterDTO


LinkFlags = Mapping[tuple[str, str], bool]
"""``(component_id, parameter_name) -> link_to_source``."""


class CopyDecompositionUseCases:
    """Build changes that clone one spectrum's fit onto other spectra."""

    def __init__(self, query: QueryService) -> None:
        """
        Initialize with a query façade.

        Parameters
        ----------
        query
            Read-only query façade for collection and DTO access.
        """
        self._query = query

    def copy_decomposition(
        self,
        source_spectrum_id: str,
        target_spectrum_ids: Sequence[str],
        link_flags: LinkFlags,
        *,
        rescale_intensities: bool = True,
        overwrite_targets: set[str] | frozenset[str] | None = None,
    ) -> list[tuple[str, BaseChange]]:
        """
        Build one composite change per target spectrum.

        Targets that already have regions and are not listed in
        ``overwrite_targets`` are skipped (no change). Targets in
        ``overwrite_targets`` have existing regions removed first.

        Parameters
        ----------
        source_spectrum_id
            Spectrum whose regions/components are the template.
        target_spectrum_ids
            Destinations (order preserved).
        link_flags
            Per-parameter flags: when True, the copy's parameter expr references
            the source component; when False, values are copied and source exprs
            are rewritten onto the new id map.
        rescale_intensities
            Scale intensity parameters via each spectrum's normalization context.
        overwrite_targets
            Target ids whose existing regions should be deleted before copy.

        Returns
        -------
        list of (str, BaseChange)
            ``(target_spectrum_id, CompositeChange)`` for each prepared target.
        """
        overwrite = set(overwrite_targets or ())
        source_region_ids = list(self._query.get_regions_ids(source_spectrum_id))
        if not source_region_ids:
            return []

        src_ctx = self._norm_ctx(source_spectrum_id)
        known_ids = self._all_component_ids()
        changes_out: list[tuple[str, BaseChange]] = []

        for target_id in target_spectrum_ids:
            if target_id == source_spectrum_id:
                continue
            existing = list(self._query.get_regions_ids(target_id))
            if existing and target_id not in overwrite:
                continue

            tgt_ctx = self._norm_ctx(target_id)
            parts: list[BaseChange] = []
            if existing and target_id in overwrite:
                for rid in existing:
                    parts.append(FullRemoveObject(obj_id=rid))

            id_map: dict[str, str] = {}
            create_ops: list[tuple[str, ComponentDTO, str]] = []

            for source_region_id in source_region_ids:
                start, stop = self._query.get_region_slice(source_region_id, mode="value")
                new_region_id = f"r{uuid4().hex}"
                parts.append(
                    CreateRegion(
                        spectrum_id=target_id,
                        start=start,
                        stop=stop,
                        region_id=new_region_id,
                        mode="value",
                    )
                )

                bg_id = self._query.get_background_id(source_region_id)
                if bg_id is not None:
                    bg_dto = self._query.get_component_dto(bg_id, normalized=False)
                    new_bg_id = f"b{uuid4().hex}"
                    id_map[bg_id] = new_bg_id
                    params = self._parameter_values(
                        bg_dto,
                        src_ctx=src_ctx,
                        tgt_ctx=tgt_ctx,
                        rescale=rescale_intensities,
                    )
                    parts.append(
                        CreateBackground(
                            region_id=new_region_id,
                            model_name=bg_dto.model.name,
                            parameters=params,
                            background_id=new_bg_id,
                            name=bg_dto.name,
                        )
                    )
                    create_ops.append((new_region_id, bg_dto, new_bg_id))

                for peak_id in self._query.get_peaks_ids(source_region_id):
                    peak_dto = self._query.get_component_dto(peak_id, normalized=False)
                    new_peak_id = f"p{uuid4().hex}"
                    id_map[peak_id] = new_peak_id
                    params = self._parameter_values(
                        peak_dto,
                        src_ctx=src_ctx,
                        tgt_ctx=tgt_ctx,
                        rescale=rescale_intensities,
                    )
                    parts.append(
                        CreatePeak(
                            region_id=new_region_id,
                            model_name=peak_dto.model.name,
                            parameters=params,
                            peak_id=new_peak_id,
                            name=peak_dto.name,
                        )
                    )
                    create_ops.append((new_region_id, peak_dto, new_peak_id))

            for _new_region_id, dto, new_cid in create_ops:
                parts.extend(
                    self._parameter_meta_and_expr_changes(
                        source_dto=dto,
                        new_component_id=new_cid,
                        id_map=id_map,
                        link_flags=link_flags,
                        known_component_ids=known_ids,
                    )
                )

            if parts:
                changes_out.append((target_id, CompositeChange(changes=parts)))

        return changes_out

    def _all_component_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for region_id in self._query.get_all_regions_ids():
            ids.extend(self._query.get_components_ids(region_id))
        return tuple(ids)

    def _norm_ctx(self, spectrum_id: str) -> NormalizationContext:
        spectrum = self._query.get_spectrum_dto(spectrum_id, normalized=False)
        try:
            return NormalizationContext.from_array(spectrum.y)
        except ValueError:
            return NormalizationContext(offset=0.0, scale=1.0)

    def _parameter_values(
        self,
        dto: ComponentDTO,
        *,
        src_ctx: NormalizationContext,
        tgt_ctx: NormalizationContext,
        rescale: bool,
    ) -> dict[str, float]:
        model = ModelRegistry.get(dto.model.name)
        intensity_names = set(model.normalization_target_parameters)
        values: dict[str, float] = {}
        for name, param in dto.parameters.items():
            value = float(param.value)
            if rescale and name in intensity_names:
                value = self._rescale_intensity(model, value, src_ctx, tgt_ctx)
            values[name] = value
        return values

    @staticmethod
    def _rescale_intensity(
        model: object,
        value: float,
        src_ctx: NormalizationContext,
        tgt_ctx: NormalizationContext,
    ) -> float:
        normalize = getattr(model, "normalize_value", None)
        denormalize = getattr(model, "denormalize_value", None)
        if normalize is None or denormalize is None:
            if src_ctx.scale == 0:
                return value
            return value * (tgt_ctx.scale / src_ctx.scale)
        normalized = normalize(value, src_ctx)
        return float(denormalize(normalized, tgt_ctx))

    def _parameter_meta_and_expr_changes(
        self,
        *,
        source_dto: ComponentDTO,
        new_component_id: str,
        id_map: Mapping[str, str],
        link_flags: LinkFlags,
        known_component_ids: Sequence[str],
    ) -> list[BaseChange]:
        changes: list[BaseChange] = []
        source_id = source_dto.id_
        for name, param in source_dto.parameters.items():
            changes.extend(self._bounds_and_vary_changes(new_component_id, name, param))
            linked = bool(link_flags.get((source_id, name), False))
            expr_change = self._expr_change(
                source_id=source_id,
                source_param=param,
                param_name=name,
                new_component_id=new_component_id,
                id_map=id_map,
                linked=linked,
                known_component_ids=known_component_ids,
            )
            if expr_change is not None:
                changes.append(expr_change)
        return changes

    @staticmethod
    def _bounds_and_vary_changes(
        component_id: str,
        name: str,
        param: ParameterDTO,
    ) -> list[BaseChange]:
        return [
            UpdateParameter(
                component_id=component_id,
                name=name,
                parameter_field="lower",
                new_value=float(param.lower),
            ),
            UpdateParameter(
                component_id=component_id,
                name=name,
                parameter_field="upper",
                new_value=float(param.upper),
            ),
            UpdateParameter(
                component_id=component_id,
                name=name,
                parameter_field="vary",
                new_value=bool(param.vary),
            ),
        ]

    @staticmethod
    def _expr_change(
        *,
        source_id: str,
        source_param: ParameterDTO,
        param_name: str,
        new_component_id: str,
        id_map: Mapping[str, str],
        linked: bool,
        known_component_ids: Sequence[str],
    ) -> UpdateParameter | None:
        if linked:
            short = shortest_unique_prefix(source_id, known_component_ids)
            return UpdateParameter(
                component_id=new_component_id,
                name=param_name,
                parameter_field="expr",
                new_value=short,
            )
        if not source_param.expr:
            return None
        rewritten = rewrite_expression_component_ids(source_param.expr, id_map)
        return UpdateParameter(
            component_id=new_component_id,
            name=param_name,
            parameter_field="expr",
            new_value=rewritten,
        )
