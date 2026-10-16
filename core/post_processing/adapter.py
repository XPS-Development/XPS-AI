"""Extract matching descriptors from physical component DTOs."""

from collections.abc import Iterable
from math import isfinite

from core.dto import ComponentDTO
from core.evaluation import peak_area
from core.post_processing.types import ExcludedComponent, PeakDescriptor, SeriesStep


def step_from_components(spectrum_id: str, components: Iterable[ComponentDTO]) -> SeriesStep:
    """Collect all regions of a spectrum without retaining mutable DTO contents.

    Normalized inputs are rejected: the application must request physical DTOs.
    Backgrounds and malformed fits are recorded as exclusions, never filtered by area.
    """
    peaks: list[PeakDescriptor] = []
    excluded: list[ExcludedComponent] = []
    seen: set[str] = set()
    for component in components:
        if component.id_ in seen:
            raise ValueError(f"Duplicate component ID: {component.id_}")
        seen.add(component.id_)
        if component.normalized:
            raise ValueError("Post-processing requires physical, unnormalized parameters")
        region_id = component.parent_id
        if not region_id:
            raise ValueError(f"Component {component.id_} has no source region")
        reason = None
        parameters = {name: param.value for name, param in component.parameters.items()}
        if component.kind == "background":
            reason = "background"
        elif "cen" not in parameters or not isfinite(parameters["cen"]):
            reason = "missing_or_invalid_position"
        elif any(spec.name not in parameters for spec in component.model.parameter_schema):
            reason = "missing_model_parameter"
        elif any(not isfinite(value) for value in parameters.values()):
            reason = "nonfinite_model_parameter"
        elif "sig" in parameters and parameters["sig"] <= 0:
            reason = "invalid_sig"
        if reason:
            excluded.append(ExcludedComponent(component.id_, region_id, reason))
            continue
        try:
            area = peak_area(component)
        except (ValueError, ArithmeticError):
            excluded.append(ExcludedComponent(component.id_, region_id, "invalid_area"))
            continue
        if area is not None and not isfinite(area):
            excluded.append(ExcludedComponent(component.id_, region_id, "invalid_area"))
            continue
        sig = parameters.get("sig")
        fwhm = 2 * sig if component.model.name == "pseudo-voigt" and sig is not None else None
        if fwhm is not None and not isfinite(fwhm):
            excluded.append(ExcludedComponent(component.id_, region_id, "invalid_fwhm"))
            continue
        peaks.append(
            PeakDescriptor(
                component_id=component.id_,
                region_id=region_id,
                model_name=component.model.name,
                position=parameters["cen"],
                area=area,
                fwhm=fwhm,
                sig=sig,
                shape_gl=parameters.get("frac"),
            )
        )
    return SeriesStep(spectrum_id, tuple(peaks), tuple(excluded))
