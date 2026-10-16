"""Fingerprint decomposition inputs independently of names and view state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from core.objects import Component, Region, Spectrum

if TYPE_CHECKING:
    from core.collection import CoreCollection
    from core.objects import CoreObject
    from core.post_processing.state import PostProcessingSeries


def input_fingerprint(collection: CoreCollection, series: PostProcessingSeries) -> str:
    """Hash spectrum order, region bounds, physical fit parameters and settings.

    Include all model parameters, even features not consumed by the matcher, so
    changing an asymmetric profile invalidates its old descriptor snapshot.
    Names and table order do not affect matching. Missing spectra remain steps.
    """
    children: dict[str | None, list[CoreObject]] = {}
    for obj in sorted(collection.objects_index.values(), key=lambda obj: obj.id_):
        children.setdefault(obj.parent_id, []).append(obj)
    spectra = []
    for spectrum_id in series.spectrum_ids:
        spectrum = collection.objects_index.get(spectrum_id)
        if not isinstance(spectrum, Spectrum):
            spectra.append((spectrum_id, None))
            continue
        regions = []
        for region in children.get(spectrum_id, []):
            if not isinstance(region, Region):
                continue
            components = [
                (
                    component.id_,
                    component.model.name,
                    type(component).__name__,
                    tuple(
                        (
                            name,
                            float(param.value),
                            float(param.lower),
                            float(param.upper),
                            param.vary,
                            param.expr,
                        )
                        for name, param in sorted(component.parameters.items())
                    ),
                )
                for component in children.get(region.id_, [])
                if isinstance(component, Component)
            ]
            regions.append((region.id_, region.slice_.start, region.slice_.stop, components))
        spectra.append((spectrum_id, regions))
    payload = (spectra, asdict(series.settings), asdict(series.constraints))
    # Existing document JSON supports infinite parameter bounds. Their canonical
    # spelling here is deterministic and is never used as a numerical input.
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()
