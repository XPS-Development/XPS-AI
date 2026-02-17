"""
Change objects: immutable DTO-like outputs from application services.

Application services produce Change instances when they need to affect
the core data model. Changes describe "what" to do; they are turned
into Command objects by CommandRegistry for execution and undo/redo.
"""

from dataclasses import dataclass

from typing import Literal, Union, Optional
from numpy.typing import NDArray
import numpy as np

from core.metadata import Metadata


# Type alias for parameter field names; maps to RuntimeParameter attributes.
ParameterField = Literal["name", "value", "lower", "upper", "vary", "expr"]


@dataclass(frozen=True)
class UpdateParameter:
    """
    Change to a single attribute of a component's runtime parameter.

    Maps to RuntimeParameter attributes and ComponentService.set_parameter(**kwargs).
    """

    component_id: str
    name: str
    parameter_field: ParameterField
    new_value: str | bool | float


@dataclass(frozen=True)
class UpdateRegionSlice:
    """Change to update the index slice of an existing region."""

    region_id: str
    start: int
    stop: int


@dataclass(frozen=True)
class RemoveObject:
    """
    Change to remove an object from the collection.

    Maps to CollectionQueryService.detach() via RemoveObjectCommand.
    """

    obj_id: str


@dataclass(frozen=True)
class RemoveMetadata:
    """Change to remove metadata for an object by ID."""

    obj_id: str


@dataclass(frozen=True)
class FullRemoveObject:
    """
    Change to remove an object and its metadata from both collection and metadata store.

    Cascades to children: removes metadata for all descendants with metadata, then removes
    the object subtree from the collection.
    """

    obj_id: str


@dataclass(frozen=True)
class UpdateMultipleParameterValues:
    """
    Change to update values of multiple parameters at once.

    Maps to ComponentService.set_values().
    Convenience change for batch parameter updates.
    """

    component_id: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class CreateSpectrum:
    """
    Change to create a new spectrum.

    Maps to SpectrumService.create_spectrum().
    """

    x: NDArray[np.floating]
    y: NDArray[np.floating]
    spectrum_id: Optional[str] = None


@dataclass(frozen=True)
class CreateRegion:
    """
    Change to create a new region bound to a spectrum.

    Maps to RegionService.create_region().
    """

    spectrum_id: str
    start: int
    stop: int
    region_id: Optional[str] = None


@dataclass(frozen=True)
class CreatePeak:
    """
    Change to create a new peak component.

    Maps to ComponentService.create_peak().
    """

    region_id: str
    model_name: str
    parameters: Optional[dict[str, float]] = None
    peak_id: Optional[str] = None


@dataclass(frozen=True)
class CreateBackground:
    """
    Change to create or replace a background component.

    Maps to ComponentService.replace_background().
    If a background already exists, it will be replaced.
    """

    region_id: str
    model_name: str
    parameters: Optional[dict[str, float]] = None
    background_id: Optional[str] = None


@dataclass(frozen=True)
class ReplacePeakModel:
    """
    Change to replace a peak's model and update its parameters accordingly.

    This effectively removes the old peak and creates a new one with
    the new model. The old peak's ID is preserved if possible.
    Maps to ComponentService.remove_component() + ComponentService.create_peak().
    """

    peak_id: str
    new_model_name: str
    parameters: Optional[dict[str, float]] = None


@dataclass(frozen=True)
class ReplaceBackgroundModel:
    """
    Change to replace a background's model and update its parameters accordingly.

    Identifies the background by region. Maps to ComponentService.replace_background().
    """

    region_id: str
    new_model_name: str
    parameters: Optional[dict[str, float]] = None
    background_id: Optional[str] = None


@dataclass(frozen=True)
class SetMetadata:
    """
    Change to set metadata for an object.

    Maps to MetadataService.set_metadata().
    """

    obj_id: str
    metadata: Metadata


@dataclass(frozen=True)
class CompositeChange:
    """
    Change that groups multiple changes for batch execution.

    Handled specially by CommandRegistry to build a CompositeCommand.
    """

    changes: list["BaseChange"]


# Union type for typed dispatch in CommandRegistry and CommandExecutor.
BaseChange = Union[
    UpdateParameter,
    UpdateRegionSlice,
    RemoveObject,
    RemoveMetadata,
    FullRemoveObject,
    CreateSpectrum,
    CreateRegion,
    CreatePeak,
    CreateBackground,
    ReplacePeakModel,
    ReplaceBackgroundModel,
    UpdateMultipleParameterValues,
    SetMetadata,
    CompositeChange,
]
