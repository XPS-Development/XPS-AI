"""
Change objects: immutable DTO-like outputs from application services.

Application services produce Change instances when they need to affect
the core data model. Changes describe "what" to do; they are turned
into Command objects by CommandRegistry for execution and undo/redo.
"""

from dataclasses import dataclass

from typing import Literal, Union


# Type alias for any change; mapper and executor accept BaseChange.
ParameterField = Literal["name", "value", "lower", "upper", "vary", "expr"]


@dataclass(frozen=True)
class ParameterChange:
    """
    Change to a single attribute of a component's runtime parameter.

    Maps to RuntimeParameter attributes and ComponentService.set_parameter(**kwargs).
    """

    component_id: str
    name: str
    parameter_field: ParameterField
    new_value: str | bool | float


@dataclass(frozen=True)
class SetParameterValue:
    """
    Change that sets only the value of a component parameter.

    Convenience change for the common case; equivalent to
    ParameterChange(..., parameter_field="value", new_value=...).
    """

    component_id: str
    name: str
    new_value: float


@dataclass(frozen=True)
class UpdateRegionSlice:
    """Change to update the index slice of an existing region."""

    region_id: str
    start: int
    stop: int


@dataclass(frozen=True)
class RemoveComponent:
    """
    Change to remove a component from the collection.

    Undo requires a snapshot of the component (built by CommandRegistry from context).
    """

    component_id: str


# Union type for typed dispatch in CommandRegistry and CommandExecutor.
BaseChange = Union[
    ParameterChange,
    SetParameterValue,
    UpdateRegionSlice,
    RemoveComponent,
]
