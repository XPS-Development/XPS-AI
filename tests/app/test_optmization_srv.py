import numpy as np
import pytest

from app.optimization import (
    OptimizationService,
    components_to_changes,
)
from core.services import CoreContext
from app.command.changes import UpdateMultipleParameterValues, CompositeChange
from app.command.core import CommandExecutor, UndoRedoStack, create_default_registry
from tools.dto import DTOService
from tools.optimization import OptimizedComponent


@pytest.fixture
def srv():
    return OptimizationService()


@pytest.fixture
def dto_service(simple_collection):
    return DTOService(simple_collection)


def test_components_to_changes():
    """components_to_changes produces UpdateMultipleParameterValues from OptimizedComponent."""
    components = [
        OptimizedComponent(component_id="p1", parameters={"amp": 1.0, "cen": 5.0}, normalized=False),
        OptimizedComponent(component_id="b1", parameters={"const": 2.0}, normalized=False),
    ]
    result = components_to_changes(components)

    assert isinstance(result, CompositeChange)
    assert len(result.changes) == 2
    assert result.changes[0].component_id == "p1"
    assert result.changes[0].parameters == {"amp": 1.0, "cen": 5.0}
    assert result.changes[1].component_id == "b1"
    assert result.changes[1].parameters == {"const": 2.0}


def test_optimize_regions_returns_composite_change(srv, dto_service, region_id):
    """optimize_regions returns CompositeChange with UpdateMultipleParameterValues."""
    region_reprs = [dto_service.get_region_repr(region_id, normalized=True)]
    change = srv.optimize_regions(region_reprs, method="least_squares")

    assert isinstance(change, CompositeChange)
    assert len(change.changes) > 0
    assert all(isinstance(c, UpdateMultipleParameterValues) for c in change.changes)


def test_optimize_regions_changes_apply_via_command_executor(
    srv, dto_service, simple_collection, region_id, peak_id
):
    """Execute returned changes via CommandExecutor and verify parameter updates."""
    region_reprs = [dto_service.get_region_repr(region_id, normalized=True)]
    change = srv.optimize_regions(region_reprs, method="least_squares")

    assert len(change.changes) > 0

    ctx = CoreContext.from_collection(simple_collection)
    stack = UndoRedoStack()
    executor = CommandExecutor(ctx, stack, create_default_registry())

    executor.execute(change)

    comp_srv = ctx.component
    params = comp_srv.get_parameters(peak_id, normalized=True)
    assert len(params) > 0
    for p in params.values():
        assert np.isfinite(p["value"])
