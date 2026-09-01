"""Tests for hierarchy use-cases."""

from app.command.changes import CompositeChange, FullRemoveObject, SetMetadata
from app.query_service import QueryService
from app.usecases.hierarchy import HierarchyUseCases
from core.metadata import SpectrumMetadata
from core.services import CoreContext


def _hierarchy_with_metadata(simple_collection) -> tuple[HierarchyUseCases, QueryService, str]:
    ctx = CoreContext.from_collection(simple_collection)
    query = QueryService(ctx)
    spectrum_id = query.get_all_spectra_ids()[0]
    ctx.metadata.set_metadata(
        spectrum_id,
        SpectrumMetadata(name="spec-1", group="group-a", file="file-a"),
    )
    return HierarchyUseCases(query), query, spectrum_id


def test_rename_spectrum_updates_metadata(simple_collection) -> None:
    """rename_spectrum builds a SetMetadata change with the new name."""
    hierarchy, _, spectrum_id = _hierarchy_with_metadata(simple_collection)

    change = hierarchy.rename_spectrum(spectrum_id, "Renamed")

    assert isinstance(change, SetMetadata)
    assert change.obj_id == spectrum_id
    assert isinstance(change.metadata, SpectrumMetadata)
    assert change.metadata.name == "Renamed"
    assert change.metadata.group == "group-a"
    assert change.metadata.file == "file-a"


def test_rename_group_returns_composite_change(simple_collection) -> None:
    """rename_group updates metadata for spectra in the target group."""
    hierarchy, _, _spectrum_id = _hierarchy_with_metadata(simple_collection)

    change = hierarchy.rename_group("file-a", "group-a", "group-b")

    assert isinstance(change, CompositeChange)
    assert len(change.changes) == 1
    assert all(isinstance(c, SetMetadata) for c in change.changes)


def test_remove_group_returns_full_remove_changes(simple_collection) -> None:
    """remove_group builds FullRemoveObject changes for group members."""
    hierarchy, _, _spectrum_id = _hierarchy_with_metadata(simple_collection)

    change = hierarchy.remove_group("file-a", "group-a")

    assert isinstance(change, CompositeChange)
    assert len(change.changes) == 1
    assert all(isinstance(c, FullRemoveObject) for c in change.changes)
