"""Tests for hierarchy use-cases."""

from app.command.changes import CompositeChange, FullRemoveObject, SetMetadata
from app.command.core import CommandExecutor, UndoRedoStack
from app.query_service import QueryService
from app.usecases.hierarchy import HierarchyUseCases
from core.metadata import SpectrumMetadata


def _hierarchy(hierarchy_ctx) -> tuple[HierarchyUseCases, QueryService]:
    query = QueryService(hierarchy_ctx)
    return HierarchyUseCases(query), query


def _apply_change(ctx, change) -> None:
    executor = CommandExecutor(ctx, UndoRedoStack())
    executor.execute(change)


def test_rename_spectrum_updates_metadata(hierarchy_ctx) -> None:
    """rename_spectrum builds a SetMetadata change with the new name."""
    hierarchy, _query = _hierarchy(hierarchy_ctx)

    change = hierarchy.rename_spectrum("s1", "Renamed")

    assert isinstance(change, SetMetadata)
    assert change.obj_id == "s1"
    assert isinstance(change.metadata, SpectrumMetadata)
    assert change.metadata.name == "Renamed"
    assert change.metadata.group == "group-a"
    assert change.metadata.file == "file-a"


def test_rename_group_returns_composite_change(hierarchy_ctx) -> None:
    """rename_group updates metadata for spectra in the target group."""
    hierarchy, _query = _hierarchy(hierarchy_ctx)

    change = hierarchy.rename_group("file-a", "group-a", "group-c")

    assert isinstance(change, CompositeChange)
    assert len(change.changes) == 2
    assert all(isinstance(c, SetMetadata) for c in change.changes)


def test_rename_file_returns_composite_change(hierarchy_ctx) -> None:
    """rename_file updates metadata.file for all spectra in the file bucket."""
    hierarchy, _query = _hierarchy(hierarchy_ctx)

    change = hierarchy.rename_file("file-a", "file-b")

    assert isinstance(change, CompositeChange)
    assert len(change.changes) == 3
    for sub in change.changes:
        assert isinstance(sub, SetMetadata)
        assert isinstance(sub.metadata, SpectrumMetadata)
        assert sub.metadata.file == "file-b"


def test_rename_file_returns_none_when_no_match(hierarchy_ctx) -> None:
    """rename_file returns None when no spectra match the file label."""
    hierarchy, _query = _hierarchy(hierarchy_ctx)

    assert hierarchy.rename_file("missing-file", "file-b") is None


def test_rename_group_returns_none_when_no_match(hierarchy_ctx) -> None:
    """rename_group returns None when no spectra match the file/group bucket."""
    hierarchy, _query = _hierarchy(hierarchy_ctx)

    assert hierarchy.rename_group("missing-file", "group-a", "group-c") is None
    assert hierarchy.rename_group("file-a", "missing-group", "group-c") is None


def test_rename_group_only_updates_matching_spectra(hierarchy_ctx) -> None:
    """rename_group applies only to spectra in the target file/group."""
    hierarchy, query = _hierarchy(hierarchy_ctx)

    change = hierarchy.rename_group("file-a", "group-a", "group-c")
    assert change is not None
    _apply_change(hierarchy_ctx, change)

    for spectrum_id in ("s1", "s2"):
        metadata = query.get_metadata(spectrum_id)
        assert isinstance(metadata, SpectrumMetadata)
        assert metadata.group == "group-c"

    metadata_s3 = query.get_metadata("s3")
    assert isinstance(metadata_s3, SpectrumMetadata)
    assert metadata_s3.group == "group-b"


def test_rename_file_updates_all_spectra_in_file(hierarchy_ctx) -> None:
    """rename_file updates every spectrum that shares the file label."""
    hierarchy, query = _hierarchy(hierarchy_ctx)

    change = hierarchy.rename_file("file-a", "file-renamed")
    assert change is not None
    _apply_change(hierarchy_ctx, change)

    for spectrum_id in ("s1", "s2", "s3"):
        metadata = query.get_metadata(spectrum_id)
        assert isinstance(metadata, SpectrumMetadata)
        assert metadata.file == "file-renamed"


def test_rename_spectrum_without_metadata_uses_defaults(hierarchy_collection) -> None:
    """rename_spectrum falls back to empty group/file when metadata is missing."""
    from core.services import CoreContext

    ctx = CoreContext.from_collection(hierarchy_collection)
    hierarchy = HierarchyUseCases(QueryService(ctx))

    change = hierarchy.rename_spectrum("s1", "OnlyName")

    assert isinstance(change, SetMetadata)
    assert change.metadata == SpectrumMetadata(name="OnlyName", group="", file="")


def test_remove_group_returns_full_remove_changes(hierarchy_ctx) -> None:
    """remove_group builds FullRemoveObject changes for group members."""
    hierarchy, _query = _hierarchy(hierarchy_ctx)

    change = hierarchy.remove_group("file-a", "group-a")

    assert isinstance(change, CompositeChange)
    assert len(change.changes) == 2
    assert all(isinstance(c, FullRemoveObject) for c in change.changes)
