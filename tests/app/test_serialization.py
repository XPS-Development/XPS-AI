"""Tests for app SerializationService."""

from pathlib import Path

import pytest

from app.serialization import SerializationService
from core.services import MetadataService


def test_is_saved_initially_false_without_path(simple_collection):
    """Without default path, is_saved is False even when not dirty."""
    metadata_service = MetadataService(simple_collection)
    manager = SerializationService(simple_collection, metadata_service)
    assert manager.is_saved is False


def test_is_saved_after_set_path_and_no_changes(simple_collection):
    """With default path set and no dirty, is_saved is True."""
    metadata_service = MetadataService(simple_collection)
    manager = SerializationService(simple_collection, metadata_service)
    manager.set_default_path("/some/path.json")
    assert manager.is_saved is True


def test_mark_dirty_makes_is_saved_false(simple_collection, tmp_path):
    """After mark_dirty, is_saved is False when path is set."""
    metadata_service = MetadataService(simple_collection)
    manager = SerializationService(simple_collection, metadata_service)
    manager.set_default_path(tmp_path / "x.json")
    manager.dump()
    assert manager.is_saved is True
    manager.mark_dirty()
    assert manager.is_saved is False


def test_set_default_path_get_default_path(simple_collection):
    """set_default_path and get_default_path round-trip."""
    metadata_service = MetadataService(simple_collection)
    manager = SerializationService(simple_collection, metadata_service)
    assert manager.get_default_path() is None
    path = Path("/foo/bar.json")
    manager.set_default_path(path)
    assert manager.get_default_path() == path


def test_dump_sets_path_and_clears_dirty(simple_collection, tmp_path):
    """dump with explicit path sets default path and clears dirty."""
    metadata_service = MetadataService(simple_collection)
    manager = SerializationService(simple_collection, metadata_service)
    manager.mark_dirty()
    fp = tmp_path / "out.json"
    manager.dump(path=fp)
    assert manager.get_default_path() == fp
    assert manager.is_saved is True
    assert fp.exists()


def test_dump_without_path_uses_default(simple_collection, tmp_path):
    """dump with no path uses default path."""
    metadata_service = MetadataService(simple_collection)
    manager = SerializationService(simple_collection, metadata_service)
    fp = tmp_path / "default.json"
    manager.set_default_path(fp)
    manager.dump()
    assert fp.exists()


def test_dump_without_path_and_no_default_raises(simple_collection):
    """dump with no path and no default raises ValueError."""
    metadata_service = MetadataService(simple_collection)
    manager = SerializationService(simple_collection, metadata_service)
    with pytest.raises(ValueError, match="No path provided"):
        manager.dump()


def test_load_replace_updates_path_and_clears_dirty(simple_collection, tmp_path):
    """load with replace sets default path and clears dirty."""
    metadata_service = MetadataService(simple_collection)
    manager = SerializationService(simple_collection, metadata_service)
    fp = tmp_path / "data.json"
    manager.dump(path=fp)
    manager.mark_dirty()
    manager.load(fp, mode="replace")
    assert manager.get_default_path() == fp
    assert manager.is_saved is True


def test_load_new_returns_tuple(simple_collection, tmp_path):
    """load with mode=new returns (collection, metadata_service)."""
    metadata_service = MetadataService(simple_collection)
    manager = SerializationService(simple_collection, metadata_service)
    fp = tmp_path / "data.json"
    manager.dump(path=fp)
    result = manager.load(fp, mode="new")
    assert result is not None
    new_collection, new_metadata_service = result
    assert new_collection is not simple_collection
    assert len(new_collection.objects_index) == len(simple_collection.objects_index)
    assert manager.get_default_path() == fp
    assert manager.is_saved is True
