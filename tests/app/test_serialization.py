"""Tests for app SerializationService."""

from pathlib import Path

from app.serialization import SerializationService
from core.collection import CoreCollection
from core.services import MetadataService


def test_initial_state_not_dirty():
    """New service starts not dirty."""
    service = SerializationService()
    assert service.is_dirty is False


def test_mark_dirty_sets_flag():
    """mark_dirty sets the dirty flag."""
    service = SerializationService()
    assert service.is_dirty is False
    service.mark_dirty()
    assert service.is_dirty is True


def test_dump_clears_dirty_and_writes_file(simple_collection, tmp_path):
    """dump writes JSON file and clears dirty flag."""
    metadata_service = MetadataService(simple_collection)
    service = SerializationService()
    service.mark_dirty()
    path = tmp_path / "out.json"

    service.dump(path=path, collection=simple_collection, metadata_service=metadata_service)

    assert path.exists()
    assert service.is_dirty is False


def test_load_replace_clears_dirty(simple_collection, tmp_path):
    """load with replace loads data and clears dirty flag."""
    metadata_service = MetadataService(simple_collection)
    service = SerializationService()
    path = tmp_path / "data.json"

    # First dump the current collection
    service.dump(path=path, collection=simple_collection, metadata_service=metadata_service)
    service.mark_dirty()

    # Prepare fresh collection/metadata to load into
    target_collection = CoreCollection()
    target_metadata = MetadataService(target_collection)

    service.load(
        path=path,
        collection=target_collection,
        metadata_service=target_metadata,
        mode="replace",
    )

    assert service.is_dirty is False
    assert len(target_collection.objects_index) == len(simple_collection.objects_index)


def test_load_append_preserves_existing_objects(simple_collection, tmp_path):
    """load with append adds objects without clearing existing ones."""
    metadata_service = MetadataService(simple_collection)
    service = SerializationService()
    path = tmp_path / "data.json"

    service.dump(path=path, collection=simple_collection, metadata_service=metadata_service)

    # Start with a collection that already has one object
    base_collection = CoreCollection()
    base_metadata = MetadataService(base_collection)

    # Append into base_collection
    service.load(
        path=path,
        collection=base_collection,
        metadata_service=base_metadata,
        mode="append",
    )

    assert len(base_collection.objects_index) == len(simple_collection.objects_index)
