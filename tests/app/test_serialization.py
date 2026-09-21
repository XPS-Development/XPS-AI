"""Tests for app SerializationService."""

from app.serialization import SerializationService
from core.collection import CoreCollection
from core.services import MetadataService


def test_dump_writes_file(simple_collection, tmp_path):
    """dump writes JSON file."""
    metadata_service = MetadataService(simple_collection)
    service = SerializationService()
    path = tmp_path / "out.json"

    service.dump(path=path, collection=simple_collection, metadata_service=metadata_service)

    assert path.exists()


def test_dump_gzip_roundtrip(simple_collection, tmp_path):
    """dump with use_gzip writes gzip JSON; load restores collection."""
    metadata_service = MetadataService(simple_collection)
    service = SerializationService()
    path = tmp_path / "out.json.gz"

    service.dump(
        path=path,
        collection=simple_collection,
        metadata_service=metadata_service,
        use_gzip=True,
        compresslevel=3,
    )
    assert path.exists()
    with path.open("rb") as f:
        assert f.read(2) == b"\x1f\x8b"

    target_collection = CoreCollection()
    target_metadata = MetadataService(target_collection)
    service.load(
        path=path,
        collection=target_collection,
        metadata_service=target_metadata,
        mode="replace",
    )
    assert len(target_collection.objects_index) == len(simple_collection.objects_index)


def test_load_replace_restores_collection(simple_collection, tmp_path):
    """load with replace loads data into the target collection."""
    metadata_service = MetadataService(simple_collection)
    service = SerializationService()
    path = tmp_path / "data.json"

    service.dump(path=path, collection=simple_collection, metadata_service=metadata_service)

    target_collection = CoreCollection()
    target_metadata = MetadataService(target_collection)

    service.load(
        path=path,
        collection=target_collection,
        metadata_service=target_metadata,
        mode="replace",
    )

    assert len(target_collection.objects_index) == len(simple_collection.objects_index)


def test_load_append_preserves_existing_objects(simple_collection, tmp_path):
    """load with append adds objects without clearing existing ones."""
    metadata_service = MetadataService(simple_collection)
    service = SerializationService()
    path = tmp_path / "data.json"

    service.dump(path=path, collection=simple_collection, metadata_service=metadata_service)

    base_collection = CoreCollection()
    base_metadata = MetadataService(base_collection)

    service.load(
        path=path,
        collection=base_collection,
        metadata_service=base_metadata,
        mode="append",
    )

    assert len(base_collection.objects_index) == len(simple_collection.objects_index)
