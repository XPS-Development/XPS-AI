"""
Tests for collection serialization and deserialization.
"""

import json

import numpy as np
import pytest

from core.collection import CoreCollection
from core.metadata import PeakMetadata, SpectrumMetadata
from core.objects import Background, Peak, Region, Spectrum
from core.services import MetadataService
from tools.serialization import VERSION, _json_default, deserialize, dump, load, serialize


def test_serialize_deserialize_simple_collection(simple_collection):
    """
    Test serialization and deserialization of a simple collection.

    Verifies that a collection can be serialized to a dictionary and then
    deserialized back to an identical collection structure.
    """
    # Serialize collection to dict
    data = serialize(simple_collection)
    assert data is not None
    assert "version" in data
    assert "objects" in data
    assert data["version"] == VERSION

    # Deserialize back (mode=new returns tuple)
    result = deserialize(data, mode="new")
    restored_collection = result[0]

    # Verify collection structure
    assert len(restored_collection.objects_index) == len(simple_collection.objects_index)

    # Verify all objects exist
    for obj_id in simple_collection.objects_index:
        assert obj_id in restored_collection.objects_index
        original_obj = simple_collection.objects_index[obj_id]
        restored_obj = restored_collection.objects_index[obj_id]

        # Verify object types match
        assert type(original_obj) == type(restored_obj)

        # Verify IDs match
        assert original_obj.id_ == restored_obj.id_
        assert restored_obj.id_ == obj_id

        # Verify parent relationships
        assert original_obj.parent_id == restored_obj.parent_id

        # Type-specific checks
        if isinstance(original_obj, Spectrum):
            assert np.allclose(original_obj.x, restored_obj.x, atol=1e-1)
            assert np.allclose(original_obj.y, restored_obj.y, atol=1e-1)
        elif isinstance(original_obj, Region):
            assert original_obj.slice_.start == restored_obj.slice_.start
            assert original_obj.slice_.stop == restored_obj.slice_.stop
        elif isinstance(original_obj, (Peak, Background)):
            assert original_obj.model.name == restored_obj.model.name
            # Verify parameters match
            for param_name in original_obj.parameters:
                orig_param = original_obj.parameters[param_name]
                rest_param = restored_obj.parameters[param_name]
                assert orig_param.value == rest_param.value
                assert orig_param.lower == rest_param.lower
                assert orig_param.upper == rest_param.upper
                assert orig_param.vary == rest_param.vary
                assert orig_param.expr == rest_param.expr


def test_serialize_deserialize_with_metadata(simple_collection):
    """
    Test serialization and deserialization with metadata.
    """
    # Create metadata service and add metadata
    metadata_service = MetadataService(simple_collection)
    spectrum_id = next(
        obj.id_ for obj in simple_collection.objects_index.values() if isinstance(obj, Spectrum)
    )
    peak_id = next(obj.id_ for obj in simple_collection.objects_index.values() if isinstance(obj, Peak))

    metadata_service.set_metadata(
        spectrum_id, SpectrumMetadata(name="Test", group="Group1", file="test.dat")
    )
    metadata_service.set_metadata(peak_id, PeakMetadata(element_type="C 1s"))

    # Serialize with metadata
    data = serialize(simple_collection, metadata_service=metadata_service)

    # Deserialize with metadata service (replace mode)
    restored_collection = CoreCollection()
    restored_metadata_service = MetadataService(restored_collection)
    deserialize(
        data,
        collection=restored_collection,
        metadata_service=restored_metadata_service,
        mode="replace",
    )

    # Verify metadata was restored
    restored_spectrum_metadata = restored_metadata_service.get_metadata(spectrum_id)
    assert restored_spectrum_metadata is not None
    assert restored_spectrum_metadata.name == "Test"
    assert restored_spectrum_metadata.group == "Group1"
    assert restored_spectrum_metadata.file == "test.dat"

    restored_peak_metadata = restored_metadata_service.get_metadata(peak_id)
    assert restored_peak_metadata is not None
    assert restored_peak_metadata.element_type == "C 1s"


def test_serialize_deserialize_empty_collection(empty_collection):
    """
    Test serialization and deserialization of empty collection.
    """
    data = serialize(empty_collection)
    result = deserialize(data, mode="new")
    restored_collection = result[0]

    assert len(restored_collection.objects_index) == 0
    assert data["version"] == VERSION
    assert data["objects"] == []


def test_append_to_existing_collection(simple_collection, empty_collection):
    """
    Test appending deserialized objects to an existing collection.
    """
    # Serialize simple_collection
    data = serialize(simple_collection)

    # Deserialize into empty collection (append mode)
    deserialize(data, collection=empty_collection, mode="append")

    assert len(empty_collection.objects_index) == len(simple_collection.objects_index)

    # Try to deserialize again (should skip duplicates)
    deserialize(data, collection=empty_collection, mode="append")

    # Should still have same number of objects (no duplicates)
    assert len(empty_collection.objects_index) == len(simple_collection.objects_index)


def test_version_compatibility():
    """
    Test version checking in deserialization.
    """
    # Create data with wrong version
    invalid_data = {"version": "2.0", "objects": []}

    # Should raise ValueError (mode=new to reach version check)
    with pytest.raises(ValueError, match="Version mismatch"):
        deserialize(invalid_data, mode="new")


def test_dump_and_load_file(simple_collection, tmp_path):
    """
    Test serialization to file and loading from file.
    """
    file_path = tmp_path / "test_collection.json"

    # Serialize to file
    dump(simple_collection, file_path)

    # Verify file exists
    assert file_path.exists()

    # Load from file (mode=new returns tuple)
    result = load(file_path, mode="new")
    restored_collection = result[0]

    # Verify collection structure
    assert len(restored_collection.objects_index) == len(simple_collection.objects_index)

    # Verify objects match
    for obj_id in simple_collection.objects_index:
        assert obj_id in restored_collection.objects_index
        original_obj = simple_collection.objects_index[obj_id]
        restored_obj = restored_collection.objects_index[obj_id]
        assert type(original_obj) == type(restored_obj)
        assert original_obj.id_ == restored_obj.id_


def test_dump_and_load_with_metadata(simple_collection, tmp_path):
    """
    Test serialization to file and loading from file with metadata.
    """
    file_path = tmp_path / "test_collection.json"

    # Create metadata service and add metadata
    metadata_service = MetadataService(simple_collection)
    spectrum_id = next(
        obj.id_ for obj in simple_collection.objects_index.values() if isinstance(obj, Spectrum)
    )
    peak_id = next(obj.id_ for obj in simple_collection.objects_index.values() if isinstance(obj, Peak))

    metadata_service.set_metadata(
        spectrum_id, SpectrumMetadata(name="Test", group="Group1", file="test.dat")
    )
    metadata_service.set_metadata(peak_id, PeakMetadata(element_type="C 1s"))

    # Serialize to file with metadata
    dump(simple_collection, file_path, metadata_service=metadata_service)

    # Load from file with metadata service (replace mode)
    restored_collection = CoreCollection()
    restored_metadata_service = MetadataService(restored_collection)
    load(
        file_path,
        collection=restored_collection,
        metadata_service=restored_metadata_service,
        mode="replace",
    )

    # Verify metadata was restored
    restored_spectrum_metadata = restored_metadata_service.get_metadata(spectrum_id)
    assert restored_spectrum_metadata is not None
    assert restored_spectrum_metadata.name == "Test"
    assert restored_spectrum_metadata.group == "Group1"
    assert restored_spectrum_metadata.file == "test.dat"

    restored_peak_metadata = restored_metadata_service.get_metadata(peak_id)
    assert restored_peak_metadata is not None
    assert restored_peak_metadata.element_type == "C 1s"


def test_dump_creates_directory(tmp_path):
    """
    Test that dump creates parent directories if they don't exist.
    """
    collection = CoreCollection()
    file_path = tmp_path / "nested" / "dir" / "test_collection.json"

    # Should create nested directories
    dump(collection, file_path)

    assert file_path.exists()
    assert file_path.parent.exists()


def test_serialize_spectrum_x_axis_optimization(simple_collection):
    """
    Test that spectrum x-axis is serialized efficiently using start, step, num_points.
    """
    data = serialize(simple_collection)

    # Find spectrum object in serialized data
    spectrum_data = next(obj for obj in data["objects"] if obj["type"] == "Spectrum")

    # Verify x-axis is stored as start, step, num_points (not full array)
    assert "x_start" in spectrum_data
    assert "x_step" in spectrum_data
    assert "x_num_points" in spectrum_data
    assert "x" not in spectrum_data  # Full x array should not be stored

    # Verify y array is stored
    assert "y" in spectrum_data
    assert isinstance(spectrum_data["y"], list)


def test_deserialize_spectrum_reconstructs_x_axis(simple_collection):
    """
    Test that deserialized spectrum correctly reconstructs x-axis from start, step, num_points.
    """
    data = serialize(simple_collection)

    # Get original spectrum
    original_spectrum = next(
        obj for obj in simple_collection.objects_index.values() if isinstance(obj, Spectrum)
    )

    # Deserialize (mode=new returns tuple)
    result = deserialize(data, mode="new")
    restored_collection = result[0]
    restored_spectrum = next(
        obj for obj in restored_collection.objects_index.values() if isinstance(obj, Spectrum)
    )

    # Verify x-axis is correctly reconstructed
    assert np.allclose(original_spectrum.x, restored_spectrum.x, atol=1e-1)
    assert len(original_spectrum.x) == len(restored_spectrum.x)


def test_deserialize_missing_parent_raises_error():
    """
    Test that deserializing a region/component with missing parent raises ValueError.
    """
    # Create data with region referencing non-existent parent
    invalid_data = {
        "version": VERSION,
        "objects": [
            {
                "id": "r1",
                "parent_id": "nonexistent",
                "type": "Region",
                "slice_start": 0,
                "slice_stop": 10,
                "metadata": None,
            }
        ],
    }

    with pytest.raises(ValueError, match="references non-existent parent"):
        deserialize(invalid_data, mode="new")


def test_deserialize_missing_id_raises_error():
    """
    Test that deserializing an object without id raises ValueError.
    """
    invalid_data = {
        "version": VERSION,
        "objects": [
            {
                "parent_id": None,
                "type": "Spectrum",
                "x_start": 0.0,
                "x_step": 1.0,
                "x_num_points": 10,
                "y": [1.0] * 10,
                "metadata": None,
            }
        ],
    }

    with pytest.raises(ValueError, match="missing 'id' field"):
        deserialize(invalid_data, mode="new")


def test_deserialize_unknown_type_raises_error():
    """
    Test that deserializing an unknown object type raises ValueError.
    """
    invalid_data = {
        "version": VERSION,
        "objects": [
            {
                "id": "obj1",
                "parent_id": None,
                "type": "UnknownType",
                "metadata": None,
            }
        ],
    }

    with pytest.raises(ValueError, match="Unknown object type"):
        deserialize(invalid_data, mode="new")


def test_serialize_without_metadata_service(simple_collection):
    """
    Test that serialization works without metadata service (metadata should be None).
    """
    data = serialize(simple_collection)

    # All objects should have metadata: None
    for obj_data in data["objects"]:
        assert obj_data.get("metadata") is None


def test_deserialize_without_metadata_service(simple_collection):
    """
    Test that deserialization works without metadata service.
    """
    data = serialize(simple_collection)

    # Deserialize without metadata service (mode=new returns tuple)
    result = deserialize(data, mode="new")
    restored_collection = result[0]

    assert len(restored_collection.objects_index) == len(simple_collection.objects_index)


def test_json_serialization_inf_nan_handling():
    """
    Test that JSON serialization handles inf and nan values correctly.
    """
    # Create a collection with a peak that has inf/nan bounds
    collection = CoreCollection()
    x = np.linspace(0, 10, 100)
    y = np.linspace(1, 2, 100)  # Use varying values to avoid zero scale
    spectrum = Spectrum(x=x, y=y, id_="s1")
    collection.add(spectrum)

    region = Region(slice(0, 100), parent_id="s1", id_="r1")
    collection.add(region)

    from core.math_models import PseudoVoigtPeakModel

    peak = Peak(model=PseudoVoigtPeakModel(), region_id="r1", component_id="p1", amp=1, cen=5, sig=1, frac=0)
    # Set parameter with inf bounds
    param = peak.get_param("amp")
    param.set(lower=-np.inf, upper=np.inf)
    collection.add(peak)

    # Serialize and deserialize
    data = serialize(collection)
    json_str = json.dumps(data, default=_json_default)

    # Should not raise error
    data_restored = json.loads(json_str)
    result = deserialize(data_restored, mode="new")
    restored_collection = result[0]

    # Verify parameter bounds are restored correctly
    restored_peak = next(obj for obj in restored_collection.objects_index.values() if isinstance(obj, Peak))
    amp_param = restored_peak.get_param("amp")
    assert amp_param.lower == -np.inf
    assert amp_param.upper == np.inf


def test_deserialize_mode_replace_clears_and_fills_in_place(simple_collection, tmp_path):
    """
    mode=replace clears existing collection and metadata, then deserializes into same refs.
    """
    metadata_service = MetadataService(simple_collection)
    spectrum_id = next(
        obj.id_ for obj in simple_collection.objects_index.values() if isinstance(obj, Spectrum)
    )
    metadata_service.set_metadata(spectrum_id, SpectrumMetadata(name="Before", group="G", file="f"))

    file_path = tmp_path / "coll.json"
    dump(simple_collection, file_path, metadata_service=metadata_service)

    # Replace: clear and load into same collection/metadata
    collection_ref = simple_collection
    metadata_ref = metadata_service
    result = deserialize(
        serialize(simple_collection, metadata_service),
        collection=simple_collection,
        metadata_service=metadata_service,
        mode="replace",
    )
    assert result is collection_ref
    assert len(collection_ref.objects_index) == len(simple_collection.objects_index)
    assert metadata_ref.get_metadata(spectrum_id) is not None


def test_deserialize_mode_new_returns_tuple(simple_collection):
    """mode=new with no collection/metadata creates both and returns (collection, metadata_service)."""
    data = serialize(simple_collection)
    result = deserialize(data, mode="new")
    assert isinstance(result, tuple)
    new_collection, new_metadata_service = result
    assert new_collection is not simple_collection
    assert len(new_collection.objects_index) == len(simple_collection.objects_index)
    assert new_metadata_service.collection is new_collection


def test_load_mode_new_returns_tuple(simple_collection, tmp_path):
    """load(..., mode='new') returns (collection, metadata_service)."""
    file_path = tmp_path / "coll.json"
    dump(simple_collection, file_path)
    result = load(file_path, mode="new")
    assert isinstance(result, tuple)
    new_collection, new_metadata_service = result
    assert len(new_collection.objects_index) == len(simple_collection.objects_index)


def test_deserialize_replace_requires_collection_and_metadata():
    """mode=replace raises if collection or metadata_service is missing."""
    data = {"version": VERSION, "objects": []}
    with pytest.raises(ValueError, match="mode='replace' requires both"):
        deserialize(data, collection=CoreCollection(), mode="replace")
    with pytest.raises(ValueError, match="mode='replace' requires both"):
        deserialize(
            data,
            metadata_service=MetadataService(CoreCollection()),
            mode="replace",
        )


def test_deserialize_new_requires_none():
    """mode=new raises if collection or metadata_service is provided."""
    data = {"version": VERSION, "objects": []}
    with pytest.raises(ValueError, match="mode='new' requires"):
        deserialize(data, collection=CoreCollection(), mode="new")
    with pytest.raises(ValueError, match="mode='new' requires"):
        deserialize(
            data,
            metadata_service=MetadataService(CoreCollection()),
            mode="new",
        )
