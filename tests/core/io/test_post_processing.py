import copy
from dataclasses import replace

import pytest

from core.collection import CoreCollection
from core.io.serialization import VERSION, DeserializeResult, deserialize, dump, load, serialize
from core.metadata import SpectrumMetadata
from core.post_processing import (
    MatchingSeries,
    MetricDefinition,
    PeakDescriptor,
    SeriesStep,
    match_series,
)
from core.post_processing.fingerprint import input_fingerprint
from core.post_processing.state import PostProcessingSeries
from core.services import MetadataService


def new_document(result: DeserializeResult) -> tuple[CoreCollection, MetadataService]:
    assert isinstance(result, tuple)
    return result


@pytest.fixture
def saved(simple_collection):
    definition = PostProcessingSeries(
        "pp",
        "Experiment",
        ("s1",),
        track_names=(("p1", "Main"),),
        metrics=(
            MetricDefinition(
                "shift", "Shift", "position_shift", ("track/p1",), reference_spectrum_id="s1"
            ),
        ),
        column_order=("shift", "area"),
    )
    matching = MatchingSeries(
        "pp", (SeriesStep("s1", (PeakDescriptor("p1", "r1", "pseudo-voigt", 0, 1, 2, 1, 0),)),)
    )
    definition = replace(
        definition,
        result=match_series(matching),
        input_fingerprint=input_fingerprint(simple_collection, definition),
    )
    simple_collection.post_processing[definition.id_] = definition
    return simple_collection, definition


def test_series_full_round_trip_through_dict_and_json(saved, tmp_path):
    collection, definition = saved
    data = serialize(collection)
    assert data["version"] == "1.1"
    assert data["post_processing"][0]["metrics"][0]["reference_spectrum_id"] == "s1"
    restored, _ = new_document(deserialize(data, mode="new"))
    assert restored.post_processing == {"pp": definition}
    assert input_fingerprint(restored, definition) == definition.input_fingerprint
    path = tmp_path / "series.json"
    dump(collection, path)
    restored, _ = new_document(load(path, mode="new"))
    assert restored.post_processing == {"pp": definition}
    assert input_fingerprint(restored, definition) == definition.input_fingerprint


def test_gzip_round_trip_preserves_section(saved, tmp_path):
    collection, definition = saved
    path = tmp_path / "series.gz"
    dump(collection, path, use_gzip=True)
    restored, _ = new_document(load(path, mode="new"))
    assert restored.post_processing["pp"] == definition


def test_unsolved_series_and_unresolved_operands_round_trip(simple_collection):
    definition = PostProcessingSeries(
        "pp",
        "S",
        ("s1", "missing"),
        metrics=(MetricDefinition("m", "Area", "area", ("track/gone",)),),
    )
    simple_collection.post_processing["pp"] = definition
    restored, _ = new_document(deserialize(serialize(simple_collection), mode="new"))
    assert restored.post_processing["pp"] == definition


def test_stale_result_and_missing_source_are_retained(saved):
    collection, definition = saved
    collection.remove("s1")
    restored, _ = new_document(deserialize(serialize(collection), mode="new"))
    assert restored.post_processing["pp"] == definition
    assert input_fingerprint(restored, definition) != definition.input_fingerprint
    assert definition.result.series.steps[0].peaks[0].component_id == "p1"


def test_append_remaps_only_series_collisions_and_keeps_references(saved):
    collection, definition = saved
    data = serialize(collection)
    metadata = MetadataService(collection)
    deserialize(data, collection, metadata, mode="append")
    assert len(collection.objects_index) == 4
    assert len(collection.post_processing) == 2
    assert collection.post_processing["pp"] == definition
    imported = next(value for key, value in collection.post_processing.items() if key != "pp")
    assert imported.id_ != "pp"
    assert imported.result.series.id_ == imported.id_
    assert imported.spectrum_ids == definition.spectrum_ids
    assert imported.metrics == definition.metrics
    assert imported.track_names == definition.track_names
    assert imported.result.tracks == definition.result.tracks
    assert input_fingerprint(collection, imported) == imported.input_fingerprint


def test_append_keeps_stale_fingerprint_when_colliding_sources_differ(saved):
    collection, definition = saved
    data = serialize(collection)
    collection.get("p1").parameters["amp"].value = 123
    deserialize(data, collection, mode="append")
    assert len(collection.post_processing) == 2
    assert all(
        input_fingerprint(collection, s) != s.input_fingerprint
        for s in collection.post_processing.values()
    )
    assert definition.result.series.steps[0].peaks[0].area == 1


def test_replace_with_v1_document_clears_series(saved):
    collection, _ = saved
    data = serialize(collection)
    data["version"] = "1.0"
    del data["post_processing"]
    deserialize(data, collection, MetadataService(collection), mode="replace")
    assert collection.post_processing == {}
    assert len(collection.objects_index) == 4


def test_append_v1_document_preserves_existing_series(saved):
    collection, definition = saved
    deserialize({"version": "1.0", "objects": []}, collection, mode="append")
    assert collection.post_processing == {"pp": definition}


def test_v1_cannot_silently_contain_series(saved):
    collection, _ = saved
    data = serialize(collection)
    data["version"] = "1.0"
    with pytest.raises(ValueError, match="cannot contain"):
        deserialize(data, mode="new")


@pytest.mark.parametrize(
    "broken", ["section", "duplicate", "result", "tracks", "metric", "objects"]
)
@pytest.mark.parametrize("mode", ["replace", "append"])
def test_invalid_document_leaves_target_and_metadata_untouched(saved, broken, mode):
    collection, definition = saved
    metadata = MetadataService(collection)
    original_metadata = SpectrumMetadata(name="Original", file="File", group="Group")
    metadata.set_metadata("s1", original_metadata)
    data = copy.deepcopy(serialize(collection))
    if broken == "section":
        data["post_processing"] = {}
    elif broken == "duplicate":
        data["post_processing"].append(data["post_processing"][0])
    elif broken == "result":
        data["post_processing"][0]["result"]["series"]["id_"] = "other"
    elif broken == "tracks":
        data["post_processing"][0]["result"]["tracks"] = []
    elif broken == "metric":
        data["post_processing"][0]["metrics"][0]["reference_spectrum_id"] = None
    else:
        data["objects"].append({"id": "bad", "type": "Unknown"})
    original_objects = dict(collection.objects_index)
    generation = collection.generation
    with pytest.raises(ValueError, match=r"Expected|Duplicate|Result|cover|requires|Unknown"):
        deserialize(data, collection, metadata, mode=mode)
    assert collection.objects_index == original_objects
    assert collection.post_processing == {"pp": definition}
    assert metadata.get_metadata("s1") == original_metadata
    assert collection.generation == generation


def test_new_v1_document_starts_without_series():
    restored, metadata = new_document(deserialize({"version": "1.0", "objects": []}, mode="new"))
    assert isinstance(restored, CoreCollection)
    assert restored.post_processing == {}
    assert metadata.collection is restored
    assert VERSION == "1.1"
