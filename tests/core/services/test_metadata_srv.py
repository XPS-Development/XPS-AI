"""Tests for MetadataService."""

import numpy as np
import pytest

from core.services import MetadataService, SpectrumService
from core.metadata import SpectrumMetadata, RegionMetadata, PeakMetadata


@pytest.fixture
def srv(simple_collection):
    return MetadataService(simple_collection)


def test_get_metadata_returns_none_when_empty(srv, spectrum_id):
    """get_metadata returns None when no metadata stored."""
    md = srv.get_metadata(spectrum_id)
    assert md is None


def test_set_and_get_metadata_roundtrip(srv, spectrum_id):
    """set_metadata and get_metadata round-trip."""
    metadata = SpectrumMetadata(name="Sample 1", group="Group A", file="/path/to/file.vms")
    srv.set_metadata(spectrum_id, metadata)
    assert srv.get_metadata(spectrum_id) == metadata


def test_get_region_metadata_returns_none_when_empty(srv, region_id):
    """get_metadata returns None when no region metadata stored."""
    md = srv.get_metadata(region_id)
    assert md is None


def test_set_and_get_region_metadata_roundtrip(srv, region_id):
    """set_metadata and get_metadata round-trip for region."""
    metadata = RegionMetadata()
    srv.set_metadata(region_id, metadata)
    assert srv.get_metadata(region_id) == metadata


def test_get_peak_metadata_returns_none_when_empty(srv, peak_id):
    """get_metadata returns None when no peak metadata stored."""
    md = srv.get_metadata(peak_id)
    assert md is None


def test_set_and_get_peak_metadata_roundtrip(srv, peak_id):
    """set_metadata and get_metadata round-trip for peak."""
    metadata = PeakMetadata(element_type="Fe 2p")
    srv.set_metadata(peak_id, metadata)
    assert srv.get_metadata(peak_id) == metadata


def test_set_metadata_raises_key_error_for_unknown_id(srv):
    """set_metadata raises KeyError for non-existent object."""
    metadata = SpectrumMetadata(name="x", group="y", file="z")
    with pytest.raises(KeyError):
        srv.set_metadata("nonexistent-spectrum-id", metadata)


def test_metadata_persists_across_multiple_objects(srv, spectrum_id, region_id, peak_id):
    """Metadata for different object types is stored independently."""
    spec_md = SpectrumMetadata(name="Spec", group="G", file="f.vms")
    peak_md = PeakMetadata(element_type="C 1s")

    srv.set_metadata(spectrum_id, spec_md)
    srv.set_metadata(peak_id, peak_md)

    assert srv.get_metadata(spectrum_id) == spec_md
    assert srv.get_metadata(region_id) is None
    assert srv.get_metadata(peak_id) == peak_md


def test_remove_metadata(srv, spectrum_id):
    """remove_metadata removes stored metadata."""
    metadata = SpectrumMetadata(name="x", group="y", file="z")
    srv.set_metadata(spectrum_id, metadata)
    assert srv.get_metadata(spectrum_id) == metadata

    srv.remove_metadata(spectrum_id)
    assert srv.get_metadata(spectrum_id) is None


def test_remove_metadata_idempotent(srv, spectrum_id):
    """remove_metadata is idempotent when no metadata exists."""
    srv.remove_metadata(spectrum_id)
    srv.remove_metadata(spectrum_id)
    assert srv.get_metadata(spectrum_id) is None


# Tests for find_objects method


def test_find_objects_exact_match(srv, spectrum_id):
    """find_objects with exact match returns matching object IDs."""
    metadata = SpectrumMetadata(name="Sample 1", group="Group A", file="/path/to/file.vms")
    srv.set_metadata(spectrum_id, metadata)

    result = srv.find_objects("name", "Sample 1", match_exact=True)
    assert result == (spectrum_id,)


def test_find_objects_exact_match_no_match(srv, spectrum_id):
    """find_objects with exact match returns empty tuple when no match."""
    metadata = SpectrumMetadata(name="Sample 1", group="Group A", file="/path/to/file.vms")
    srv.set_metadata(spectrum_id, metadata)

    result = srv.find_objects("name", "Sample 2", match_exact=True)
    assert result == ()


def test_find_objects_fuzzy_match_substring(srv, spectrum_id):
    """find_objects with fuzzy match finds substring matches."""
    metadata = SpectrumMetadata(name="Sample 1", group="Group A", file="/path/to/file.vms")
    srv.set_metadata(spectrum_id, metadata)

    result = srv.find_objects("name", "Sample", match_exact=False)
    assert result == (spectrum_id,)


def test_find_objects_fuzzy_match_case_insensitive(srv, spectrum_id):
    """find_objects fuzzy match is case-insensitive."""
    metadata = SpectrumMetadata(name="Sample 1", group="Group A", file="/path/to/file.vms")
    srv.set_metadata(spectrum_id, metadata)

    result = srv.find_objects("name", "sample", match_exact=False)
    assert result == (spectrum_id,)

    result = srv.find_objects("name", "SAMPLE", match_exact=False)
    assert result == (spectrum_id,)


def test_find_objects_fuzzy_match_partial(srv, spectrum_id):
    """find_objects fuzzy match finds partial matches."""
    metadata = SpectrumMetadata(name="Sample 1", group="Group A", file="/path/to/file.vms")
    srv.set_metadata(spectrum_id, metadata)

    result = srv.find_objects("name", "amp", match_exact=False)
    assert result == (spectrum_id,)


def test_find_objects_multiple_objects(srv, spectrum_id, simple_collection):
    """find_objects returns multiple matching object IDs."""
    # Create additional spectrum in the same collection
    from core.services import SpectrumService

    spec_srv = SpectrumService(simple_collection)
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    spec2_id = spec_srv.create_spectrum(x, y, spectrum_id="s2")

    metadata1 = SpectrumMetadata(name="Sample 1", group="Group A", file="/path/to/file1.vms")
    metadata2 = SpectrumMetadata(name="Sample 2", group="Group A", file="/path/to/file2.vms")

    srv.set_metadata(spectrum_id, metadata1)
    srv.set_metadata(spec2_id, metadata2)

    result = srv.find_objects("group", "Group A", match_exact=True)
    assert len(result) == 2
    assert spectrum_id in result
    assert spec2_id in result


def test_find_objects_type_filter_spectrum(srv, spectrum_id, peak_id):
    """find_objects filters by metadata type (SpectrumMetadata)."""
    spec_metadata = SpectrumMetadata(name="Sample 1", group="Group A", file="/path/to/file.vms")
    peak_metadata = PeakMetadata(element_type="Fe 2p")

    srv.set_metadata(spectrum_id, spec_metadata)
    srv.set_metadata(peak_id, peak_metadata)

    # Search for "Sample" should only return spectrum, not peak
    result = srv.find_objects("name", "Sample", match_exact=False, tp=SpectrumMetadata)
    assert result == (spectrum_id,)

    # Search without type filter should return empty (peak doesn't have "name" field)
    result = srv.find_objects("name", "Sample", match_exact=False)
    assert result == (spectrum_id,)


def test_find_objects_type_filter_peak(srv, spectrum_id, peak_id):
    """find_objects filters by metadata type (PeakMetadata)."""
    spec_metadata = SpectrumMetadata(name="Sample 1", group="Group A", file="/path/to/file.vms")
    peak_metadata = PeakMetadata(element_type="Fe 2p")

    srv.set_metadata(spectrum_id, spec_metadata)
    srv.set_metadata(peak_id, peak_metadata)

    result = srv.find_objects("element_type", "Fe", match_exact=False, tp=PeakMetadata)
    assert result == (peak_id,)


def test_find_objects_different_fields_spectrum(srv, spectrum_id):
    """find_objects works with different spectrum metadata fields."""
    metadata = SpectrumMetadata(name="Sample 1", group="Group A", file="/path/to/file.vms")
    srv.set_metadata(spectrum_id, metadata)

    # Search by name
    result = srv.find_objects("name", "Sample 1", match_exact=True)
    assert result == (spectrum_id,)

    # Search by group
    result = srv.find_objects("group", "Group A", match_exact=True)
    assert result == (spectrum_id,)

    # Search by file
    result = srv.find_objects("file", "/path/to/file.vms", match_exact=True)
    assert result == (spectrum_id,)


def test_find_objects_empty_when_no_metadata(srv, spectrum_id):
    """find_objects returns empty tuple when no metadata exists."""
    result = srv.find_objects("name", "Sample", match_exact=False)
    assert result == ()


def test_find_objects_empty_when_field_not_exists(srv, spectrum_id):
    """find_objects returns empty tuple when field doesn't exist."""
    metadata = SpectrumMetadata(name="Sample 1", group="Group A", file="/path/to/file.vms")
    srv.set_metadata(spectrum_id, metadata)

    result = srv.find_objects("nonexistent_field", "value", match_exact=False)
    assert result == ()


def test_find_objects_fuzzy_match_multiple_matches(srv, spectrum_id, simple_collection):
    """find_objects fuzzy match finds multiple objects with matching substrings."""
    from core.services import SpectrumService

    spec_srv = SpectrumService(simple_collection)
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    spec2_id = spec_srv.create_spectrum(x, y, spectrum_id="s2")
    spec3_id = spec_srv.create_spectrum(x, y, spectrum_id="s3")

    metadata1 = SpectrumMetadata(name="Sample 1", group="Group A", file="/path/to/file1.vms")
    metadata2 = SpectrumMetadata(name="Sample 2", group="Group B", file="/path/to/file2.vms")
    metadata3 = SpectrumMetadata(name="Test 1", group="Group C", file="/path/to/file3.vms")

    srv.set_metadata(spectrum_id, metadata1)
    srv.set_metadata(spec2_id, metadata2)
    srv.set_metadata(spec3_id, metadata3)

    # Fuzzy match should find both "Sample 1" and "Sample 2"
    result = srv.find_objects("name", "Sample", match_exact=False)
    assert len(result) == 2
    assert spectrum_id in result
    assert spec2_id in result
    assert spec3_id not in result


def test_find_objects_exact_vs_fuzzy(srv, spectrum_id):
    """find_objects distinguishes between exact and fuzzy matching."""
    metadata = SpectrumMetadata(name="Sample 1", group="Group A", file="/path/to/file.vms")
    srv.set_metadata(spectrum_id, metadata)

    # Exact match: full string required
    result = srv.find_objects("name", "Sample 1", match_exact=True)
    assert result == (spectrum_id,)

    result = srv.find_objects("name", "Sample", match_exact=True)
    assert result == ()

    # Fuzzy match: substring works
    result = srv.find_objects("name", "Sample", match_exact=False)
    assert result == (spectrum_id,)


def test_find_objects_peak_element_type(srv, peak_id):
    """find_objects works with peak element_type field."""
    metadata = PeakMetadata(element_type="Fe 2p")
    srv.set_metadata(peak_id, metadata)

    result = srv.find_objects("element_type", "Fe", match_exact=False)
    assert result == (peak_id,)

    result = srv.find_objects("element_type", "Fe 2p", match_exact=True)
    assert result == (peak_id,)

    result = srv.find_objects("element_type", "2p", match_exact=False)
    assert result == (peak_id,)
