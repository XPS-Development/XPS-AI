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
