"""Tests for MetadataService."""

import numpy as np
import pytest

from core.services import MetadataService, SpectrumService
from core.metadata import SpectrumMetadata, RegionMetadata, PeakMetadata


@pytest.fixture
def srv(simple_collection):
    return MetadataService(simple_collection)


def test_get_spectrum_metadata_returns_default_when_empty(srv, spectrum_id):
    """get_spectrum_metadata returns default when no metadata stored."""
    md = srv.get_spectrum_metadata(spectrum_id)
    assert md == SpectrumMetadata(name="", group="", file="")


def test_set_and_get_spectrum_metadata(srv, spectrum_id):
    """set_spectrum_metadata and get_spectrum_metadata round-trip."""
    metadata = SpectrumMetadata(name="Sample 1", group="Group A", file="/path/to/file.vms")
    srv.set_spectrum_metadata(spectrum_id, metadata)
    assert srv.get_spectrum_metadata(spectrum_id) == metadata


def test_get_region_metadata_returns_default_when_empty(srv, region_id):
    """get_region_metadata returns default when no metadata stored."""
    md = srv.get_region_metadata(region_id)
    assert md == RegionMetadata()


def test_set_and_get_region_metadata(srv, region_id):
    """set_region_metadata and get_region_metadata round-trip."""
    metadata = RegionMetadata()
    srv.set_region_metadata(region_id, metadata)
    assert srv.get_region_metadata(region_id) == metadata


def test_get_peak_metadata_returns_default_when_empty(srv, peak_id):
    """get_peak_metadata returns default when no metadata stored."""
    md = srv.get_peak_metadata(peak_id)
    assert md == PeakMetadata(element_type="")


def test_set_and_get_peak_metadata(srv, peak_id):
    """set_peak_metadata and get_peak_metadata round-trip."""
    metadata = PeakMetadata(element_type="Fe 2p")
    srv.set_peak_metadata(peak_id, metadata)
    assert srv.get_peak_metadata(peak_id) == metadata


def test_get_spectrum_metadata_raises_key_error_for_unknown_id(srv):
    """get_spectrum_metadata raises KeyError for non-existent spectrum."""
    with pytest.raises(KeyError):
        srv.get_spectrum_metadata("nonexistent-spectrum-id")


def test_set_spectrum_metadata_raises_key_error_for_unknown_id(srv):
    """set_spectrum_metadata raises KeyError for non-existent spectrum."""
    metadata = SpectrumMetadata(name="x", group="y", file="z")
    with pytest.raises(KeyError):
        srv.set_spectrum_metadata("nonexistent-spectrum-id", metadata)


def test_get_peak_metadata_raises_type_error_for_spectrum_id(srv, spectrum_id):
    """get_peak_metadata raises TypeError when given spectrum ID instead of peak ID."""
    with pytest.raises(TypeError, match="not Peak"):
        srv.get_peak_metadata(spectrum_id)


def test_get_spectrum_metadata_raises_type_error_for_peak_id(srv, peak_id):
    """get_spectrum_metadata raises TypeError when given peak ID instead of spectrum ID."""
    with pytest.raises(TypeError, match="not Spectrum"):
        srv.get_spectrum_metadata(peak_id)


def test_metadata_persists_across_multiple_objects(srv, spectrum_id, region_id, peak_id):
    """Metadata for different object types is stored independently."""
    spec_md = SpectrumMetadata(name="Spec", group="G", file="f.vms")
    peak_md = PeakMetadata(element_type="C 1s")

    srv.set_spectrum_metadata(spectrum_id, spec_md)
    srv.set_peak_metadata(peak_id, peak_md)

    assert srv.get_spectrum_metadata(spectrum_id) == spec_md
    assert srv.get_region_metadata(region_id) == RegionMetadata()
    assert srv.get_peak_metadata(peak_id) == peak_md


def test_find_spectra_by_metadata_exact(srv, spectrum_id):
    """find_spectra_by_metadata_exact returns IDs with exact metadata match."""
    metadata = SpectrumMetadata(name="Sample A", group="Group 1", file="data.vms")
    srv.set_spectrum_metadata(spectrum_id, metadata)

    result = srv.find_spectra_by_metadata_exact(metadata)
    assert result == (spectrum_id,)

    result_mismatch = srv.find_spectra_by_metadata_exact(
        SpectrumMetadata(name="Other", group="Group 1", file="data.vms")
    )
    assert result_mismatch == ()


def test_find_spectra_by_metadata_similar(srv, spectrum_id, simple_collection):
    """find_spectra_by_metadata_similar returns IDs with substring match."""
    srv.set_spectrum_metadata(
        spectrum_id,
        SpectrumMetadata(name="Iron Oxide", group="Oxides", file="/path/fe_sample.vms"),
    )

    # Create second spectrum with different metadata
    spec_srv = SpectrumService(simple_collection)
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 5, 50)  # Non-constant for valid normalization
    sid2 = spec_srv.create_spectrum(x, y)
    srv.set_spectrum_metadata(
        sid2,
        SpectrumMetadata(name="Carbon", group="Elements", file="c_sample.vms"),
    )

    # Substring in name (case-insensitive)
    result = srv.find_spectra_by_metadata_similar(
        SpectrumMetadata(name="iron", group="", file="")
    )
    assert spectrum_id in result
    assert sid2 not in result

    # Substring in group
    result = srv.find_spectra_by_metadata_similar(
        SpectrumMetadata(name="", group="oxide", file="")
    )
    assert spectrum_id in result

    # Empty filter returns all spectra
    result_all = srv.find_spectra_by_metadata_similar(
        SpectrumMetadata(name="", group="", file="")
    )
    assert len(result_all) == 2
    assert spectrum_id in result_all
    assert sid2 in result_all


def test_find_peaks_by_metadata_exact(srv, peak_id):
    """find_peaks_by_metadata_exact returns IDs with exact metadata match."""
    metadata = PeakMetadata(element_type="Fe 2p")
    srv.set_peak_metadata(peak_id, metadata)

    result = srv.find_peaks_by_metadata_exact(metadata)
    assert result == (peak_id,)

    result_mismatch = srv.find_peaks_by_metadata_exact(PeakMetadata(element_type="C 1s"))
    assert result_mismatch == ()


def test_find_peaks_by_metadata_similar(srv, peak_id):
    """find_peaks_by_metadata_similar returns IDs with substring match."""
    srv.set_peak_metadata(peak_id, PeakMetadata(element_type="Fe 2p3/2"))

    result = srv.find_peaks_by_metadata_similar(PeakMetadata(element_type="Fe"))
    assert result == (peak_id,)

    result = srv.find_peaks_by_metadata_similar(PeakMetadata(element_type="2p"))
    assert result == (peak_id,)

    result = srv.find_peaks_by_metadata_similar(PeakMetadata(element_type=""))
    assert result == (peak_id,)


def test_find_regions_by_metadata_exact_and_similar(srv, region_id):
    """find_regions_by_metadata returns all regions (RegionMetadata has no fields)."""
    exact = srv.find_regions_by_metadata_exact(RegionMetadata())
    similar = srv.find_regions_by_metadata_similar(RegionMetadata())

    assert region_id in exact
    assert region_id in similar
    assert exact == similar
