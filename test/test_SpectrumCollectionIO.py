import pytest
from pathlib import Path

from lib.spectra import SpectrumCollection, Spectrum, Region, Peak, PeakParameter
from lib.io_utils import SpectrumCollectionIO


import numpy as np
import pytest

from lib.spectra import SpectrumCollection, Spectrum, Peak
from lib.io_utils import SpectrumCollectionIO


@pytest.fixture
def simple_collection():
    """
    Spectrum with one region and one peak.
    """
    col = SpectrumCollection()

    x = np.linspace(0, 10, 100, dtype=np.float32)
    y = np.exp(-((x - 5) ** 2))

    spectrum = Spectrum(x=x, y=y, name="spec1", file="file1", group="group1", id="s1")
    col.register(spectrum)

    region = spectrum.create_region(10, 90, background_type="shirley", region_id="r1")
    col.add_link(spectrum, region)

    peak = Peak(amp=1, cen=1, sig=1, frac=1, id_="p1")

    col.add_link(region, peak)

    return col


@pytest.fixture
def io(tmp_path):
    return SpectrumCollectionIO(default_folder=tmp_path)


def test_serialize_collection(simple_collection, io):
    data = io._serialize_collection(simple_collection)

    assert "spectra" in data
    assert len(data["spectra"]) == 1

    spectrum_data = next(iter(data["spectra"].values()))

    assert "energy" in spectrum_data
    assert "intensity" in spectrum_data
    assert "regions" in spectrum_data
    assert "peaks" in spectrum_data

    region_data = next(iter(spectrum_data["regions"].values()))
    assert "start_idx" in region_data
    assert "end_idx" in region_data
    assert "background_type" in region_data
    assert isinstance(region_data["peaks"], list)

    peak_data = next(iter(spectrum_data["peaks"].values()))
    for key in ("amp", "cen", "sig", "frac"):
        assert key in peak_data
        assert "value" in peak_data[key]


def test_deserialize_collection_roundtrip(simple_collection, io):
    serialized = io._serialize_collection(simple_collection)
    restored = io._deserialize_collection(serialized)

    assert isinstance(restored, SpectrumCollection)
    assert len(restored.spectra_index) == 1

    spectrum = next(iter(restored.spectra_index.values()))
    assert spectrum.name == "spec1"
    assert spectrum.file == "file1"
    assert spectrum.group == "group1"
    assert spectrum.id == "s1"

    regions = spectrum.regions
    assert len(regions) == 1

    region = restored.get(regions[0])
    assert region.id == "r1"
    assert len(region.peaks) == 1

    peak = restored.get(region.peaks[0])
    assert peak.id == "p1"
    assert pytest.approx(peak.amp, rel=1e-6) == 1.0
    assert pytest.approx(peak.cen, rel=1e-6) == 1.0
    assert pytest.approx(peak.sig, rel=1e-6) == 1.0
    assert pytest.approx(peak.frac, rel=1e-6) == 1.0


def test_on_example(io):
    coll = io.load(Path("test/data/test_example.json"))
    assert len(coll.peaks_index) == 13
    assert len(coll.regions_index) == 2
    assert "OU44 H2O hv20-NAV_2_results" in coll.spectra_index
    assert "OU44 H2O hv20-NAV_3_results" in coll.spectra_index

    # get example peak
    peak_0 = coll.get("peak_0")
    assert peak_0 is not None
    assert pytest.approx(peak_0.amp, rel=1e-4) == 849.591
    assert pytest.approx(peak_0.cen, rel=1e-4) == 390.326
    assert pytest.approx(peak_0.sig, rel=1e-4) == 1.312
    assert pytest.approx(peak_0.frac, rel=1e-4) == 0
