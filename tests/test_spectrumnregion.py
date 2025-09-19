import sys
import os

# Добавляем корень проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from uuid import uuid4

from lib.spectra import Spectrum, Region, Peak


@pytest.fixture
def sample_spectrum():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    return Spectrum(x=x, y=y, name="test_spectrum")


@pytest.fixture
def smoothed_spectrum(sample_spectrum):
    spec = sample_spectrum
    spec.y_smoothed = spec.y  # для упрощения
    return spec


def test_add_and_remove_peak():
    region = Region(spectrum_id=uuid4().hex)
    peak = Peak(region_id=region.id)

    region.add_peak(peak)
    assert peak in region.peaks

    region.remove_peak(peak)
    assert peak not in region.peaks

    # через ID
    region.add_peak(peak)
    region.remove_peak(peak.id)
    assert peak not in region.peaks


def test_update_range():
    region = Region(spectrum_id=uuid4().hex)
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    region.update_range(x, y, i_1=10, i_2=20)

    assert np.array_equal(region.x, x)
    assert np.array_equal(region.y, y)
    assert region.i_1 == 10
    assert region.i_2 == 20


def test_repr_region():
    region = Region(spectrum_id="abc123")
    s = repr(region)
    assert "Region" in s
    assert "peaks=0" in s


def test_add_and_remove_region(smoothed_spectrum):
    spec = smoothed_spectrum
    region = Region(spectrum_id=spec.id)
    spec.add_region(region)

    assert region in spec.regions

    spec.remove_region(region)
    assert region not in spec.regions

    # через ID
    spec.add_region(region)
    spec.remove_region(region.id)
    assert region not in spec.regions


def test_create_region_without_normalization(sample_spectrum):
    spec = sample_spectrum
    with pytest.raises(ValueError):
        spec.create_region(0, 10)


def test_create_region_with_normalization(smoothed_spectrum):
    spec = smoothed_spectrum
    region = spec.create_region(0, 10)
    assert region in spec.regions
    assert np.array_equal(region.x, spec.x[0:10])
    assert np.array_equal(region.y, spec.y[0:10])


def test_charge_correction(smoothed_spectrum):
    spec = smoothed_spectrum
    x_copy = spec.x.copy()

    spec.set_charge_correction(1.0)
    assert np.allclose(spec.x, x_copy + 1.0)
    assert spec.charge_correction == 1.0

    spec.remove_charge_correction()
    assert np.allclose(spec.x, x_copy)
    assert spec.charge_correction == 0.0


def test_summary_and_repr(smoothed_spectrum):
    spec = smoothed_spectrum
    region = spec.create_region(0, 10)
    summary = spec.summary()

    assert "Spectrum ID" in summary
    assert f"Regions: {len(spec.regions)}" in summary

    s = repr(spec)
    assert "Spectrum" in s
    assert f"regions={len(spec.regions)}" in s
