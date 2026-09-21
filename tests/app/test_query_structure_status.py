"""Tests for spectrum structure status used by tree status dots."""

import numpy as np

from app.query_service import QueryService
from core.objects import Region, Spectrum
from core.services import CoreContext


def test_spectrum_structure_status_empty(empty_collection) -> None:
    """A spectrum with no regions reports empty."""
    x = np.linspace(0, 1, 10)
    spectrum = Spectrum(x, x + 1.0, id_="s_empty")
    empty_collection.add(spectrum)
    query = QueryService(CoreContext.from_collection(empty_collection))
    assert query.get_spectrum_structure_status("s_empty") == "empty"


def test_spectrum_structure_status_peaks(simple_collection, spectrum_id: str) -> None:
    """Default simple collection with a peak reports peaks."""
    query = QueryService(CoreContext.from_collection(simple_collection))
    assert query.get_spectrum_structure_status(spectrum_id) == "peaks"


def test_spectrum_structure_status_regions_only(empty_collection) -> None:
    """A region without peaks reports regions."""
    x = np.linspace(0, 1, 20)
    spectrum = Spectrum(x, x + 1.0, id_="s_reg")
    empty_collection.add(spectrum)
    region = Region(slice(2, 18), parent_id="s_reg", id_="r_reg")
    empty_collection.add(region)
    query = QueryService(CoreContext.from_collection(empty_collection))
    assert query.get_spectrum_structure_status("s_reg") == "regions"
