import pytest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from lib.spectra import SpectrumCollection
from lib.io_utils import RawReader


@pytest.mark.parametrize("x_type", ["BE", "KE"])
@pytest.mark.parametrize("y_type", ["Counts", "CPS"])
def test_read_casa_text(x_type, y_type):
    reader = RawReader(collection=None, x_type=x_type, y_type=y_type)
    path = Path("test/data/test_cl2p_first.txt")
    with path.open("r") as f:
        x, y = reader.read_casa_text(f)

    assert len(x) == len(y)
    assert len(x) == 151


@pytest.mark.parametrize("x_type", ["BE", "KE"])
@pytest.mark.parametrize("y_type", ["Counts", "CPS"])
def test_read_vms(x_type, y_type):
    collection = SpectrumCollection()
    reader = RawReader(collection)
    reader.read_vms(Path("test/data/test_16_total.vms"))
    assert len(collection.spectra_index) == 16

    # rely on test_read_casa_text
    path = Path("test/data/test_cl2p_first.txt")
    with path.open("r") as f:
        x, y = reader.read_casa_text(f)

    # find the same spectrum with len = 151 and name = Cl2p
    for spectrum in collection.spectra_index.values():
        if len(spectrum.x) == 151 and spectrum.name == "Cl2p":
            break

    assert_allclose(spectrum.x, x, rtol=1e-6)
    assert_allclose(spectrum.y, y, rtol=1e-6)


@pytest.mark.parametrize("x_type", ["BE", "KE"])
@pytest.mark.parametrize("y_type", ["Counts", "CPS"])
def test_read_files(x_type, y_type):
    collection = SpectrumCollection()
    reader = RawReader(collection)
    reader.read_files([Path("test/data/test_cl2p_first.txt"), Path("test/data/test_16_total.vms")])
    assert len(collection.spectra_index) == 17


def test_read_specs():
    reader = RawReader(collection=None)
    with pytest.raises(NotImplementedError):
        reader.read_files([Path("test/data/test_ag3d.xml")])
