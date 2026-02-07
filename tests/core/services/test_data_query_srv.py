import numpy as np
import pytest

from core.services import DataQueryService


@pytest.fixture
def srv(simple_collection):
    return DataQueryService(simple_collection)


def test_get_norm_ctx_by_spectrum(srv, spectrum_id):
    ctx = srv.get_norm_ctx(spectrum_id=spectrum_id)

    assert ctx is not None


def test_get_norm_ctx_by_region(srv, region_id):
    ctx = srv.get_norm_ctx(region_id=region_id)

    assert ctx is not None


def test_get_norm_ctx_invalid_args(srv):
    with pytest.raises(ValueError):
        srv.get_norm_ctx()

    with pytest.raises(ValueError):
        srv.get_norm_ctx(spectrum_id="a", region_id="b")


def test_get_spectrum_data_raw(srv, spectrum_id):
    x, y = srv.get_spectrum_data(spectrum_id, normalized=False)

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_get_spectrum_data_normalized(srv, spectrum_id):
    x_raw, y_raw = srv.get_spectrum_data(spectrum_id, normalized=False)
    x_norm, y_norm = srv.get_spectrum_data(spectrum_id, normalized=True)

    assert np.allclose(x_raw, x_norm)
    assert not np.allclose(y_raw, y_norm)


def test_get_region_data(srv, region_id):
    x, y = srv.get_region_data(region_id)

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(x) == len(y)


def test_get_region_data_normalized(srv, region_id):
    x_raw, y_raw = srv.get_region_data(region_id, normalized=False)
    x_norm, y_norm = srv.get_region_data(region_id, normalized=True)

    assert np.allclose(x_raw, x_norm)
    assert not np.allclose(y_raw, y_norm)
