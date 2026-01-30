import numpy as np
import pytest


def test_get_norm_ctx_by_spectrum(data_query_service, spectrum_id):
    ctx = data_query_service.get_norm_ctx(spectrum_id=spectrum_id)

    assert ctx is not None


def test_get_norm_ctx_by_region(data_query_service, region_id):
    ctx = data_query_service.get_norm_ctx(region_id=region_id)

    assert ctx is not None


def test_get_norm_ctx_invalid_args(data_query_service):
    with pytest.raises(ValueError):
        data_query_service.get_norm_ctx()

    with pytest.raises(ValueError):
        data_query_service.get_norm_ctx(spectrum_id="a", region_id="b")


def test_get_spectrum_data_raw(data_query_service, spectrum_id):
    x, y = data_query_service.get_spectrum_data(spectrum_id, normalized=False)

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_get_spectrum_data_normalized(data_query_service, spectrum_id):
    x_raw, y_raw = data_query_service.get_spectrum_data(spectrum_id, normalized=False)
    x_norm, y_norm = data_query_service.get_spectrum_data(spectrum_id, normalized=True)

    assert np.allclose(x_raw, x_norm)
    assert not np.allclose(y_raw, y_norm)


def test_get_region_data(data_query_service, region_id):
    x, y = data_query_service.get_region_data(region_id)

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(x) == len(y)


def test_get_region_data_normalized(data_query_service, region_id):
    x_raw, y_raw = data_query_service.get_region_data(region_id, normalized=False)
    x_norm, y_norm = data_query_service.get_region_data(region_id, normalized=True)

    assert np.allclose(x_raw, x_norm)
    assert not np.allclose(y_raw, y_norm)
