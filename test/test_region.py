import pytest

from lib.domain import Region


def test_region_creates_id_if_missing():
    r = Region(slice_=slice(0, 10), parent_id="s1")

    assert r.id_ is not None
    assert r.id_.startswith("r")


def test_region_rejects_non_slice():
    with pytest.raises(TypeError):
        Region(slice_=(0, 10), parent_id="s1")  # type: ignore


def test_region_rejects_invalid_step():
    with pytest.raises(ValueError):
        Region(slice_=slice(0, 10, 2), parent_id="s1")


def test_region_keeps_parent_id():
    r = Region(slice_=slice(5, 15), parent_id="spectrum-1")
    assert r.parent_id == "spectrum-1"
