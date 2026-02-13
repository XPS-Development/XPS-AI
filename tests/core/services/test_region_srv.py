import pytest

from core.services import RegionService
from core.objects import Region


@pytest.fixture
def srv(simple_collection):
    return RegionService(simple_collection)


def test_create_region(srv, spectrum_id):
    rid = srv.create_region(spectrum_id, start=0, stop=10)

    region = srv.collection.get_typed(rid, Region)

    assert region.parent_id == spectrum_id
    assert region.slice_ == slice(0, 10)


@pytest.mark.parametrize(
    "start, stop",
    [
        (-1, 5),
        (0, 1000),
    ],
)
def test_create_region_out_of_bounds(srv, spectrum_id, start, stop):
    with pytest.raises(ValueError):
        srv.create_region(spectrum_id, start, stop)


def test_create_region_invalid_slice(srv, spectrum_id):
    with pytest.raises(ValueError):
        srv.create_region(spectrum_id, start=5, stop=5)


def test_update_slice(srv):
    region = next(obj for obj in srv.collection.objects_index.values() if isinstance(obj, Region))

    srv.update_slice(region.id_, start=2, stop=8)

    assert region.slice_ == slice(2, 8)


def test_update_slice_invalid(srv):
    region = next(obj for obj in srv.collection.objects_index.values() if isinstance(obj, Region))

    with pytest.raises(ValueError):
        srv.update_slice(region.id_, start=-1, stop=5)


def test_remove_region(srv):
    region = next(obj for obj in srv.collection.objects_index.values() if isinstance(obj, Region))

    removed_objects = srv.detach(region.id_)

    assert len(removed_objects) == 3  # region, peak, background
    assert isinstance(removed_objects[0], Region)
    assert removed_objects[0].id_ == region.id_

    with pytest.raises(KeyError):
        srv.collection.get(region.id_)
