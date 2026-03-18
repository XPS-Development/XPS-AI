import pytest

from core.services import RegionService
from core.objects import Region, Spectrum


@pytest.fixture
def srv(simple_collection) -> RegionService:
    return RegionService(simple_collection)


def test_create_region(srv: RegionService, spectrum_id: str) -> None:
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
def test_create_region_out_of_bounds(
    srv: RegionService, spectrum_id: str, start: int | float, stop: int | float
) -> None:
    rid = srv.create_region(spectrum_id, start, stop)
    region = srv.collection.get_typed(rid, Region)
    assert region.slice_.start == 0
    assert region.slice_.stop == 200


def test_update_slice(srv: RegionService) -> None:
    region = next(obj for obj in srv.collection.objects_index.values() if isinstance(obj, Region))

    srv.update_slice(region.id_, start=2, stop=8)

    assert region.slice_ == slice(2, 8)


def test_update_slice_invalid(srv):
    region = next(obj for obj in srv.collection.objects_index.values() if isinstance(obj, Region))
    srv.update_slice(region.id_, start=-1, stop=5)
    assert region.slice_.start == 0
    assert region.slice_.stop == 200


def test_remove_region(srv):
    region = next(obj for obj in srv.collection.objects_index.values() if isinstance(obj, Region))

    removed_objects = srv.detach(region.id_)

    assert len(removed_objects) == 3  # region, peak, background
    assert isinstance(removed_objects[0], Region)
    assert removed_objects[0].id_ == region.id_

    with pytest.raises(KeyError):
        srv.collection.get(region.id_)


# --- Value-slicing mode tests ---


def test_convert_value_to_index(srv: RegionService, spectrum_id: str) -> None:
    """_convert_value_to_index maps x-axis values to indices via searchsorted."""
    # x_axis is np.linspace(-10, 10, 200); value -10 -> 0, 10 -> last index (199), 0 -> 100
    assert srv._convert_value_to_index(spectrum_id, -10.0) == 0
    assert srv._convert_value_to_index(spectrum_id, 10.0) == 199
    assert srv._convert_value_to_index(spectrum_id, 0.0) == 100


def test_create_region_value_mode(srv: RegionService, spectrum_id: str) -> None:
    """create_region with mode='value' interprets start/stop as x-axis values."""
    # x = np.linspace(-10, 10, 200); -5 and 5 are inside
    rid = srv.create_region(spectrum_id, start=-5.0, stop=5.0, mode="value")
    region = srv.collection.get_typed(rid, Region)
    # Region stores indices; -5 and 5 map to indices ~50 and ~150
    assert region.slice_.start >= 0
    assert region.slice_.stop <= 200
    assert region.slice_.start < region.slice_.stop
    start_val, stop_val = srv.get_slice(rid, mode="value")
    assert start_val <= -4.9
    assert stop_val >= 4.9


def test_create_region_value_mode_extreme_values_clamped(srv: RegionService, spectrum_id: str) -> None:
    """create_region with mode='value' and values outside x range uses searchsorted; valid region created."""
    # Values beyond x range yield indices at 0 and len(x); slice is still valid (0 < 200 <= 200)
    rid = srv.create_region(spectrum_id, start=-100.0, stop=100.0, mode="value")
    region = srv.collection.get_typed(rid, Region)
    assert region.slice_.start == 0
    assert region.slice_.stop == 200


def test_update_slice_value_mode(srv: RegionService, region_id: str) -> None:
    """update_slice with mode='value' interprets start/stop as x-axis values."""
    srv.update_slice(region_id, start=-3.0, stop=3.0, mode="value")
    start_val, stop_val = srv.get_slice(region_id, mode="value")
    assert start_val <= -2.9
    assert stop_val >= 2.9


def test_get_slice_index_mode(srv: RegionService, region_id: str) -> None:
    """get_slice with mode='index' returns (start, stop) as indices."""
    start, stop = srv.get_slice(region_id, mode="index")
    region = srv.collection.get_typed(region_id, Region)
    assert start == region.slice_.start
    assert stop == region.slice_.stop


def test_get_slice_value_mode(srv: RegionService, region_id: str) -> None:
    """get_slice with mode='value' returns (start, stop) as x-axis values."""
    start_val, stop_val = srv.get_slice(region_id, mode="value")
    region = srv.collection.get_typed(region_id, Region)
    spectrum = srv.collection.get_typed_parent(region_id, Spectrum)
    assert start_val == spectrum.x[region.slice_.start]
    assert stop_val == spectrum.x[region.slice_.stop - 1]
