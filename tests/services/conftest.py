import numpy as np
import pytest

from core import SpectrumCollection, Spectrum, Region, Peak, Background
from core.math_models import ModelRegistry, PseudoVoigtPeakModel, ConstantBackgroundModel
from services.core import (
    CollectionQueryService,
    SpectrumService,
    RegionService,
    DataQueryService,
    ComponentService,
)
from services.dto import DTOService


@pytest.fixture()
def reset_model_registry():
    ModelRegistry._registry = {}


@pytest.fixture
def empty_collection() -> SpectrumCollection:
    """
    Empty SpectrumCollection without spectra, regions or components.
    """
    return SpectrumCollection()


@pytest.fixture
def simple_collection(empty_collection, simple_gauss_spectrum) -> SpectrumCollection:
    """
    SpectrumCollection with one spectrum, region, peak and background.
    """
    collection = empty_collection
    x, y = simple_gauss_spectrum

    s = Spectrum(x, y, id_="s1")
    r = Region(slice(20, len(x) + 1 - 20), parent_id=s.id_, id_="r1")
    # create pure gauss
    p = Peak(model=PseudoVoigtPeakModel(), region_id=r.id_, component_id="p1", amp=1, cen=0, sig=1, frac=0)
    bg = Background(model=ConstantBackgroundModel(), region_id=r.id_, component_id="b1", const=1)

    collection.add(s)
    collection.add(r)
    collection.add(p)
    collection.add(bg)

    return collection


@pytest.fixture
def spectrum_id(simple_collection):
    return next(obj.id_ for obj in simple_collection.objects_index.values() if isinstance(obj, Spectrum))


@pytest.fixture
def region_id(simple_collection):
    return next(obj.id_ for obj in simple_collection.objects_index.values() if isinstance(obj, Region))


@pytest.fixture
def peak_id(simple_collection):
    return next(obj.id_ for obj in simple_collection.objects_index.values() if isinstance(obj, Peak))


@pytest.fixture
def background_id(simple_collection):
    return next(obj.id_ for obj in simple_collection.objects_index.values() if isinstance(obj, Background))


@pytest.fixture
def query_service(simple_collection):
    return CollectionQueryService(simple_collection)


@pytest.fixture
def spectrum_service(simple_collection):
    return SpectrumService(simple_collection)


@pytest.fixture
def region_service(simple_collection):
    return RegionService(simple_collection)


@pytest.fixture
def data_query_service(simple_collection):
    return DataQueryService(simple_collection)


@pytest.fixture
def component_service(simple_collection):
    return ComponentService(simple_collection)


@pytest.fixture
def dto_service(simple_collection):
    return DTOService(simple_collection)
