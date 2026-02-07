import numpy as np
import pytest

from core.math_models.model_funcs import pvoigt

from core.objects import Spectrum, Region, Peak, Background
from core.collection import CoreCollection
from core.math_models import ModelRegistry, PseudoVoigtPeakModel, ConstantBackgroundModel


RNG = np.random.default_rng(seed=42)

X_AXIS_START: float = -10.0
X_AXIS_STOP: float = 10.0
X_AXIS_NUM_POINTS: int = 200


@pytest.fixture
def x_axis() -> np.ndarray:
    return np.linspace(X_AXIS_START, X_AXIS_STOP, X_AXIS_NUM_POINTS)


@pytest.fixture
def noise() -> np.ndarray:
    return RNG.normal(loc=0, scale=0.01, size=X_AXIS_NUM_POINTS)


@pytest.fixture
def simple_gauss(x_axis: np.ndarray) -> np.ndarray:
    amp = 1
    cen = 0
    sig = 1
    frac = 0

    y = pvoigt(x_axis, amp, cen, sig, frac)

    return y


@pytest.fixture
def simple_gauss_spectrum(x_axis, simple_gauss, noise):
    background = 1
    return x_axis, simple_gauss + noise + background


@pytest.fixture()
def reset_model_registry():
    ModelRegistry._registry = {}


@pytest.fixture
def empty_collection() -> CoreCollection:
    """
    Empty SpectrumCollection without spectra, regions or components.
    """
    return CoreCollection()


@pytest.fixture
def simple_collection(empty_collection, simple_gauss_spectrum) -> CoreCollection:
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
