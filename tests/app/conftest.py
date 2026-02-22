import pytest

from core.services import (
    CollectionQueryService,
    SpectrumService,
    RegionService,
    DataQueryService,
    ComponentService,
    CoreContext,
)
from app.dto import DTOService
from app.evaluation import EvaluationService


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


@pytest.fixture
def evaluation_service():
    return EvaluationService()


@pytest.fixture
def ctx(simple_collection):
    """Application context built from simple_collection for command execution."""
    return CoreContext.from_collection(simple_collection)
