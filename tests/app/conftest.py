import pytest

from core.services import CoreContext


@pytest.fixture
def ctx(simple_collection):
    """Application context built from simple_collection for command execution."""
    return CoreContext.from_collection(simple_collection)
