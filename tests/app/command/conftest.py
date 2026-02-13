import pytest

from app.command.utils import ApplicationContext


@pytest.fixture
def app_context(simple_collection):
    """Application context built from simple_collection for command execution."""
    return ApplicationContext.from_collection(simple_collection)
