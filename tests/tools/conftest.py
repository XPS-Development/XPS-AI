"""Shared fixtures for tools tests (evaluation, optimization, etc.)."""

import pytest

from core.services import CoreContext
from tools.dto import DTOService


@pytest.fixture
def ctx(simple_collection):
    """Application context built from simple_collection for command execution."""
    return CoreContext.from_collection(simple_collection)


@pytest.fixture
def dto_service(ctx):
    return DTOService(ctx)
