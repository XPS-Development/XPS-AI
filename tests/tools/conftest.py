"""Shared fixtures for tools tests (evaluation, optimization, etc.)."""

import pytest

from app.dto_service import DTOService
from core.services import CoreContext


@pytest.fixture
def ctx(simple_collection):
    """Application context built from simple_collection for command execution."""
    return CoreContext.from_collection(simple_collection)


@pytest.fixture
def dto_service(ctx):
    return DTOService(ctx)
