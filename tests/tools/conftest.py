"""Shared fixtures for tools tests (evaluation, optimization, etc.)."""

import pytest

from tools.dto import DTOService


@pytest.fixture
def dto_service(simple_collection):
    """DTO service yielding ComponentLike / RegionLike / SpectrumLike objects (DTOs)."""
    return DTOService(simple_collection)
