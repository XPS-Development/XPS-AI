"""Tests for ApplicationContext."""

import pytest

from app.command.utils import ApplicationContext
from core.services import (
    CollectionQueryService,
    SpectrumService,
    RegionService,
    DataQueryService,
    ComponentService,
)


def test_application_context_from_collection_returns_context(empty_collection):
    """ApplicationContext.from_collection returns context with all services."""
    ctx = ApplicationContext.from_collection(empty_collection)
    assert ctx is not None


def test_application_context_exposes_all_attributes(empty_collection):
    """Context exposes collection, spectrum, region, data, component attributes."""
    ctx = ApplicationContext.from_collection(empty_collection)
    assert isinstance(ctx.collection, CollectionQueryService)
    assert isinstance(ctx.spectrum, SpectrumService)
    assert isinstance(ctx.region, RegionService)
    assert isinstance(ctx.data, DataQueryService)
    assert isinstance(ctx.component, ComponentService)


def test_application_context_services_share_collection(empty_collection):
    """All services in context share the same underlying collection."""
    ctx = ApplicationContext.from_collection(empty_collection)
    assert ctx.collection.collection is empty_collection
    assert ctx.spectrum.collection is empty_collection
    assert ctx.region.collection is empty_collection
    assert ctx.data.collection is empty_collection
    assert ctx.component.collection is empty_collection
