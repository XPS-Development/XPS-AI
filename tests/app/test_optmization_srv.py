import numpy as np
import pytest

from app.optimization import OptimizationService


@pytest.fixture
def srv(simple_collection):
    return OptimizationService(simple_collection)


def test_pipe(srv, region_id):
    srv.optimize_regions(region_id)
