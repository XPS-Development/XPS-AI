"""Tests for two-column .dat parser."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tools.parsers.dat import parse_dat
from tools.parsers.types import ParsedSpectrum


def test_parse_dat_returns_single_spectrum():
    """parse_dat returns a single-element list."""
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
        f.write(b"100.0 500.0\n101.0 510.0\n102.0 520.0\n")
        path = Path(f.name)
    try:
        result = parse_dat(path)
        assert len(result) == 1
        assert isinstance(result[0], ParsedSpectrum)
    finally:
        path.unlink()


def test_parse_dat_extracts_x_y():
    """parse_dat extracts two columns as x and y."""
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
        np.savetxt(f, np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
        path = Path(f.name)
    try:
        result = parse_dat(path)
        ps = result[0]
        np.testing.assert_array_almost_equal(ps.x, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(ps.y, [10.0, 20.0, 30.0])
    finally:
        path.unlink()


def test_parse_dat_uses_filename_stem_as_name():
    """parse_dat uses file stem as spectrum name."""
    with tempfile.NamedTemporaryFile(suffix=".dat", prefix="my_spectrum_", delete=False) as f:
        f.write(b"0.0 0.0\n1.0 1.0\n")
        path = Path(f.name)
    try:
        result = parse_dat(path)[0]
        assert "my_spectrum" in result.metadata.name
    finally:
        path.unlink()


def test_parse_dat_single_column_raises():
    """parse_dat raises ValueError for single-column file."""
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
        f.write(b"1.0\n2.0\n3.0\n")
        path = Path(f.name)
    try:
        with pytest.raises(ValueError, match="at least two columns"):
            parse_dat(path)
    finally:
        path.unlink()
