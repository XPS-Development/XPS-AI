"""Tests for parse_spectrum_file dispatcher."""

import tempfile
from pathlib import Path

import pytest

from tools.parsers import parse_spectrum_file


def test_parse_spectrum_file_txt_dispatches_to_casa():
    """parse_spectrum_file dispatches .txt to casa parser."""
    result = parse_spectrum_file("tests/data/test_1_spec.txt")
    assert len(result) == 1
    assert result[0].metadata.name == "Ag3d"


def test_parse_spectrum_file_dat_dispatches_to_dat():
    """parse_spectrum_file dispatches .dat to dat parser."""
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
        f.write(b"1.0 10.0\n2.0 20.0\n")
        path = Path(f.name)
    try:
        result = parse_spectrum_file(path)
        assert len(result) == 1
        assert len(result[0].x) == 2
    finally:
        path.unlink()


def test_parse_spectrum_file_vms_dispatches_to_vamas():
    """parse_spectrum_file dispatches .vms to vamas parser."""
    result = parse_spectrum_file("tests/data/test_1_spec.vms")
    assert len(result) == 1
    assert result[0].metadata.name == "N1s"

    result = parse_spectrum_file("tests/data/test_18_spec.vms")
    assert len(result) == 18


def test_parse_spectrum_file_unsupported_extension_raises():
    """parse_spectrum_file raises ValueError for unsupported extension."""
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        path = Path(f.name)
    try:
        with pytest.raises(ValueError, match="Unsupported file extension"):
            parse_spectrum_file(path)
    finally:
        path.unlink()
