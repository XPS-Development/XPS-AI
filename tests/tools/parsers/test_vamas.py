"""Tests for VAMAS parser."""

import pytest

from tools.parsers.vamas import parse_vamas
from tools.parsers.types import ParsedSpectrum


def test_parse_vamas_returns_list_of_spectra():
    """parse_vamas returns one ParsedSpectrum per block."""
    result = parse_vamas("tests/data/test_18_spec.vms")
    assert len(result) == 18
    for ps in result:
        assert isinstance(ps, ParsedSpectrum)


def test_parse_vamas_extracts_data():
    """parse_vamas extracts x and y arrays from blocks."""
    result = parse_vamas("tests/data/test_1_spec.vms")
    ps = result[0]
    assert len(ps.x) == len(ps.y)
    assert len(ps.x) > 0


def test_parse_vamas_full_metadata():
    """parse_vamas sets file path in result."""
    result = parse_vamas("tests/data/test_1_spec.vms")[0]
    assert "test_1_spec.vms" in result.metadata.file
    assert result.metadata.name == "N1s"
    assert result.metadata.group == "N1s hv50"
