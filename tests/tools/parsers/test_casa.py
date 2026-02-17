"""Tests for casa-like .txt parser."""

import pytest

from tools.parsers.casa import parse_casa_txt
from tools.parsers.types import ParsedSpectrum


def test_parse_casa_txt_returns_single_spectrum():
    """parse_casa_txt returns a single-element list."""
    result = parse_casa_txt("tests/data/test_1_spec.txt")
    assert len(result) == 1
    assert isinstance(result[0], ParsedSpectrum)


def test_parse_casa_txt_extracts_name():
    """parse_casa_txt extracts spectrum name from first line."""
    result = parse_casa_txt("tests/data/test_1_spec.txt")
    assert result[0].metadata.name == "Ag3d"


def test_parse_casa_txt_extracts_binding_energy_and_cps():
    """parse_casa_txt uses B.E. and CPS columns by default."""
    result = parse_casa_txt("tests/data/test_1_spec.txt")
    ps = result[0]
    assert len(ps.x) == len(ps.y)
    assert ps.x[0] == pytest.approx(380.0)
    assert ps.y[0] == pytest.approx(37690.0)


def test_parse_casa_txt_file_path_in_metadata():
    """parse_casa_txt sets file path in result."""
    result = parse_casa_txt("tests/data/test_1_spec.txt")
    assert "test_1_spec.txt" in result[0].metadata.file


def test_parse_casa_txt_too_few_lines_raises(tmp_path):
    """parse_casa_txt raises ValueError for file with too few lines."""
    p = tmp_path / "short.txt"
    p.write_text("name\n")
    with pytest.raises(ValueError, match="too few lines"):
        parse_casa_txt(p)
