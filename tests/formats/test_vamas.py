"""Tests for VAMAS parser."""

from pathlib import Path

import pytest

from formats.types import ParsedSpectrum
from formats.vamas import UnsupportedVamasFormat, parse_vamas


def test_parse_vamas_single_block_fixture() -> None:
    """parse_vamas returns one spectrum with expected metadata and shape."""
    result = parse_vamas("tests/data/test_1_spec.vms")
    assert len(result) == 1
    ps = result[0]
    assert isinstance(ps, ParsedSpectrum)
    assert ps.metadata.name == "N1s"
    assert ps.metadata.group == "N1s hv50"
    assert "test_1_spec.vms" in ps.metadata.file
    assert len(ps.x) == len(ps.y) == 201
    assert ps.x[0] == pytest.approx(393.0)
    assert ps.x[-1] == pytest.approx(413.0)
    assert ps.y[0] == pytest.approx(14879.0)
    assert ps.y[100] == pytest.approx(15346.0)


def test_parse_vamas_multi_block_fixture() -> None:
    """parse_vamas returns one ParsedSpectrum per block in a multi-block file."""
    result = parse_vamas("tests/data/test_18_spec.vms")
    assert len(result) == 18
    for ps in result:
        assert isinstance(ps, ParsedSpectrum)
        assert len(ps.x) == len(ps.y) > 0
        assert ps.x[0] < ps.x[-1]
    assert result[0].metadata.name == "02-Pd"
    assert result[6].metadata.name == "02-Al2p"
    assert len(result[0].x) == 241
    assert len(result[8].x) == 5526


def test_parse_vamas_kinetic_energy_axis() -> None:
    """use_binding_energy=False returns the kinetic energy axis."""
    result = parse_vamas("tests/data/test_1_spec.vms", use_binding_energy=False)
    ps = result[0]
    assert ps.x[0] == pytest.approx(840.6)
    assert ps.x[-1] == pytest.approx(860.6)


def test_parse_vamas_unsupported_experiment_mode(tmp_path: Path) -> None:
    """MAP experiment mode raises UnsupportedVamasFormat."""
    lines = Path("tests/data/test_1_spec.vms").read_text(encoding="utf-8").splitlines()
    lines[7] = "MAP"
    path = tmp_path / "map_mode.vms"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(UnsupportedVamasFormat, match="experiment_mode"):
        parse_vamas(path)
