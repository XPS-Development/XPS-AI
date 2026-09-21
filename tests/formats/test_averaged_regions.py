"""Tests for wide averaged-regions CSV/DAT parser."""

from pathlib import Path

import numpy as np
import pytest

from formats.averaged_regions import looks_like_averaged_regions_header, parse_averaged_regions
from formats.types import ParsedSpectrum

_CSV = Path("tests/data/test_averaged_regions.csv")
_DAT = Path("tests/data/test_averaged_regions.dat")


def test_looks_like_averaged_regions_header_csv():
    """CSV-style header with BE/Intensity pairs is recognized."""
    assert looks_like_averaged_regions_header(
        "C1s_BE_eV,C1s_Intensity_cps,O1s_BE_eV,O1s_Intensity_cps"
    )


def test_looks_like_averaged_regions_header_rejects_peak_table():
    """Casa peak-table headers are not treated as averaged-regions."""
    assert not looks_like_averaged_regions_header("Position,Area,FWHM,GL")


def test_parse_averaged_regions_csv_returns_three_spectra():
    """parse_averaged_regions splits a CSV into one spectrum per line pair."""
    result = parse_averaged_regions(_CSV)
    assert len(result) == 3
    assert all(isinstance(ps, ParsedSpectrum) for ps in result)
    assert [ps.metadata.name for ps in result] == ["C1s", "O1s", "Ti2p"]


def test_parse_averaged_regions_skips_empty_cells():
    """Trailing empty cells do not produce points; Ti2p is longer."""
    result = parse_averaged_regions(_CSV)
    by_name = {ps.metadata.name: ps for ps in result}
    assert len(by_name["C1s"].x) == 2
    assert len(by_name["O1s"].x) == 2
    assert len(by_name["Ti2p"].x) == 4
    np.testing.assert_array_almost_equal(by_name["C1s"].x, [279.9607, 280.0607])
    np.testing.assert_array_almost_equal(by_name["C1s"].y, [1094.6137, 1130.6137])
    np.testing.assert_array_almost_equal(
        by_name["Ti2p"].x, [450.9776, 451.0776, 451.1776, 451.2776]
    )


def test_parse_averaged_regions_dat_matches_csv():
    """Tab-separated .dat file yields the same spectra as the CSV twin."""
    csv_result = parse_averaged_regions(_CSV)
    dat_result = parse_averaged_regions(_DAT)
    assert len(csv_result) == len(dat_result)
    for a, b in zip(csv_result, dat_result, strict=True):
        assert a.metadata.name == b.metadata.name
        np.testing.assert_array_almost_equal(a.x, b.x)
        np.testing.assert_array_almost_equal(a.y, b.y)


def test_parse_averaged_regions_metadata_file_and_group():
    """Metadata uses spectrum name, empty group, and source path."""
    ps = parse_averaged_regions(_CSV)[0]
    assert ps.metadata.name == "C1s"
    assert ps.metadata.group == ""
    assert "test_averaged_regions.csv" in ps.metadata.file


def test_parse_averaged_regions_peak_table_raises(tmp_path: Path):
    """Unrecognized CSV (peak table) raises ValueError."""
    path = tmp_path / "peaks.csv"
    path.write_text("Position,Area,FWHM,GL\n284.18,40588.62,1.11,0.30\n", encoding="utf-8")
    with pytest.raises(ValueError, match="column pairs"):
        parse_averaged_regions(path)


def test_parse_averaged_regions_empty_file_raises(tmp_path: Path):
    """Empty file raises ValueError."""
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        parse_averaged_regions(path)
