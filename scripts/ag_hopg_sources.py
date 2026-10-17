"""Source recipes and acquisition metadata for the native Ag/HOPG preparation utility."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from formats import parse_spectrum_file
from formats.vamas import _LineReader, _parse_block, _parse_header

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Recipe:
    key: str
    label: str
    prefix: str
    steps: int
    silver_blocked: bool
    cell: int


RECIPES = (
    Recipe("sample2_after_atm", "Sample 2 after atm", "Sample 2 after atm", 42, True, 21),
    Recipe("new_hopg_after_atm", "New HOPG after atm", "New Ag HOPG clb after atm", 13, False, 24),
    Recipe(
        "new_hopg_after_240mbar",
        "New HOPG after 240 mbar",
        "New Ag HOPG clb after 240mbar",
        13,
        False,
        27,
    ),
)


def stable_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def read_parameters(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [
            {name: float(row[name]) for name in ("Position", "Area", "FWHM", "GL")}
            for row in csv.DictReader(stream)
        ]
    if any(
        not all(math.isfinite(value) for value in row.values())
        or row["FWHM"] <= 0
        or row["Area"] < 0
        or not 0 <= row["GL"] <= 1
        for row in rows
    ):
        raise ValueError(f"Invalid exported component parameters: {path}")
    return rows


def source_record(path: Path, data_root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(data_root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


class RecordingReader(_LineReader):
    """Retain block header tokens while reusing the existing VAMAS parser."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__(lines)
        self.tokens: list[str] = []

    def next_str(self) -> str:
        value = super().next_str()
        self.tokens.append(value)
        return value


@dataclass(frozen=True)
class Measurement:
    path: Path
    block: int
    name: str
    group: str
    dwell: float
    scans: int
    x: np.ndarray
    y: np.ndarray


def measurements(path: Path) -> list[Measurement]:
    spectra = parse_spectrum_file(path)
    reader = RecordingReader(path.read_text().splitlines())
    header = _parse_header(reader)
    result = []
    if len(spectra) != header.num_blocks:
        raise ValueError("Source contains skipped blocks; review block indexing")
    for index, spectrum in enumerate(spectra):
        reader.tokens = []
        block = _parse_block(header, reader)
        pulse = reader.tokens.index("pulse counting")
        dwell, scans = float(reader.tokens[pulse + 1]), int(reader.tokens[pulse + 2])
        if dwell <= 0 or scans < 1 or block.name != spectrum.metadata.name:
            raise ValueError("Unexpected intensity normalization or block order")
        result.append(
            Measurement(
                path,
                index + 1,
                block.name,
                block.sample,
                dwell,
                scans,
                spectrum.x,
                spectrum.y / (dwell * scans),
            )
        )
    return result


def dat_exports(folder: Path, sources: list[Measurement]) -> list[dict[str, object]]:
    """Identify available exported raw curves by equality to time-normalized VAMAS."""
    matches = []
    for path in sorted(folder.parent.glob("aghv20_sample*/*.dat")):
        rows = [line.split() for line in path.read_text().splitlines()[1:] if line.strip()]
        if not rows:
            continue
        xy = np.asarray([[float(row[0]), float(row[1])] for row in rows])
        complete = [row for row in rows if len(row) > 2]
        for source in sources:
            if len(source.x) != len(xy) or not np.allclose(source.x, xy[:, 0], rtol=0, atol=1e-8):
                continue
            error = float(np.max(np.abs(source.y - xy[:, 1])))
            if error < 0.002:
                matches.append(
                    {
                        "dat_path": str(path),
                        "vamas_path": str(source.path),
                        "block": source.block,
                        "name": source.name,
                        "group": source.group,
                        "dwell_seconds": source.dwell,
                        "max_abs_intensity_difference": error,
                        "fit_bounds": [float(complete[0][0]), float(complete[-1][0])]
                        if complete
                        else None,
                    }
                )
    return matches


def phase_sources(recipe: Recipe, all_sources: list[Measurement]) -> dict[str, list[Measurement]]:
    prefix = "After 240mbar" if recipe.key.endswith("240mbar") else "After atm"
    filename = "Sample 2 after atm.vms" if recipe.silver_blocked else "New-HOPG-after-clb.vms"
    result = {}
    for region in ("Ag3d", "C1s"):
        selected = [
            source
            for source in all_sources
            if source.path.name == filename
            and source.group.startswith(prefix)
            and "part" in source.group
            and f"{region}-hv20" in source.name
        ]
        if len(selected) != recipe.steps:
            raise ValueError(f"Wrong source count for {recipe.key}/{region}")
        if any(source.scans != 1 for source in selected):
            raise ValueError("Selected notebook series unexpectedly contains repeated scans")
        offset = 1 if region == "Ag3d" else 2
        numbers = [int(re.match(r"(\d+)-", source.name)[1]) for source in selected]
        if numbers != [3 * i + offset for i in range(recipe.steps)]:
            raise ValueError(f"Nonconsecutive acquisition numbering: {recipe.key}/{region}")
        result[region] = selected
    for ag, carbon in zip(result["Ag3d"], result["C1s"], strict=True):
        if carbon.block != ag.block + 1 or carbon.group != ag.group:
            raise ValueError("Ag and C windows are not adjacent acquisitions in the same group")
    return result
