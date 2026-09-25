"""Tests for save-dialog path helpers."""

from __future__ import annotations

from pathlib import Path

from ui.file_dialogs import ensure_suffix_from_filter, split_open_paths


def test_ensure_suffix_appends_from_filter_when_missing() -> None:
    """Linux-style bare names pick up the selected filter extension."""
    assert ensure_suffix_from_filter("collection", "JSON files (*.json)") == Path("collection.json")
    assert ensure_suffix_from_filter("/tmp/out", "CSV files (*.csv);;Text files (*.txt)") == Path(
        "/tmp/out.csv"
    )
    assert ensure_suffix_from_filter("doc", "Gzip JSON (*.json.gz)") == Path("doc.json.gz")


def test_ensure_suffix_keeps_existing_extension() -> None:
    """Paths that already have a suffix are left unchanged."""
    assert ensure_suffix_from_filter("data.json", "JSON files (*.json)") == Path("data.json")
    assert ensure_suffix_from_filter("notes.txt", "CSV files (*.csv)") == Path("notes.txt")


def test_ensure_suffix_uses_fallback_for_all_files_filter() -> None:
    """All-files filter has no pattern; fallback extension is applied."""
    assert ensure_suffix_from_filter("untitled", "All files (*)", fallback=".json") == Path(
        "untitled.json"
    )
    assert ensure_suffix_from_filter("untitled", "All files (*)", fallback=".json.gz") == Path(
        "untitled.json.gz"
    )
    assert ensure_suffix_from_filter("bare", "All files (*)") == Path("bare")


def test_split_open_paths_treats_bare_and_gzip_as_collections() -> None:
    """Extensionless and gzip saves are openable as collections."""
    spectra, collections = split_open_paths(
        [
            "/data/a.txt",
            "/data/b.json",
            "/data/c.json.gz",
            "/data/noext",
            "/data/d.vms",
        ]
    )
    assert spectra == [Path("/data/a.txt"), Path("/data/d.vms")]
    assert collections == [
        Path("/data/b.json"),
        Path("/data/c.json.gz"),
        Path("/data/noext"),
    ]
