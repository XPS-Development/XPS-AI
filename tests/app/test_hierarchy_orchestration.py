"""Integration tests for hierarchy rename operations via AppOrchestrator."""

from core.metadata import SpectrumMetadata


def test_rename_spectrum_persists_and_undo_restores(hierarchy_orchestrator) -> None:
    """rename_spectrum updates metadata and undo restores the previous name."""
    orch = hierarchy_orchestrator

    orch.rename_spectrum("s1", "Renamed")
    metadata = orch.query.get_metadata("s1")
    assert isinstance(metadata, SpectrumMetadata)
    assert metadata.name == "Renamed"
    assert orch.can_undo is True
    assert orch.is_dirty is True

    orch.undo()
    restored = orch.query.get_metadata("s1")
    assert isinstance(restored, SpectrumMetadata)
    assert restored.name == "spec-1"
    assert orch.can_redo is True


def test_rename_group_updates_only_target_group(hierarchy_orchestrator) -> None:
    """rename_group updates spectra in the matching file/group bucket only."""
    orch = hierarchy_orchestrator

    orch.rename_group("file-a", "group-a", "group-c")

    for spectrum_id in ("s1", "s2"):
        metadata = orch.query.get_metadata(spectrum_id)
        assert isinstance(metadata, SpectrumMetadata)
        assert metadata.group == "group-c"

    metadata_s3 = orch.query.get_metadata("s3")
    assert isinstance(metadata_s3, SpectrumMetadata)
    assert metadata_s3.group == "group-b"


def test_rename_file_updates_all_spectra_in_file(hierarchy_orchestrator) -> None:
    """rename_file updates metadata.file for every spectrum in the file bucket."""
    orch = hierarchy_orchestrator

    orch.rename_file("file-a", "file-renamed")

    for spectrum_id in ("s1", "s2", "s3"):
        metadata = orch.query.get_metadata(spectrum_id)
        assert isinstance(metadata, SpectrumMetadata)
        assert metadata.file == "file-renamed"


def test_rename_group_no_match_does_not_push_undo(hierarchy_orchestrator) -> None:
    """rename_group with no matching spectra is a no-op on the undo stack."""
    orch = hierarchy_orchestrator
    before_s1 = orch.query.get_metadata("s1")

    orch.rename_group("missing-file", "group-a", "group-c")

    assert orch.query.get_metadata("s1") == before_s1
    assert orch.can_undo is False
