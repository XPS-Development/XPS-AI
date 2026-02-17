"""Tests for import service."""

import pytest

from core.metadata import SpectrumMetadata
from app.import_service import import_spectra
from app.command.changes import CompositeChange, CreateSpectrum, SetSpectrumMetadata
from app.command.core import CommandExecutor, UndoRedoStack, create_default_registry


def test_import_spectra_returns_composite_change():
    """import_spectra returns a CompositeChange."""
    change = import_spectra("tests/data/test_1_spec.txt")
    assert isinstance(change, CompositeChange)
    assert len(change.changes) >= 2


def test_import_spectra_creates_spectrum_and_metadata_changes():
    """import_spectra produces CreateSpectrum and SetSpectrumMetadata per spectrum."""
    change = import_spectra("tests/data/test_1_spec.txt")
    create_changes = [c for c in change.changes if isinstance(c, CreateSpectrum)]
    set_md_changes = [c for c in change.changes if isinstance(c, SetSpectrumMetadata)]
    assert len(create_changes) == 1
    assert len(set_md_changes) == 1


def test_import_spectra_execute_via_command_executor(empty_collection):
    """import_spectra change can be executed via CommandExecutor."""
    from app.command.utils import ApplicationContext

    ctx = ApplicationContext.from_collection(empty_collection)
    stack = UndoRedoStack()
    executor = CommandExecutor(ctx, stack, create_default_registry())

    change = import_spectra("tests/data/test_1_spec.txt")
    executor.execute(change)

    assert len(empty_collection.objects_index) >= 1
    spectrum_ids = [
        oid for oid, obj in empty_collection.objects_index.items()
        if obj.__class__.__name__ == "Spectrum"
    ]
    assert len(spectrum_ids) == 1
    metadata = ctx.metadata.get_spectrum_metadata(spectrum_ids[0])
    assert metadata.name == "Ag3d"
    assert "test_1_spec.txt" in metadata.file


def test_import_spectra_undo_removes_spectra(empty_collection):
    """Undo after import removes the created spectra."""
    from app.command.utils import ApplicationContext

    ctx = ApplicationContext.from_collection(empty_collection)
    stack = UndoRedoStack()
    executor = CommandExecutor(ctx, stack, create_default_registry())

    change = import_spectra("tests/data/test_1_spec.txt")
    executor.execute(change)
    assert len(empty_collection.objects_index) >= 1

    executor.undo()
    assert len(empty_collection.objects_index) == 0


def test_import_spectra_vamas_creates_multiple_spectra(empty_collection):
    """Import from VAMAS file creates one spectrum per block."""
    from app.command.utils import ApplicationContext

    ctx = ApplicationContext.from_collection(empty_collection)
    stack = UndoRedoStack()
    executor = CommandExecutor(ctx, stack, create_default_registry())

    change = import_spectra("tests/data/test_1_spec.vms")
    executor.execute(change)

    spectrum_count = sum(
        1 for obj in empty_collection.objects_index.values()
        if obj.__class__.__name__ == "Spectrum"
    )
    assert spectrum_count >= 1
