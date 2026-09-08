"""Tests for :class:`app.usecases.document.DocumentUseCases`."""

from pathlib import Path

from app.command.core import CommandExecutor, UndoRedoStack, create_default_registry
from app.parameters import AppParameters
from app.serialization import SerializationService
from app.usecases.document import DocumentUseCases
from core.services import CoreContext


def _document(collection, params: AppParameters | None = None) -> DocumentUseCases:
    ctx = CoreContext.from_collection(collection)
    serialization = SerializationService()
    params = params or AppParameters()
    executor = CommandExecutor(ctx, UndoRedoStack(), create_default_registry())
    return DocumentUseCases(
        collection,
        ctx.metadata,
        serialization,
        params,
        executor,
    )


def test_dump_and_load_roundtrip(simple_collection, spectrum_id: str, tmp_path) -> None:
    """dump_collection writes a file that load_collection can restore."""
    from core.collection import CoreCollection

    params = AppParameters()
    doc = _document(simple_collection, params)
    path = tmp_path / "doc.json"

    doc.dump_collection(path)
    assert path.exists()
    assert params.default_serialization_path == path

    empty = CoreCollection()
    loaded = _document(empty, AppParameters())
    loaded.load_collection(path, mode="replace")
    assert spectrum_id in empty.objects_index


def test_new_collection_clears_state(simple_collection, spectrum_id: str) -> None:
    """new_collection empties the collection and marks the document dirty."""
    params = AppParameters(default_serialization_path=Path("/tmp/x.json"))
    serialization = SerializationService()
    ctx = CoreContext.from_collection(simple_collection)
    executor = CommandExecutor(ctx, UndoRedoStack(), create_default_registry())
    doc = DocumentUseCases(
        simple_collection,
        ctx.metadata,
        serialization,
        params,
        executor,
    )

    doc.new_collection()

    assert spectrum_id not in simple_collection.objects_index
    assert params.default_serialization_path is None
    assert serialization.is_dirty is True
    assert executor.stack.can_undo is False
