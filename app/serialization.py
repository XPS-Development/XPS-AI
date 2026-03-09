"""
App-level serialization service for collection and metadata.

Wraps tools/serialization and tracks dirty state.
Caller (orchestrator) is responsible for clearing undo/redo and replacing
context when loading with replace mode.
"""

from pathlib import Path
from typing import Literal

from core.collection import CoreCollection
from core.services import MetadataService
from tools.serialization import dump as dump_collection, load as load_collection


class SerializationService:
    """
    App-level service for saving/loading collection and metadata.

    Tracks whether the current state is dirty.
    Does not clear the undo/redo stack or replace orchestrator state;
    the orchestrator must do that when loading with replace mode.
    """

    def __init__(self) -> None:
        """
        Initialize the serialization service.
        """
        self._dirty = False

    @property
    def is_dirty(self) -> bool:
        """True if there are unsaved changes."""
        return self._dirty

    def mark_dirty(self) -> None:
        """Mark the document as having unsaved changes."""
        self._dirty = True

    def dump(
        self,
        path: str | Path,
        collection: CoreCollection,
        metadata_service: MetadataService,
        *,
        indent: int | None = None,
    ) -> None:
        """
        Serialize collection and metadata to a JSON file.

        Parameters
        ----------
        path : str or Path
            File path.
        collection : CoreCollection
            Collection to serialize.
        metadata_service : MetadataService
            Metadata service for the collection.
        indent : int or None, optional
            JSON indentation level.

        """
        dump_collection(
            collection=collection,
            fp=path,
            metadata_service=metadata_service,
            indent=indent,
        )
        self._dirty = False

    def load(
        self,
        path: str | Path,
        collection: CoreCollection,
        metadata_service: MetadataService,
        *,
        mode: Literal["append", "replace"] = "replace",  # "new" is not supported
    ) -> None:
        """
        Load collection and metadata from a JSON file.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file.
        collection : CoreCollection
            Collection to load into.
        metadata_service : MetadataService
            Metadata service for the collection.
        mode : {"append", "replace"}, optional
            - append: add loaded objects to existing collection/metadata.
            - replace: clear then fill existing collection/metadata in-place.

        Note: new mode is not supported.

        Notes
        -----
        On success, dirty is cleared. For replace, the caller must clear the
        execution manager (undo/redo stack) and replace collection/context.
        """
        load_collection(
            fp=path,
            collection=collection,
            metadata_service=metadata_service,
            mode=mode,
        )
        self._dirty = False
