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
        use_gzip: bool = False,
        compresslevel: int = 9,
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
        use_gzip : bool, optional
            If True, write gzip-compressed JSON.
        compresslevel : int, optional
            Gzip compression level (0--9) when ``use_gzip`` is True.

        """
        dump_collection(
            collection=collection,
            fp=path,
            metadata_service=metadata_service,
            indent=indent,
            use_gzip=use_gzip,
            compresslevel=compresslevel,
        )
        self._dirty = False

    def load(
        self,
        path: str | Path,
        collection: CoreCollection,
        metadata_service: MetadataService,
        *,
        mode: Literal["append", "replace"] = "replace",  # "new" is not supported
        use_gzip: bool | None = None,
    ) -> None:
        """
        Load collection and metadata from a JSON file.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file (plain or gzip-compressed).
        collection : CoreCollection
            Collection to load into.
        metadata_service : MetadataService
            Metadata service for the collection.
        mode : {"append", "replace"}, optional
            - append: add loaded objects to existing collection/metadata.
            - replace: clear then fill existing collection/metadata in-place.
        use_gzip : bool or None, optional
            If True, read as gzip. If False, plain text. If None, detect from
            path suffix or file magic bytes.

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
            use_gzip=use_gzip,
        )
        self._dirty = False
