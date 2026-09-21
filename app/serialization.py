"""
App-level serialization service for collection and metadata.

Wraps core.io.serialization for dump/load. Dirty state is tracked on the
undo stack (:class:`~app.command.core.UndoRedoStack`); the orchestrator
calls :meth:`UndoRedoStack.mark_saved` after a successful save or load.
"""

from pathlib import Path
from typing import Literal

from core.collection import CoreCollection
from core.io.serialization import dump as dump_collection
from core.io.serialization import load as load_collection
from core.services import MetadataService


class SerializationService:
    """
    App-level service for saving/loading collection and metadata.

    Does not clear the undo/redo stack or replace orchestrator state;
    the orchestrator must do that when loading.
    """

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

    def load(
        self,
        path: str | Path,
        collection: CoreCollection,
        metadata_service: MetadataService,
        *,
        mode: Literal["append", "replace"] = "replace",
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

        Notes
        -----
        For replace, the caller must clear the execution manager (undo/redo stack).
        """
        load_collection(
            fp=path,
            collection=collection,
            metadata_service=metadata_service,
            mode=mode,
            use_gzip=use_gzip,
        )
