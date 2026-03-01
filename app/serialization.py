"""
App-level serialization service for collection and metadata.

Wraps tools/serialization and tracks default save path and dirty state.
Caller (orchestrator) is responsible for clearing undo/redo and replacing
context when loading with replace or new mode.
"""

from pathlib import Path
from typing import Literal

from core.collection import CoreCollection
from core.services import MetadataService
from tools.serialization import CollectionSerializer


class SerializationService:
    """
    App-level service for saving/loading collection and metadata.

    Tracks whether the current state is saved and the default save path.
    Does not clear the undo/redo stack or replace orchestrator state;
    the orchestrator must do that when loading with replace or new mode.
    """

    def __init__(
        self,
        collection: CoreCollection,
        metadata_service: MetadataService,
        *,
        serializer: CollectionSerializer | None = None,
    ) -> None:
        """
        Initialize the manager with collection and metadata service.

        Parameters
        ----------
        collection : CoreCollection
            The collection to serialize/deserialize.
        metadata_service : MetadataService
            The metadata service for the same collection.
        serializer : CollectionSerializer or None, optional
            Serializer instance; if None, a new one is created.
        """
        self._collection = collection
        self._metadata_service = metadata_service
        self._serializer = serializer if serializer is not None else CollectionSerializer()
        self._default_path: Path | None = None
        self._dirty = False

    @property
    def is_saved(self) -> bool:
        """True if there is a default path and no unsaved changes."""
        return self._default_path is not None and not self._dirty

    def set_default_path(self, path: str | Path) -> None:
        """Set the default save/load path."""
        self._default_path = Path(path)

    def get_default_path(self) -> Path | None:
        """Return the current default save path, or None."""
        return self._default_path

    def mark_dirty(self) -> None:
        """Mark the document as having unsaved changes."""
        self._dirty = True

    def dump(
        self,
        path: str | Path | None = None,
        *,
        indent: int | None = None,
    ) -> None:
        """
        Serialize collection and metadata to a JSON file.

        Parameters
        ----------
        path : str or Path or None, optional
            File path. If None, uses the default path from set_default_path.
        indent : int or None, optional
            JSON indentation level.

        Raises
        ------
        ValueError
            If path is None and no default path is set.
        """
        if path is not None:
            fp = Path(path)
        elif self._default_path is not None:
            fp = self._default_path
        else:
            raise ValueError("No path provided and no default path set")
        self._serializer.dump(
            self._collection,
            fp,
            metadata_service=self._metadata_service,
            indent=indent,
        )
        self._default_path = fp
        self._dirty = False

    def load(
        self,
        path: str | Path,
        *,
        mode: Literal["append", "replace", "new"] = "replace",
    ) -> tuple[CoreCollection, MetadataService] | None:
        """
        Load collection and metadata from a JSON file.

        For append/replace, uses the manager's collection and metadata_service.
        For new, the serializer creates new instances; the caller must replace
        orchestrator state and create a new SerializationManager for them.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file.
        mode : {"append", "replace", "new"}, optional
            - append: add loaded objects to existing collection/metadata.
            - replace: clear then fill existing collection/metadata in-place.
            - new: create new collection and metadata service; return them.

        Returns
        -------
        tuple[CoreCollection, MetadataService] or None
            For mode "new": (new_collection, new_metadata_service). For append
            or replace: None (state updated in-place).

        Notes
        -----
        On success, default path is set and dirty is cleared. For replace/new
        the caller must clear the execution manager (undo/redo stack) and for
        new must replace collection/context and wire a new SerializationManager.
        """
        path = Path(path)
        if mode == "new":
            result = self._serializer.load(
                path,
                collection=None,
                metadata_service=None,
                mode="new",
            )
            assert isinstance(result, tuple)
            self._default_path = path
            self._dirty = False
            return result
        result = self._serializer.load(
            path,
            collection=self._collection,
            metadata_service=self._metadata_service,
            mode=mode,
        )
        self._default_path = path
        self._dirty = False
        return None
