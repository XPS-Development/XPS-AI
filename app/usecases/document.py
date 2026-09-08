"""Document use-cases: dump, load, and reset of the working collection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from app.command.core import CommandExecutor
    from app.parameters import AppParameters
    from app.serialization import SerializationService
    from core.collection import CoreCollection
    from core.services import MetadataService


class DocumentUseCases:
    """Persist and reset the document (collection + metadata + undo stack)."""

    def __init__(
        self,
        collection: CoreCollection,
        metadata: MetadataService,
        serialization: SerializationService,
        params: AppParameters,
        executor: CommandExecutor,
    ) -> None:
        """
        Initialize document use-cases.

        Parameters
        ----------
        collection
            Mutable spectrum collection.
        metadata
            Metadata store bound to the same document.
        serialization
            Dirty-tracking dump/load adapter.
        params
            Application parameters providing dump/load defaults.
        executor
            Command executor whose undo/redo stacks are cleared on replace/new.
        """
        self._collection = collection
        self._metadata = metadata
        self._serialization = serialization
        self._params = params
        self._executor = executor

    def dump_collection(
        self,
        path: str | Path | None = None,
        *,
        indent: int | None = None,
    ) -> Path:
        """
        Save collection and metadata to a JSON file.

        Parameters
        ----------
        path
            File path. If None, uses ``AppParameters.default_serialization_path``.
        indent
            JSON indentation. If None, uses parameter defaults.

        Returns
        -------
        Path
            Resolved path that was written.

        Raises
        ------
        ValueError
            If path is None and no default path is set.
        """
        from pathlib import Path as PathCls

        resolved_path = path if path is not None else self._params.default_serialization_path
        if resolved_path is None:
            raise ValueError(
                "path is required when AppParameters.default_serialization_path is not set"
            )
        resolved_indent = (
            indent if indent is not None else self._params.default_serialization_indent
        )
        self._serialization.dump(
            path=resolved_path,
            collection=self._collection,
            metadata_service=self._metadata,
            indent=resolved_indent,
            use_gzip=self._params.default_serialization_use_gzip,
            compresslevel=self._params.default_serialization_compresslevel,
        )
        self._params.default_serialization_path = resolved_path
        return PathCls(resolved_path)

    def load_collection(
        self,
        path: str | Path,
        *,
        mode: Literal["append", "replace"] | None = None,
    ) -> None:
        """
        Load collection and metadata from a JSON file.

        Parameters
        ----------
        path
            Path to the JSON file (plain or gzip-compressed).
        mode
            ``append`` or ``replace``. If None, uses parameter default.

        Raises
        ------
        ValueError
            If the resolved mode is not append or replace.
        """
        resolved_mode = mode if mode is not None else self._params.default_serialization_mode
        if resolved_mode not in ("append", "replace"):
            raise ValueError(
                f"mode must be 'append' or 'replace', got {resolved_mode!r}; "
                "AppParameters.default_serialization_mode='new' is not supported"
            )
        self._serialization.load(
            path=path,
            collection=self._collection,
            metadata_service=self._metadata,
            mode=resolved_mode,
        )
        self._params.default_serialization_path = path
        if resolved_mode == "replace":
            self._executor.clear()

    def new_collection(self) -> None:
        """Clear collection, metadata, undo stack, and reset the save path."""
        self._collection.clear()
        self._metadata.clear()
        self._params.default_serialization_path = None
        self._serialization.mark_dirty()
        self._executor.clear()
