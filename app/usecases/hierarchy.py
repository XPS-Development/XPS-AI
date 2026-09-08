"""Hierarchy use-cases: bulk metadata rename and subtree removal."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.command.changes import CompositeChange, FullRemoveObject, SetMetadata
from core.metadata import SpectrumMetadata

if TYPE_CHECKING:
    from collections.abc import Callable

    from app.query_service import QueryService


class HierarchyUseCases:
    """Build Change objects for spectrum-tree hierarchy operations."""

    def __init__(self, query: QueryService) -> None:
        """
        Initialize hierarchy use-cases.

        Parameters
        ----------
        query
            Read-only query façade for collection and metadata access.
        """
        self._query = query

    def rename_spectrum(self, spectrum_id: str, new_name: str) -> SetMetadata:
        """
        Build a change that renames a single spectrum.

        Parameters
        ----------
        spectrum_id
            Spectrum identifier.
        new_name
            New display name.

        Returns
        -------
        SetMetadata
            Metadata update for the spectrum.
        """
        metadata = self._query.get_metadata(spectrum_id)
        if isinstance(metadata, SpectrumMetadata):
            updated = SpectrumMetadata(
                name=new_name,
                group=metadata.group,
                file=metadata.file,
            )
        else:
            updated = SpectrumMetadata(name=new_name, group="", file="")
        return SetMetadata(obj_id=spectrum_id, metadata=updated)

    def rename_group(
        self,
        file_label: str,
        old_group_label: str,
        new_group_label: str,
    ) -> CompositeChange | None:
        """
        Build a change that renames a group within a file bucket.

        Parameters
        ----------
        file_label
            File label whose group should be renamed.
        old_group_label
            Existing group label.
        new_group_label
            New group label.

        Returns
        -------
        CompositeChange or None
            Combined metadata updates, or None when no spectra match.
        """
        changes = self._metadata_updates_for_spectra(
            file_label=file_label,
            group_label=old_group_label,
            update=lambda md: SpectrumMetadata(
                name=md.name,
                group=new_group_label,
                file=md.file,
            ),
        )
        return CompositeChange(changes=changes) if changes else None

    def rename_file(self, old_file_label: str, new_file_label: str) -> CompositeChange | None:
        """
        Build a change that renames a file bucket for all matching spectra.

        Parameters
        ----------
        old_file_label
            Existing file label.
        new_file_label
            New file label.

        Returns
        -------
        CompositeChange or None
            Combined metadata updates, or None when no spectra match.
        """
        spectrum_ids = self._spectra_in_file(old_file_label)
        changes: list[SetMetadata] = []
        for spectrum_id in spectrum_ids:
            md = self._query.get_metadata(spectrum_id)
            if isinstance(md, SpectrumMetadata):
                updated = SpectrumMetadata(name=md.name, group=md.group, file=new_file_label)
                changes.append(SetMetadata(obj_id=spectrum_id, metadata=updated))
        return CompositeChange(changes=changes) if changes else None

    def remove_group(self, file_label: str, group_label: str) -> CompositeChange | None:
        """
        Build a change that removes all spectra in a file/group combination.

        Parameters
        ----------
        file_label
            File label whose group contents should be removed.
        group_label
            Group label to remove.

        Returns
        -------
        CompositeChange or None
            Combined full-remove changes, or None when no spectra match.
        """
        spectrum_ids = self._spectra_in_file_group(file_label, group_label)
        if not spectrum_ids:
            return None
        changes = [FullRemoveObject(obj_id=spectrum_id) for spectrum_id in spectrum_ids]
        return CompositeChange(changes=changes)

    def remove_file(self, file_label: str) -> CompositeChange | None:
        """
        Build a change that removes all spectra associated with a file label.

        Parameters
        ----------
        file_label
            File label whose spectra should be removed.

        Returns
        -------
        CompositeChange or None
            Combined full-remove changes, or None when no spectra match.
        """
        spectrum_ids = self._spectra_in_file(file_label)
        if not spectrum_ids:
            return None
        changes = [FullRemoveObject(obj_id=spectrum_id) for spectrum_id in spectrum_ids]
        return CompositeChange(changes=changes)

    def _spectra_in_file(self, file_label: str) -> tuple[str, ...]:
        """Return spectrum ids whose metadata file label matches exactly."""
        return self._query.find_objects(
            md_field="file",
            md_value=file_label,
            match_exact=True,
            tp=SpectrumMetadata,
        )

    def _spectra_in_file_group(self, file_label: str, group_label: str) -> tuple[str, ...]:
        """Return spectrum ids in the given file/group bucket."""
        return tuple(
            spectrum_id
            for spectrum_id in self._spectra_in_file(file_label)
            if self._spectrum_group(spectrum_id) == group_label
        )

    def _spectrum_group(self, spectrum_id: str) -> str | None:
        md = self._query.get_metadata(spectrum_id)
        if isinstance(md, SpectrumMetadata):
            return md.group
        return None

    def _metadata_updates_for_spectra(
        self,
        *,
        file_label: str,
        group_label: str,
        update: Callable[[SpectrumMetadata], SpectrumMetadata],
    ) -> list[SetMetadata]:
        changes: list[SetMetadata] = []
        for spectrum_id in self._spectra_in_file_group(file_label, group_label):
            md = self._query.get_metadata(spectrum_id)
            if isinstance(md, SpectrumMetadata):
                changes.append(SetMetadata(obj_id=spectrum_id, metadata=update(md)))
        return changes
