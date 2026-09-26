"""Shared file/group/spectrum grouping and structure-status dots."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator

from ..component_colors import STATUS_COLOR_EMPTY, STATUS_COLOR_PEAKS, STATUS_COLOR_REGIONS

ID_DISPLAY_CHARS = 5

STATUS_COLORS: dict[str, str] = {
    "empty": STATUS_COLOR_EMPTY,
    "regions": STATUS_COLOR_REGIONS,
    "peaks": STATUS_COLOR_PEAKS,
}
_STATUS_RANK = {"empty": 0, "regions": 1, "peaks": 2}

_NodeT = TypeVar("_NodeT")


class _SpectrumQuery(Protocol):
    """Read API used to group spectra and color their status dots."""

    def get_all_spectra_ids(self) -> Iterable[str]:
        """Return every spectrum id in the document."""

    def get_metadata(self, obj_id: str) -> object:
        """Return metadata for ``obj_id``, or None when none is stored."""

    def get_spectrum_structure_status(self, spectrum_id: str) -> str:
        """Return ``empty``, ``regions``, or ``peaks`` for one spectrum."""


def file_label(file_key: object) -> str:
    """Return the basename shown for a file group, or ``No file``."""
    if not file_key:
        return "No file"
    base = str(file_key).split("/")[-1]
    return base or "No file"


def group_label(group_key: object) -> str:
    """Return the group caption, or ``No group``."""
    if not group_key:
        return "No group"
    return str(group_key)


def spectrum_label(name: object) -> str:
    """Return the spectrum caption, or ``No name``."""
    if not name:
        return "No name"
    return str(name)


def _raw_key(value: object) -> str:
    """Return the metadata key stored for rename/remove, or ``""`` when missing."""
    if not value:
        return ""
    return str(value)


@dataclass(frozen=True)
class GroupedSpectrum:
    """One spectrum row inside a file/group bucket."""

    label: str
    raw_name: str
    spectrum_id: str


@dataclass(frozen=True)
class GroupedGroup:
    """One group bucket and the spectra inside it."""

    label: str
    raw_key: str
    spectra: tuple[GroupedSpectrum, ...]


@dataclass(frozen=True)
class GroupedFile:
    """One file bucket and the groups inside it."""

    label: str
    raw_key: str
    groups: tuple[GroupedGroup, ...]


def iter_spectrum_groups(
    query: _SpectrumQuery,
    *,
    skip_ids: Collection[str] = (),
) -> Iterator[GroupedFile]:
    """
    Yield spectra grouped by file, then group.

    ``raw_key`` / ``raw_name`` keep the metadata values used by rename and
    remove. ``label`` is the text shown in the tree.

    Parameters
    ----------
    query
        Read façade with spectrum ids and metadata.
    skip_ids
        Spectrum ids left out of the tree (for example the copy source).

    Yields
    ------
    GroupedFile
        One file bucket with its groups and spectra.
    """
    skip = set(skip_ids)
    grouped: dict[object, dict[object, list[tuple[object, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for spectrum_id in query.get_all_spectra_ids():
        if spectrum_id in skip:
            continue
        metadata = query.get_metadata(spectrum_id)
        file_key = getattr(metadata, "file", None)
        group_key = getattr(metadata, "group", None)
        name = getattr(metadata, "name", None)
        grouped[file_key][group_key].append((name, spectrum_id))

    def _sort_key(key: object) -> str:
        return str(key or "")

    for file_key, groups in sorted(grouped.items(), key=lambda item: _sort_key(item[0])):
        group_rows: list[GroupedGroup] = []
        for group_key, spectra in sorted(groups.items(), key=lambda item: _sort_key(item[0])):
            rows = tuple(
                GroupedSpectrum(spectrum_label(name), _raw_key(name), spectrum_id)
                for name, spectrum_id in sorted(spectra, key=lambda pair: str(pair[0] or ""))
            )
            group_rows.append(GroupedGroup(group_label(group_key), _raw_key(group_key), rows))
        yield GroupedFile(file_label(file_key), _raw_key(file_key), tuple(group_rows))


def collect_spectrum_ids(
    root: _NodeT,
    *,
    children: Callable[[_NodeT], Iterable[_NodeT]],
    spectrum_id_of: Callable[[_NodeT], str | None],
) -> list[str]:
    """
    Collect spectrum ids under ``root``, including ``root`` when it is a spectrum.

    Parameters
    ----------
    root
        Tree node to walk.
    children
        Returns the node's children.
    spectrum_id_of
        Returns the spectrum id for a spectrum node, otherwise None.
    """
    ids: list[str] = []
    seen: set[str] = set()

    def walk(node: _NodeT) -> None:
        spectrum_id = spectrum_id_of(node)
        if spectrum_id is not None and spectrum_id not in seen:
            seen.add(spectrum_id)
            ids.append(spectrum_id)
        for child in children(node):
            walk(child)

    walk(root)
    return ids


def structure_status(
    query: _SpectrumQuery,
    *,
    kind: str,
    object_id: str | None,
    child_spectrum_ids: Iterable[str],
) -> str:
    """
    Return the structure-dot status for a file, group, or spectrum node.

    A spectrum uses its own status. A file or group uses the strongest status
    among the spectra under it.

    Parameters
    ----------
    query
        Read façade that can report one spectrum's structure status.
    kind
        Node kind: ``file``, ``group``, ``spectrum``, or anything else.
    object_id
        Spectrum id when ``kind`` is ``spectrum``.
    child_spectrum_ids
        Spectrum ids under a file or group node.
    """
    if kind == "spectrum" and object_id is not None:
        return query.get_spectrum_structure_status(object_id)
    if kind in {"file", "group"}:
        best = "empty"
        for spectrum_id in child_spectrum_ids:
            status = query.get_spectrum_structure_status(spectrum_id)
            if _STATUS_RANK[status] > _STATUS_RANK[best]:
                best = status
        return best
    return "empty"
