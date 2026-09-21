"""Helpers for expression component-id tokens in the UI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def shortest_unique_prefix(
    component_id: str,
    all_ids: Iterable[str],
    *,
    min_len: int = 5,
) -> str:
    """
    Return the shortest unique ``startswith`` prefix of ``component_id``.

    Matches the resolution rules used by ``match_component_reference``: a
    prefix is unique when exactly one known id starts with it. Prefers at
    least ``min_len`` characters when the id is long enough (tree display
    uses 5).

    Parameters
    ----------
    component_id : str
        Full component identifier to abbreviate.
    all_ids : iterable of str
        Known component ids in the insertion scope.
    min_len : int, optional
        Prefer prefixes at least this long when possible.

    Returns
    -------
    str
        Unique prefix, or the full id if no shorter unique prefix exists.
    """
    if not component_id:
        return component_id
    ids = {cid for cid in all_ids if cid}
    ids.add(component_id)
    start = min(max(1, min_len), len(component_id))
    for length in range(start, len(component_id) + 1):
        prefix = component_id[:length]
        matches = [cid for cid in ids if cid.startswith(prefix)]
        if len(matches) == 1:
            return prefix
    return component_id
