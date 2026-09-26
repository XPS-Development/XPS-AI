"""Shared label-and-id search text for editor trees."""

from __future__ import annotations


def make_search_text(*parts: str | None) -> str:
    """
    Join display parts into one lowercase search string.

    Empty parts are skipped. A spectrum, region, or component row passes its
    label and id; a file or group row passes only its label.

    Parameters
    ----------
    *parts : str or None
        Label, identifier, and any other text that should match.

    Returns
    -------
    str
        Space-separated lowercase text, or ``""`` when every part is empty.
    """
    tokens = [part.strip() for part in parts if part and part.strip()]
    return " ".join(tokens).lower()


def matches_search(needle: str, search_text: str, label: str = "") -> bool:
    """
    Return True when ``needle`` is empty or found in the search string or label.

    Parameters
    ----------
    needle : str
        Text typed into a search box.
    search_text : str
        Prebuilt lowercase string from :func:`make_search_text`.
    label : str, optional
        Display label, checked as well when it is not already in ``search_text``.

    Returns
    -------
    bool
        Whether the row matches.
    """
    query = needle.strip().lower()
    if not query:
        return True
    if query in search_text:
        return True
    return query in label.lower()
