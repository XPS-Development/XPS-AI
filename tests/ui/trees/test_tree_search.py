"""Tests for shared tree search text."""

from ui.trees.tree_search import make_search_text, matches_search


def test_make_search_text_joins_label_and_id() -> None:
    """A row's search string is the lowercase label plus its id."""
    assert make_search_text("Spec 1", "s1") == "spec 1 s1"
    assert make_search_text("No file") == "no file"
    assert make_search_text("", None, "  ") == ""


def test_matches_search_finds_label_or_id() -> None:
    """An empty query matches; otherwise the needle must sit in the text or label."""
    text = make_search_text("Spec 1", "s1")
    assert matches_search("", text, "Spec 1")
    assert matches_search("s1", text, "Spec 1")
    assert matches_search("SPEC", text, "Spec 1")
    assert not matches_search("s2", text, "Spec 1")
