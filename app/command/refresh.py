"""UI invalidation flags for undoable commands."""

from __future__ import annotations

from enum import IntFlag, auto


class UiRefresh(IntFlag):
    """Bit flags indicating which UI areas need refresh after a command."""

    HIERARCHY = auto()
    PLOT = auto()
    PROPERTIES = auto()
    DOCUMENT = auto()

    ALL = HIERARCHY | PLOT | PROPERTIES | DOCUMENT
    FIT = PLOT | PROPERTIES | DOCUMENT
    METADATA = HIERARCHY | PROPERTIES | DOCUMENT
