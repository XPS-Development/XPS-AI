"""UI invalidation flags for undoable commands."""

from __future__ import annotations

from enum import IntFlag, auto


class UiRefresh(IntFlag):
    """Bit flags indicating which UI areas need refresh after a command."""

    HIERARCHY = auto()
    PLOT = auto()
    PROPERTIES = auto()
    DOCUMENT = auto()
    POST_PROCESSING = auto()

    ALL = HIERARCHY | PLOT | PROPERTIES | DOCUMENT | POST_PROCESSING
    FIT = PLOT | PROPERTIES | DOCUMENT | POST_PROCESSING
    METADATA = HIERARCHY | PROPERTIES | DOCUMENT
