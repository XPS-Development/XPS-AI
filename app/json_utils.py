"""
JSON parsing helpers for application code.

These functions keep JSON-related ``try``/``except`` out of UI modules while
still providing structured error reporting to callers.
"""

from __future__ import annotations

import json
from typing import Any


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Parse ``text`` as a JSON object (mapping).

    Parameters
    ----------
    text
        Raw JSON text, possibly surrounded by whitespace.

    Returns
    -------
    tuple[dict[str, Any] | None, str | None]
        ``(mapping, None)`` on success, or ``(None, error_message)`` on failure.
    """
    cleaned = text.strip()
    if not cleaned:
        return ({}, None)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return (None, f"Invalid JSON: {exc}")
    if not isinstance(value, dict):
        return (None, "JSON must be an object (mapping).")
    return (value, None)
