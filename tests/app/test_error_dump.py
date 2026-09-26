"""Tests for exception reporting helpers."""

from __future__ import annotations

import pytest

from app.error_dump import (
    format_exception_report,
    orchestrator_error_user_feedback_done,
    report_exception,
    safe_execution,
)


def test_format_exception_report_includes_type_message_and_traceback() -> None:
    """Report text should include type, message, and a traceback section."""
    try:
        raise ValueError("boom")
    except ValueError as exc:
        report = format_exception_report(exc)

    assert "Exception type: ValueError" in report
    assert "Exception message: boom" in report
    assert "Traceback:" in report
    assert "ValueError: boom" in report


def test_report_exception_marks_feedback_when_ui_disabled() -> None:
    """Without UI enabled, report still marks the exception as handled."""
    exc = RuntimeError("no ui")
    report_exception(exc)
    assert orchestrator_error_user_feedback_done(exc) is False
    # Second call is a no-op (already handled).
    report_exception(exc)


def test_safe_execution_reports_then_reraises() -> None:
    """Wrapped callables re-raise after reporting."""

    @safe_execution
    def boom() -> None:
        raise KeyError("missing")

    with pytest.raises(KeyError, match="missing"):
        boom()
