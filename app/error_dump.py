"""Report unexpected exceptions to logs and an optional copyable Qt dialog."""

from __future__ import annotations

import functools
import logging
from traceback import format_exception
from typing import TYPE_CHECKING, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

logger = logging.getLogger(__name__)

_C = TypeVar("_C", bound=type[object])
_P = ParamSpec("_P")
_R = TypeVar("_R")

_ORCHESTRATOR_ERROR_USER_ATTR = "_ai_xps_orchestrator_error_user_notified"
_ORCHESTRATOR_ERROR_HANDLED_ATTR = "_ai_xps_orchestrator_error_handled"

_user_exception_ui_enabled: bool = False


def enable_user_exception_ui() -> None:
    """
    Turn on Qt dialogs for failures caught by :func:`safe_execution`.

    Call once a ``QApplication`` exists (typically right after constructing the
    application in ``main``). Headless tests and library use should leave this
    disabled so failures do not open dialogs.
    """
    global _user_exception_ui_enabled
    _user_exception_ui_enabled = True


def orchestrator_error_user_feedback_done(exc: BaseException) -> bool:
    """
    Return True if ``exc`` was already shown via :func:`report_exception`.

    Used by the Qt ``QApplication.notify`` override to avoid duplicate dialogs
    for the same exception object.

    Parameters
    ----------
    exc
        Exception propagated from orchestrator or UI code.

    Returns
    -------
    bool
        True when user feedback was already shown for ``exc``.
    """
    return getattr(exc, _ORCHESTRATOR_ERROR_USER_ATTR, False)


def format_exception_report(exc: BaseException) -> str:
    """
    Build a plain-text report with exception type, message, and traceback.

    Parameters
    ----------
    exc : BaseException
        Exception to format.

    Returns
    -------
    str
        Multi-line report suitable for display or clipboard copy.
    """
    traceback_text = "".join(format_exception(exc))
    return (
        f"Exception type: {type(exc).__name__}\n"
        f"Exception message: {exc}\n"
        f"\n"
        f"Traceback:\n"
        f"\n"
        f"{traceback_text}"
    )


def report_exception(
    exc: BaseException,
    *,
    title: str = "Unexpected error",
    context: str | None = None,
) -> None:
    """
    Log ``exc`` and show a copyable error dialog when UI feedback is enabled.

    Parameters
    ----------
    exc : BaseException
        Exception to report.
    title : str, optional
        Dialog window title.
    context : str or None, optional
        Optional log/context label (e.g. function qualname).
    """
    if getattr(exc, _ORCHESTRATOR_ERROR_HANDLED_ATTR, False):
        return
    setattr(exc, _ORCHESTRATOR_ERROR_HANDLED_ATTR, True)

    report = format_exception_report(exc)
    if context:
        logger.error("Error in %s\n%s", context, report)
    else:
        logger.error("%s", report)

    if not _user_exception_ui_enabled:
        return
    if orchestrator_error_user_feedback_done(exc):
        return
    _show_exception_dialog(title, report)
    setattr(exc, _ORCHESTRATOR_ERROR_USER_ATTR, True)


def _show_exception_dialog(title: str, details: str) -> None:
    """Open the copyable exception dialog when a Qt application is running."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    if QApplication.instance() is None:
        return
    from ui.error_dialog import show_exception_dialog

    show_exception_dialog(title, details)


def safe_execution(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Wrap a callable so failures are logged, shown to the user, then re-raised.

    On :class:`Exception`, calls :func:`report_exception` (dialog when
    :func:`enable_user_exception_ui` has been called), then re-raises.

    Parameters
    ----------
    func
        Function or method to wrap.

    Returns
    -------
    Callable[..., Any]
        Wrapped callable with the same public metadata as ``func``.
    """

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            func_name = getattr(
                func, "__qualname__", getattr(func, "__name__", type(func).__name__)
            )
            report_exception(exc, title="Error", context=func_name)
            raise

    return wrapper


def apply_safe_execution_to_class(
    cls: _C,
    *,
    skip: Collection[str] = (
        "__init__",
        "execute",
        "execute_optional",
        "undo",
        "redo",
        "peek_undo_command",
        "peek_redo_command",
        "consume_pending_ui_refresh",
    ),
) -> _C:
    """
    Apply :func:`safe_execution` to each function defined on ``cls``.

    Skips special descriptors (``property``, ``classmethod``, ``staticmethod``)
    and any name listed in ``skip`` (``'__init__'`` by default).

    Parameters
    ----------
    cls
        Class whose methods should be wrapped.
    skip
        Method names to leave unchanged.

    Returns
    -------
    type
        The same class object (mutated in place).
    """
    skip_set = set(skip)
    for name, attr in cls.__dict__.items():
        if name in skip_set:
            continue
        if isinstance(attr, (classmethod, staticmethod, property)):
            continue
        if callable(attr):
            setattr(cls, name, safe_execution(attr))
    return cls
