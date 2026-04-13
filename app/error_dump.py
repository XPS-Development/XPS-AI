from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Collection
from datetime import datetime
from pathlib import Path
from traceback import format_exception
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_C = TypeVar("_C", bound=type[Any])

_ORCHESTRATOR_ERROR_USER_ATTR = "_ai_xps_orchestrator_error_user_notified"

_user_exception_ui_enabled: bool = False


def enable_user_exception_ui() -> None:
    """
    Turn on Qt message boxes for failures caught by :func:`safe_execution`.

    Call once a ``QApplication`` exists (typically right after constructing the
    application in ``main``). Headless tests and library use should leave this
    disabled so failures do not open dialogs.
    """
    global _user_exception_ui_enabled
    _user_exception_ui_enabled = True


def orchestrator_error_user_feedback_done(exc: BaseException) -> bool:
    """
    Return True if ``exc`` was already persisted and shown via ``safe_execution``.

    Used by the Qt ``QApplication.notify`` override to avoid duplicate dialogs
    and dump files for the same exception object.

    Parameters
    ----------
    exc
        Exception propagated from orchestrator code.

    Returns
    -------
    bool
        True when :func:`safe_execution` already handled user feedback.
    """
    return getattr(exc, _ORCHESTRATOR_ERROR_USER_ATTR, False)


def _notify_orchestrator_error_ui(exc: BaseException, dump_path: Path) -> None:
    """Show a modal error dialog when UI notifications are enabled."""
    if not _user_exception_ui_enabled:
        return
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        return
    app = QApplication.instance()
    if app is None:
        return
    parent = app.activeWindow()
    QMessageBox.critical(
        parent,
        "Error",
        f"{exc}\n\nDetails were saved to:\n{dump_path}",
    )
    setattr(exc, _ORCHESTRATOR_ERROR_USER_ATTR, True)


def safe_execution(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wrap a callable so failures are persisted, logged, optionally shown in Qt,
    then re-raised.

    On :class:`Exception`, writes ``error_dumps`` via :func:`save_error_dump`,
    logs with :meth:`logging.Logger.exception`, and when
    :func:`enable_user_exception_ui` has been called, shows ``QMessageBox``.

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
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            dump_path = save_error_dump(exc)
            logger.exception("Error in %s", func.__qualname__)
            _notify_orchestrator_error_ui(exc, dump_path)
            raise

    return wrapper


def apply_safe_execution_to_class(
    cls: _C,
    *,
    skip: Collection[str] = ("__init__",),
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


def save_error_dump(exc: BaseException) -> Path:
    """
    Save an error dump with full traceback to the error_dumps folder.

    The dump file is created under ``Path.cwd() / "error_dumps"`` with a
    timestamped name. The file contains the exception type, message, and
    full traceback.

    Parameters
    ----------
    exc : BaseException
        Exception instance to serialize.

    Returns
    -------
    Path
        Path to the written dump file.
    """
    dumps_dir = Path.cwd() / "error_dumps"
    dumps_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"error_{timestamp}.txt"
    dump_path = dumps_dir / filename

    lines = format_exception(exc)
    dump_text = "".join(lines)

    header = [
        f"Exception type: {type(exc).__name__}",
        f"Exception message: {exc}",
        "",
        "Traceback:",
        "",
    ]
    dump_path.write_text("\n".join(header) + dump_text, encoding="utf-8")
    return dump_path
