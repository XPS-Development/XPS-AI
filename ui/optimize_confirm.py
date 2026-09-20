"""Confirm expression-scope expansion before running Optimize."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import QMessageBox

from core.fitting.fit_scope import format_expression_problems

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PySide6.QtWidgets import QWidget

    from .controller import ControllerWrapper


def confirm_and_optimize(
    parent: QWidget | None,
    controller: ControllerWrapper,
    *,
    region_ids: Sequence[str] | None = None,
    spectrum_ids: Sequence[str] | None = None,
    **kwargs: Any,
) -> bool:
    """
    Validate expressions, warn on scope expansion, then run optimization.

    Parameters
    ----------
    parent : QWidget or None
        Parent for confirmation / error dialogs.
    controller : ControllerWrapper
        Application controller.
    region_ids : sequence of str, optional
        Regions to optimize.
    spectrum_ids : sequence of str, optional
        Spectra whose regions to optimize.
    **kwargs
        Forwarded to :meth:`ControllerWrapper.optimize_regions`.

    Returns
    -------
    bool
        True if optimization ran, False if the user cancelled or exprs are invalid.
    """
    preview = controller.preview_fit_scope(region_ids=region_ids, spectrum_ids=spectrum_ids)
    if preview.has_expression_errors:
        QMessageBox.warning(
            parent,
            "Invalid expressions",
            format_expression_problems(preview.expression_problems),
        )
        return False

    if preview.needs_confirmation:
        n_regions = preview.extra_region_count
        n_spectra = preview.extra_spectrum_count
        region_word = "region" if n_regions == 1 else "regions"
        spectrum_word = "spectrum" if n_spectra == 1 else "spectra"
        answer = QMessageBox.question(
            parent,
            "Linked regions",
            (
                f"Optimization will also affect {n_regions} more {region_word} "
                f"in {n_spectra} {spectrum_word}. Continue?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False

    controller.optimize_regions(
        region_ids=region_ids,
        spectrum_ids=spectrum_ids,
        **kwargs,
    )
    return True
