"""Confirm expression-scope expansion before running Optimize."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from PySide6.QtWidgets import QMessageBox

from core.fitting.fit_scope import format_expression_problems

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PySide6.QtWidgets import QWidget

    from .controller import ControllerWrapper

FitLinkChoice = Literal["include", "selection_only", "cancel"]


def _ask_fit_link_choice(
    parent: QWidget | None,
    *,
    extra_region_count: int,
    extra_spectrum_count: int,
) -> FitLinkChoice:
    """
    Ask whether to include linked regions or fit the selection only.

    Parameters
    ----------
    parent : QWidget or None
        Parent for the dialog.
    extra_region_count, extra_spectrum_count : int
        How many regions / spectra would be pulled in by expressions.

    Returns
    -------
    {"include", "selection_only", "cancel"}
        User choice.
    """
    n_regions = extra_region_count
    n_spectra = extra_spectrum_count
    region_word = "region" if n_regions == 1 else "regions"
    spectrum_word = "spectrum" if n_spectra == 1 else "spectra"

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Linked regions")
    box.setText(
        f"Expressions also link {n_regions} more {region_word} in {n_spectra} {spectrum_word}."
    )
    box.setInformativeText(
        "Include linked regions in this optimization, or ignore those links "
        "and optimize only the current selection?"
    )
    include_btn = box.addButton("Include linked", QMessageBox.ButtonRole.AcceptRole)
    selection_btn = box.addButton("Selection only", QMessageBox.ButtonRole.ActionRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(include_btn)
    box.exec()

    clicked = box.clickedButton()
    if clicked == include_btn:
        return "include"
    if clicked == selection_btn:
        return "selection_only"
    return "cancel"


def confirm_and_optimize(
    parent: QWidget | None,
    controller: ControllerWrapper,
    *,
    region_ids: Sequence[str] | None = None,
    spectrum_ids: Sequence[str] | None = None,
    **kwargs: Any,
) -> bool:
    """
    Validate expressions, offer fit-scope choices, then run optimization.

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

    # Selection-only errors always block (typos in what the user is fitting).
    if preview.has_selected_expression_errors:
        QMessageBox.warning(
            parent,
            "Invalid expressions",
            format_expression_problems(preview.selected_expression_problems),
        )
        return False

    expand_linked = True
    if preview.needs_confirmation:
        choice = _ask_fit_link_choice(
            parent,
            extra_region_count=preview.extra_region_count,
            extra_spectrum_count=preview.extra_spectrum_count,
        )
        if choice == "cancel":
            return False
        if choice == "selection_only":
            expand_linked = False
        elif preview.has_expression_errors:
            # Linked regions bring additional broken exprs.
            QMessageBox.warning(
                parent,
                "Invalid expressions",
                format_expression_problems(preview.expression_problems),
            )
            return False

    controller.optimize_regions(
        region_ids=region_ids,
        spectrum_ids=spectrum_ids,
        expand_linked=expand_linked,
        **kwargs,
    )
    return True
