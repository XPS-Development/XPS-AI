"""Smoke tests for :class:`ui.tree_style.EditorTreeView` click/paint safety."""

from __future__ import annotations

import sys
from typing import cast

import pytest
from PySide6.QtCore import QModelIndex
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QApplication

from ui.tree_style import EditorTreeView, apply_editor_tree_style


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return cast(QApplication, app)


def test_editor_tree_view_click_selects_without_crash(qapp: QApplication) -> None:
    """Clicking a row selects it and survives a forced repaint."""
    del qapp
    model = QStandardItemModel()
    model.appendRow(QStandardItem("alpha"))
    model.appendRow(QStandardItem("beta"))

    view = EditorTreeView()
    view.setModel(model)
    apply_editor_tree_style(view)
    view.resize(240, 160)
    view.show()
    QApplication.processEvents()

    index = model.index(1, 0)
    assert index.isValid()
    view.setCurrentIndex(index)
    view.selectionModel().select(
        index,
        view.selectionModel().SelectionFlag.ClearAndSelect
        | view.selectionModel().SelectionFlag.Rows,
    )
    view.viewport().repaint()
    QApplication.processEvents()

    assert view.selectionModel().isRowSelected(1, QModelIndex())
    view.close()
