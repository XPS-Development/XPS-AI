"""Smoke tests for :class:`ui.tree_style.EditorTreeView` click/paint safety."""

from __future__ import annotations

import sys
from typing import Any, cast

import pytest
from PySide6.QtCore import QModelIndex, QRect
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QApplication, QWidget

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


def test_apply_editor_tree_style_sets_editor_rows_property(qapp: QApplication) -> None:
    """``editorRows`` is hairlines by default and spaced when separators are off."""
    del qapp
    view = EditorTreeView()
    apply_editor_tree_style(view)
    assert view.property("editorRows") == "hairlines"
    apply_editor_tree_style(view, row_separators=False)
    assert view.property("editorRows") == "spaced"
    view.close()


def test_hover_row_invalidates_only_row_rects(qapp: QApplication) -> None:
    """Changing the hovered row updates only the old and new row strips."""
    del qapp
    model = QStandardItemModel()
    model.appendRow(QStandardItem("alpha"))
    model.appendRow(QStandardItem("beta"))
    model.appendRow(QStandardItem("gamma"))

    view = EditorTreeView()
    view.setModel(model)
    apply_editor_tree_style(view)
    view.resize(240, 200)
    view.show()
    QApplication.processEvents()

    view._set_hover_row(model.index(0, 0))
    first = QRect(view._hover_rect)
    assert not first.isNull()
    assert first.width() == view.viewport().width()

    updates: list[QRect | None] = []
    viewport = view.viewport()

    def _spy(*args: Any) -> None:
        if not args:
            updates.append(None)
        elif len(args) == 1:
            rect = args[0]
            updates.append(QRect(rect) if isinstance(rect, QRect) else None)
        else:
            updates.append(QRect(int(args[0]), int(args[1]), int(args[2]), int(args[3])))
        QWidget.update(viewport, *args)

    viewport.update = _spy  # ty: ignore[invalid-assignment]

    view._set_hover_row(model.index(2, 0))
    assert None not in updates
    assert len(updates) == 2
    assert updates[0] == first
    second = QRect(view._hover_rect)
    assert updates[1] == second
    assert second.y() != first.y()

    updates.clear()
    view._set_hover_row(QModelIndex())
    assert updates == [second]
    assert view._hover_rect.isNull()
    view.close()
