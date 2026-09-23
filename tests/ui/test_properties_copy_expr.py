"""UI tests for properties tree copy vs expression picker interaction."""

from __future__ import annotations

import sys
from typing import cast

import pytest
from PySide6.QtCore import QEvent, QModelIndex, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem

from ui.controller import ControllerWrapper
from ui.properties import (
    ItemKind,
    PropertiesView,
    PropertyItem,
    _action_icon_rect,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return cast(QApplication, app)


def _find_expr_value_index(
    view: PropertiesView,
    *,
    component_id: str,
    parameter_name: str,
) -> QModelIndex:
    """Return the value-column index of a specific parameter ``expr`` field."""
    model = view.model()

    def walk(parent: QModelIndex) -> QModelIndex:
        for row in range(model.rowCount(parent)):
            name_index = model.index(row, 0, parent)
            item = name_index.internalPointer()
            if (
                isinstance(item, PropertyItem)
                and item.kind == ItemKind.PARAMETER_FIELD
                and item.parameter_field == "expr"
                and item.component_id == component_id
                and item.parameter_name == parameter_name
            ):
                return model.index(row, 1, parent)
            child = walk(name_index)
            if child.isValid():
                return child
        return QModelIndex()

    return walk(QModelIndex())


def test_copy_expr_does_not_open_expr_popup(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
    region_id: str,
    peak_id: str,
) -> None:
    """Clicking the copy glyph on an expr row copies text without opening the popup."""
    del qapp
    controller = ControllerWrapper(collection=simple_collection)
    controller.update_parameter(peak_id, "amp", "expr", "2*pOther")
    controller.set_selection(spectrum_id, region_id, peak_id)

    view = PropertiesView(controller)
    view.resize(480, 720)
    view.show()
    view.refresh()
    QApplication.processEvents()
    view.expandAll()
    QApplication.processEvents()

    expr_index = _find_expr_value_index(view, component_id=peak_id, parameter_name="amp")
    assert expr_index.isValid()

    opened: list[QModelIndex] = []
    view._value_delegate.show_expr_popup = (  # ty: ignore[invalid-assignment]
        lambda index: opened.append(index)
    )

    cell = view.visualRect(expr_index)
    option = QStyleOptionViewItem()
    option.rect = cell
    copy_pos = _action_icon_rect(option).center()

    # Drive the view handlers directly: viewport forwarding can drop glyph hits
    # under offscreen Qt, while the press/release pairing is what regressed.
    local = QPointF(copy_pos)
    global_pos = QPointF(view.viewport().mapToGlobal(copy_pos))
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        local,
        global_pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        local,
        global_pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.mousePressEvent(press)
    view.mouseReleaseEvent(release)
    QApplication.processEvents()

    assert QApplication.clipboard().text() == "2*pOther"
    assert opened == []
    view.close()
