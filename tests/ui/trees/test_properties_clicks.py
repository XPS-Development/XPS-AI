"""Click tests for the properties tree: parameters, glyphs, and add rows."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast

import pytest
from PySide6.QtCore import QEvent, QModelIndex, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionViewItem

from ui.controller import ControllerWrapper
from ui.trees.parameter_value_editor import ParameterValueEditor
from ui.trees.properties import (
    ItemKind,
    PropertiesView,
    PropertyItem,
    _action_icon_rect,
    _action_icon_rect_at,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return cast(QApplication, app)


def _walk_name_indexes(view: PropertiesView) -> list[QModelIndex]:
    """Return every name-column index in the properties tree."""
    found: list[QModelIndex] = []
    model = view.model()

    def walk(parent: QModelIndex) -> None:
        for row in range(model.rowCount(parent)):
            name_index = model.index(row, 0, parent)
            found.append(name_index)
            walk(name_index)

    walk(QModelIndex())
    return found


def _find_name_indexes(
    view: PropertiesView,
    predicate: Callable[[PropertyItem], bool],
) -> list[QModelIndex]:
    """Return name-column indexes whose items match ``predicate``."""
    matched: list[QModelIndex] = []
    for name_index in _walk_name_indexes(view):
        item = name_index.internalPointer()
        if isinstance(item, PropertyItem) and predicate(item):
            matched.append(name_index)
    return matched


def _open_view(
    controller: ControllerWrapper,
    *,
    width: int = 480,
    height: int = 900,
) -> PropertiesView:
    """Show a properties view wired to the same refresh signal as the main window."""
    view = PropertiesView(controller)
    controller.propertiesNeedsRefresh.connect(view.refresh)
    view.resize(width, height)
    view.show()
    QApplication.processEvents()
    return view


def _click_at(view: PropertiesView, pos: QPoint, *, release: bool = True) -> None:
    """Press, and optionally release, the left button at a viewport position."""
    local = QPointF(pos)
    global_pos = QPointF(view.viewport().mapToGlobal(pos))
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        local,
        global_pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.mousePressEvent(press)
    if release:
        release_event = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            local,
            global_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        view.mouseReleaseEvent(release_event)
    QApplication.processEvents()


def _click(view: PropertiesView, index: QModelIndex) -> None:
    """Click the center of a row, expanding ancestors so the row is on screen."""
    parent = index.parent()
    while parent.isValid():
        view.expand(parent)
        parent = parent.parent()
    view.scrollTo(index)
    QApplication.processEvents()
    rect = view.visualRect(index)
    assert rect.isValid()
    _click_at(view, rect.center())


def _value_index(view: PropertiesView, name_index: QModelIndex) -> QModelIndex:
    """Return the value-column index on the same row."""
    return view.model().index(name_index.row(), 1, name_index.parent())


def _click_name_glyph(view: PropertiesView, index: QModelIndex, slot_from_right: int) -> None:
    """Click a trailing name-column action glyph (0 = delete, 1 = optimize)."""
    view.scrollTo(index)
    QApplication.processEvents()
    cell = view.visualRect(index)
    option = QStyleOptionViewItem()
    option.rect = cell
    _click_at(view, _action_icon_rect_at(option, slot_from_right).center(), release=False)


def test_second_parameter_click_opens_one_menu(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
    region_id: str,
    peak_id: str,
) -> None:
    """First click shows the slider; the next click opens only that parameter's menu."""
    del qapp
    controller = ControllerWrapper(collection=simple_collection)
    controller.set_selection(spectrum_id, region_id, peak_id)
    view = _open_view(controller, height=720)

    params = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.PARAMETER_ROW and item.component_id == peak_id,
    )
    assert len(params) >= 2
    first, second = params[0], params[1]
    opened: list[QModelIndex] = []
    view._value_delegate.show_expr_popup = lambda index: opened.append(index)  # ty: ignore[invalid-assignment]

    _click(view, first)
    value = view.model().index(first.row(), 1, first.parent())
    assert isinstance(view.indexWidget(value), ParameterValueEditor)
    assert view.isExpanded(first) is False
    assert opened == []

    _click(view, first)
    assert view.isExpanded(first) is True
    assert opened == []

    _click(view, second)
    second_value = view.model().index(second.row(), 1, second.parent())
    assert view.isExpanded(first) is False
    assert view.isExpanded(second) is False
    assert isinstance(view.indexWidget(second_value), ParameterValueEditor)
    assert view.indexWidget(value) is None
    assert opened == []
    view.close()


def test_other_component_parameter_click_opens_its_slider(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
    region_id: str,
    peak_id: str,
    background_id: str,
) -> None:
    """A press on another component's parameter selects it and opens only its slider."""
    del qapp
    controller = ControllerWrapper(collection=simple_collection)
    controller.set_selection(spectrum_id, region_id, peak_id)
    view = _open_view(controller)

    peak_params = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.PARAMETER_ROW and item.component_id == peak_id,
    )
    bg_params = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.PARAMETER_ROW and item.component_id == background_id,
    )
    assert peak_params
    assert bg_params
    _click(view, peak_params[0])
    _click(view, peak_params[0])
    assert view.isExpanded(peak_params[0]) is True

    _click(view, bg_params[0])
    bg_value = view.model().index(bg_params[0].row(), 1, bg_params[0].parent())
    assert controller.selected_component_id == background_id
    assert isinstance(view.indexWidget(bg_value), ParameterValueEditor)
    assert view.isExpanded(peak_params[0]) is False
    assert view.isExpanded(bg_params[0]) is False
    view.close()


def test_switch_away_from_open_region_or_peak_editor(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
    region_id: str,
    peak_id: str,
    background_id: str,
) -> None:
    """An open region or peak slider/menu yields immediately to the next value row."""
    del qapp
    controller = ControllerWrapper(collection=simple_collection)
    controller.create_region(spectrum_id)
    other_region_id = next(
        rid for rid in controller.query.get_regions_ids(spectrum_id) if rid != region_id
    )
    controller.set_selection(spectrum_id, region_id, peak_id)
    view = _open_view(controller)

    region_slice = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.REGION_SLICE and item.region_id == region_id,
    )[0]
    other_slice = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.REGION_SLICE and item.region_id == other_region_id,
    )[0]
    peak_param = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.PARAMETER_ROW and item.component_id == peak_id,
    )[0]
    bg_param = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.PARAMETER_ROW and item.component_id == background_id,
    )[0]

    _click(view, region_slice)
    assert controller.selected_region_id == region_id
    assert controller.selected_component_id is None
    assert isinstance(view.indexWidget(_value_index(view, region_slice)), ParameterValueEditor)

    _click(view, peak_param)
    assert controller.selected_component_id == peak_id
    assert isinstance(view.indexWidget(_value_index(view, peak_param)), ParameterValueEditor)
    assert view.indexWidget(_value_index(view, region_slice)) is None
    assert view.isExpanded(peak_param) is False

    _click(view, peak_param)
    assert view.isExpanded(peak_param) is True

    _click(view, other_slice)
    assert controller.selected_region_id == other_region_id
    assert controller.selected_component_id is None
    assert view.isExpanded(peak_param) is False
    assert view.indexWidget(_value_index(view, peak_param)) is None
    assert isinstance(view.indexWidget(_value_index(view, other_slice)), ParameterValueEditor)

    bg_param = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.PARAMETER_ROW and item.component_id == background_id,
    )[0]
    _click(view, bg_param)
    assert controller.selected_region_id == region_id
    assert controller.selected_component_id == background_id
    assert view.isExpanded(peak_param) is False
    assert view.indexWidget(_value_index(view, other_slice)) is None
    assert isinstance(view.indexWidget(_value_index(view, bg_param)), ParameterValueEditor)
    assert view.isExpanded(bg_param) is False
    view.close()


def test_other_object_parameter_opens_from_selected_region_peak_or_background(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
    region_id: str,
    peak_id: str,
    background_id: str,
) -> None:
    """A selected region, peak, or background yields its slider to another object's parameter."""
    del qapp
    controller = ControllerWrapper(collection=simple_collection)
    controller.set_selection(spectrum_id, region_id, None)
    view = _open_view(controller)

    def parameter(component_id: str) -> QModelIndex:
        rows = _find_name_indexes(
            view,
            lambda item: item.kind == ItemKind.PARAMETER_ROW and item.component_id == component_id,
        )
        assert rows
        return rows[0]

    def component_row(component_id: str) -> QModelIndex:
        rows = _find_name_indexes(
            view,
            lambda item: item.kind == ItemKind.COMPONENT and item.component_id == component_id,
        )
        assert rows
        return rows[0]

    peak_param = parameter(peak_id)
    _click(view, peak_param)
    assert controller.selected_region_id == region_id
    assert controller.selected_component_id == peak_id
    assert isinstance(view.indexWidget(_value_index(view, peak_param)), ParameterValueEditor)
    assert view.isExpanded(peak_param) is False

    _click(view, component_row(peak_id))
    assert controller.selected_component_id == peak_id
    assert view.indexWidget(_value_index(view, peak_param)) is None

    bg_param = parameter(background_id)
    _click(view, bg_param)
    assert controller.selected_region_id == region_id
    assert controller.selected_component_id == background_id
    assert isinstance(view.indexWidget(_value_index(view, bg_param)), ParameterValueEditor)
    assert view.indexWidget(_value_index(view, peak_param)) is None
    assert view.isExpanded(bg_param) is False

    _click(view, component_row(background_id))
    assert controller.selected_component_id == background_id
    assert view.indexWidget(_value_index(view, bg_param)) is None

    peak_param = parameter(peak_id)
    _click(view, peak_param)
    assert controller.selected_region_id == region_id
    assert controller.selected_component_id == peak_id
    assert isinstance(view.indexWidget(_value_index(view, peak_param)), ParameterValueEditor)
    assert view.isExpanded(peak_param) is False
    view.close()


def test_vary_checkbox_selects_its_row_and_keeps_the_menu(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
    region_id: str,
    peak_id: str,
) -> None:
    """Clicking vary selects that field and leaves the parameter menu open after refresh."""
    del qapp
    controller = ControllerWrapper(collection=simple_collection)
    controller.set_selection(spectrum_id, region_id, peak_id)
    view = _open_view(controller)

    def vary_field() -> QModelIndex:
        rows = _find_name_indexes(
            view,
            lambda item: (
                item.kind == ItemKind.PARAMETER_FIELD
                and item.parameter_field == "vary"
                and item.component_id == peak_id
            ),
        )
        assert rows
        return rows[0]

    param = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.PARAMETER_ROW and item.component_id == peak_id,
    )[0]
    _click(view, param)
    _click(view, param)
    assert view.isExpanded(param) is True

    vary_name = vary_field()
    vary_item = vary_name.internalPointer()
    assert isinstance(vary_item, PropertyItem)
    parameter_name = vary_item.parameter_name
    vary_value = _value_index(view, vary_name)
    was_checked = bool(vary_item.value)
    view.scrollTo(vary_value)
    QApplication.processEvents()
    option = QStyleOptionViewItem()
    option.rect = view.visualRect(vary_value)
    option.features = QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
    check = view.style().subElementRect(
        QStyle.SubElement.SE_ItemViewItemCheckIndicator,
        option,
        view.viewport(),
    )
    _click_at(view, check.center() if check.isValid() else option.rect.center())

    param = _find_name_indexes(
        view,
        lambda item: (
            item.kind == ItemKind.PARAMETER_ROW
            and item.component_id == peak_id
            and item.parameter_name == parameter_name
        ),
    )[0]
    vary_name = vary_field()
    current = view.selectionModel().currentIndex().internalPointer()
    assert view.isExpanded(param) is True
    assert isinstance(current, PropertyItem)
    assert current.kind == ItemKind.PARAMETER_FIELD
    assert current.parameter_field == "vary"
    assert current.parameter_name == vary_name.internalPointer().parameter_name
    assert bool(vary_name.internalPointer().value) is not was_checked
    view.close()


def test_tree_rebuild_keeps_scroll_position(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
    region_id: str,
    peak_id: str,
) -> None:
    """Expr edits and other tree rebuilds leave the properties scrollbar where it was."""
    del qapp
    controller = ControllerWrapper(collection=simple_collection)
    controller.set_selection(spectrum_id, region_id, peak_id)
    view = _open_view(controller, height=160)
    bar = view.verticalScrollBar()
    bar.setValue(bar.maximum())
    QApplication.processEvents()
    scrolled = bar.value()
    assert scrolled > 0

    controller.update_parameter(peak_id, "amp", "expr", "1")
    QApplication.processEvents()
    QApplication.processEvents()
    assert bar.value() == scrolled

    view.refresh()
    QApplication.processEvents()
    QApplication.processEvents()
    assert bar.value() == scrolled
    view.close()


def test_queued_selection_echo_keeps_other_component_parameter_slider(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
    region_id: str,
    peak_id: str,
    background_id: str,
) -> None:
    """The app's queued selection sync must not replace a just-opened parameter slider."""
    del qapp
    controller = ControllerWrapper(collection=simple_collection)
    controller.set_selection(spectrum_id, region_id, peak_id)
    view = _open_view(controller)
    controller.selectionChanged.connect(
        view.on_controller_selection_changed,
        Qt.ConnectionType.QueuedConnection,
    )

    bg_param = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.PARAMETER_ROW and item.component_id == background_id,
    )[0]
    _click(view, bg_param)

    current = view.selectionModel().currentIndex().internalPointer()
    assert isinstance(current, PropertyItem)
    assert current.kind == ItemKind.PARAMETER_ROW
    assert current.component_id == background_id
    assert isinstance(view.indexWidget(_value_index(view, bg_param)), ParameterValueEditor)

    controller.set_selection(spectrum_id, region_id, peak_id)
    QApplication.processEvents()
    current = view.selectionModel().currentIndex().internalPointer()
    assert isinstance(current, PropertyItem)
    assert current.kind == ItemKind.COMPONENT
    assert current.component_id == peak_id
    assert view.indexWidget(_value_index(view, bg_param)) is None
    view.close()


def test_copy_click_does_not_change_parameter_state(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
    region_id: str,
    peak_id: str,
    background_id: str,
) -> None:
    """Copying another parameter leaves the open slider and the selection in place."""
    del qapp
    controller = ControllerWrapper(collection=simple_collection)
    controller.set_selection(spectrum_id, region_id, peak_id)
    view = _open_view(controller)

    peak_param = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.PARAMETER_ROW and item.component_id == peak_id,
    )[0]
    bg_param = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.PARAMETER_ROW and item.component_id == background_id,
    )[0]
    _click(view, peak_param)
    peak_value = view.model().index(peak_param.row(), 1, peak_param.parent())
    bg_value = view.model().index(bg_param.row(), 1, bg_param.parent())
    view.scrollTo(bg_value)
    QApplication.processEvents()
    cell = view.visualRect(bg_value)
    option = QStyleOptionViewItem()
    option.rect = cell
    _click_at(view, _action_icon_rect(option).center(), release=False)

    assert controller.selected_component_id == peak_id
    assert isinstance(view.indexWidget(peak_value), ParameterValueEditor)
    assert view.indexWidget(bg_value) is None
    assert QApplication.clipboard().text()
    view.close()


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
    view = _open_view(controller, height=720)
    view.expandAll()
    QApplication.processEvents()

    expr_rows = _find_name_indexes(
        view,
        lambda item: (
            item.kind == ItemKind.PARAMETER_FIELD
            and item.parameter_field == "expr"
            and item.component_id == peak_id
            and item.parameter_name == "amp"
        ),
    )
    assert expr_rows
    expr_index = view.model().index(expr_rows[0].row(), 1, expr_rows[0].parent())
    opened: list[QModelIndex] = []
    view._value_delegate.show_expr_popup = lambda index: opened.append(index)  # ty: ignore[invalid-assignment]

    view.scrollTo(expr_index)
    QApplication.processEvents()
    option = QStyleOptionViewItem()
    option.rect = view.visualRect(expr_index)
    _click_at(view, _action_icon_rect(option).center())

    assert QApplication.clipboard().text() == "2*pOther"
    assert opened == []
    view.close()


def test_empty_click_clears_selection_and_open_fields(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
    region_id: str,
    peak_id: str,
) -> None:
    """A click on empty space closes parameter fields and clears region and component."""
    del qapp
    controller = ControllerWrapper(collection=simple_collection)
    controller.set_selection(spectrum_id, region_id, peak_id)
    view = _open_view(controller, height=720)

    param = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.PARAMETER_ROW and item.component_id == peak_id,
    )[0]
    _click(view, param)
    _click(view, param)
    assert view.isExpanded(param) is True

    _click_at(
        view,
        QPoint(view.viewport().width() - 4, view.viewport().height() - 4),
        release=False,
    )

    assert controller.selected_region_id is None
    assert controller.selected_component_id is None
    assert controller.selected_spectrum_id == spectrum_id
    assert view.isExpanded(param) is False
    assert view.indexWidget(view.model().index(param.row(), 1, param.parent())) is None
    view.close()


def test_delete_optimize_and_add_rows_respond_to_clicks(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    simple_collection,
    spectrum_id: str,
    region_id: str,
    peak_id: str,
) -> None:
    """Delete, optimize, and add-row clicks run their actions on the first press."""
    del qapp
    optimized: list[list[str] | None] = []

    def _record_optimize(
        _parent: object,
        _controller: ControllerWrapper,
        *,
        region_ids: list[str] | None = None,
        spectrum_ids: list[str] | None = None,
        **_kwargs: object,
    ) -> bool:
        del spectrum_ids
        optimized.append(list(region_ids) if region_ids is not None else None)
        return False

    monkeypatch.setattr("ui.trees.properties.confirm_and_optimize", _record_optimize)

    controller = ControllerWrapper(collection=simple_collection)
    controller.set_selection(spectrum_id, region_id, peak_id)
    view = _open_view(controller)
    before_regions = controller.query.get_regions_ids(spectrum_id)

    add_region = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.ACTION_ROW and item.action == "add_region",
    )
    assert add_region
    _click(view, add_region[0])
    regions = controller.query.get_regions_ids(spectrum_id)
    assert len(regions) == len(before_regions) + 1
    new_region_id = next(rid for rid in regions if rid not in before_regions)

    add_peak = _find_name_indexes(
        view,
        lambda item: (
            item.kind == ItemKind.ACTION_ROW
            and item.action == "add_peak"
            and item.region_id == new_region_id
        ),
    )
    add_background = _find_name_indexes(
        view,
        lambda item: (
            item.kind == ItemKind.ACTION_ROW
            and item.action == "add_background"
            and item.region_id == new_region_id
        ),
    )
    assert add_peak
    assert add_background
    _click(view, add_background[0])
    add_peak = _find_name_indexes(
        view,
        lambda item: (
            item.kind == ItemKind.ACTION_ROW
            and item.action == "add_peak"
            and item.region_id == new_region_id
        ),
    )
    assert add_peak
    _click(view, add_peak[0])
    components = controller.query.get_components_ids(new_region_id)
    assert len(components) == 2

    region_row = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.REGION and item.region_id == region_id,
    )[0]
    _click_name_glyph(view, region_row, 1)
    assert optimized == [[region_id]]

    peak_row = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.COMPONENT and item.component_id == peak_id,
    )[0]
    _click_name_glyph(view, peak_row, 0)
    assert controller.query.check_object_exists(peak_id) is False

    region_row = _find_name_indexes(
        view,
        lambda item: item.kind == ItemKind.REGION and item.region_id == region_id,
    )[0]
    _click_name_glyph(view, region_row, 0)
    assert controller.query.check_object_exists(region_id) is False
    view.close()
