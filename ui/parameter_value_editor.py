"""Inline value + soft-range slider editor for parameter and region rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDoubleSpinBox, QSlider, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from .controller import ControllerWrapper

_SLIDER_STEPS = 1000

# Minimal editor-style slider: thin track, solid gray thumb (fully inside cell).
_SLIDER_STYLE = """
QSlider::groove:horizontal {
    height: 2px;
    background: #d0d0d0;
    border: none;
    border-radius: 1px;
    margin: 0 6px;
}
QSlider::sub-page:horizontal,
QSlider::add-page:horizontal {
    background: #d0d0d0;
    border: none;
    border-radius: 1px;
}
QSlider::handle:horizontal {
    width: 8px;
    height: 8px;
    margin: -3px 0;
    border: none;
    border-radius: 4px;
    background: #6e6e6e;
}
QSlider::handle:horizontal:hover {
    background: #555555;
}
QSlider::handle:horizontal:pressed {
    background: #444444;
}
"""


class ParameterValueEditor(QWidget):
    """
    Value editor with a spin box and a soft-range slider beneath it.

    Supports component parameters and region start/stop bounds. While dragging,
    values are previewed via the controller without undo. Committing records a
    single undo step from the drag-start value.
    """

    editingFinished = Signal()

    def __init__(
        self,
        controller: ControllerWrapper,
        *,
        soft_lo: float,
        soft_hi: float,
        component_id: str | None = None,
        parameter_name: str | None = None,
        region_id: str | None = None,
        slice_bound: Literal["start", "stop"] | None = None,
        slice_mode: Literal["value", "index"] = "value",
        parent: QWidget | None = None,
    ) -> None:
        """
        Initialize the editor.

        Parameters
        ----------
        controller : ControllerWrapper
            Application controller for preview/commit.
        soft_lo, soft_hi : float
            Soft slider bounds.
        component_id, parameter_name : str or None, optional
            Component parameter binding (mutually exclusive with region mode).
        region_id : str or None, optional
            Region binding for start/stop editing.
        slice_bound : {"start", "stop"} or None, optional
            Which region bound this editor controls.
        slice_mode : {"value", "index"}, optional
            Region slice display mode.
        parent : QWidget or None, optional
            Parent widget.
        """
        super().__init__(parent)
        self._controller = controller
        self._component_id = component_id
        self._parameter_name = parameter_name
        self._region_id = region_id
        self._slice_bound = slice_bound
        self._slice_mode: Literal["value", "index"] = slice_mode
        self._companion_value: float | None = None
        self._old_start_index: int | None = None
        self._old_stop_index: int | None = None

        self._soft_lo = float(soft_lo)
        self._soft_hi = float(soft_hi)
        if self._soft_hi <= self._soft_lo:
            self._soft_hi = self._soft_lo + 1.0

        self._start_value: float | None = None
        self._committed = False
        self._updating = False

        decimals = 0 if self._slice_mode == "index" and self._region_id is not None else 4
        self._spin = QDoubleSpinBox(self)
        self._spin.setDecimals(decimals)
        self._spin.setRange(self._soft_lo, self._soft_hi)
        step = 1.0 if decimals == 0 else max((self._soft_hi - self._soft_lo) / 100.0, 1e-4)
        self._spin.setSingleStep(step)
        self._spin.setKeyboardTracking(False)
        self._spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self._spin.setFrame(False)
        self._spin.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._spin.setFixedHeight(20)

        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setRange(0, _SLIDER_STEPS)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(max(_SLIDER_STEPS // 20, 1))
        self._slider.setFixedHeight(14)
        self._slider.setStyleSheet(_SLIDER_STYLE)
        self._slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 4, 4)
        layout.setSpacing(1)
        layout.addWidget(self._spin)
        layout.addWidget(self._slider)
        self.setMinimumHeight(42)
        self.setMinimumWidth(90)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("ParameterValueEditor { background: transparent; }")
        self._spin.setStyleSheet("QDoubleSpinBox { background: transparent; border: none; }")

        self._spin.valueChanged.connect(self._on_spin_changed)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._spin.editingFinished.connect(self._on_spin_editing_finished)

    @property
    def committed(self) -> bool:
        """True if the editor already recorded an undoable commit."""
        return self._committed

    def set_value(self, value: float) -> None:
        """Initialize the displayed value and remember it as the drag-start baseline."""
        self._start_value = float(value)
        self._committed = False
        self._capture_region_baseline()
        self._updating = True
        try:
            clamped = min(max(float(value), self._soft_lo), self._soft_hi)
            self._spin.setValue(clamped)
            self._slider.setValue(self._value_to_slider(clamped))
        finally:
            self._updating = False

    def value(self) -> float:
        """Return the current spin-box value."""
        return float(self._spin.value())

    def commit_if_needed(self) -> None:
        """Record undo from start→current when the preview changed the value."""
        if self._committed or self._start_value is None:
            return
        new_value = self.value()
        old_value = float(self._start_value)
        if new_value == old_value:
            self._committed = True
            return
        try:
            if self._region_id is not None and self._slice_bound is not None:
                if self._old_start_index is None or self._old_stop_index is None:
                    return
                self._controller.commit_region_slice_preview(
                    self._region_id,
                    self._old_start_index,
                    self._old_stop_index,
                )
                self._capture_region_baseline()
            elif self._component_id is not None and self._parameter_name is not None:
                self._controller.commit_parameter_preview(
                    self._component_id,
                    self._parameter_name,
                    old_value,
                    new_value,
                    normalized=False,
                )
            else:
                return
        except KeyError:
            # Target was removed while the editor was still open.
            self._committed = True
            self._start_value = None
            return
        self._committed = True
        self._start_value = new_value

    def cancel_preview(self) -> None:
        """Restore the drag-start value when the editor is dismissed without commit."""
        if self._committed or self._start_value is None:
            return
        try:
            if self._region_id is not None and self._slice_bound is not None:
                if self._old_start_index is None or self._old_stop_index is None:
                    return
                self._controller.preview_region_slice(
                    self._region_id,
                    self._old_start_index,
                    self._old_stop_index,
                    mode="index",
                )
                return
            if self._component_id is not None and self._parameter_name is not None:
                self._controller.preview_parameter_value(
                    self._component_id,
                    self._parameter_name,
                    float(self._start_value),
                    normalized=False,
                )
        except KeyError:
            pass
        finally:
            self._committed = True
            self._start_value = None

    def abandon(self) -> None:
        """Drop pending preview state without touching the document."""
        self._committed = True
        self._start_value = None
        self._old_start_index = None
        self._old_stop_index = None
        self._companion_value = None

    def _capture_region_baseline(self) -> None:
        if self._region_id is None or self._slice_bound is None:
            return
        start_idx, stop_idx = self._controller.query.get_region_slice(self._region_id, mode="index")
        self._old_start_index = int(start_idx) if start_idx is not None else None
        self._old_stop_index = int(stop_idx) if stop_idx is not None else None
        start, stop = self._controller.query.get_region_slice(
            self._region_id, mode=self._slice_mode
        )
        if self._slice_bound == "start":
            self._companion_value = float(stop) if stop is not None else None
        else:
            self._companion_value = float(start) if start is not None else None

    def _bounds_for(self, value: float) -> tuple[float, float]:
        companion = float(self._companion_value) if self._companion_value is not None else value
        if self._slice_bound == "start":
            return (float(value), companion)
        return (companion, float(value))

    def _value_to_slider(self, value: float) -> int:
        span = self._soft_hi - self._soft_lo
        if span <= 0:
            return 0
        t = (value - self._soft_lo) / span
        return round(min(max(t, 0.0), 1.0) * _SLIDER_STEPS)

    def _slider_to_value(self, pos: int) -> float:
        t = pos / float(_SLIDER_STEPS)
        value = self._soft_lo + t * (self._soft_hi - self._soft_lo)
        if self._slice_mode == "index" and self._region_id is not None:
            return float(round(value))
        return value

    def _preview(self, value: float) -> None:
        if self._region_id is not None and self._slice_bound is not None:
            start, stop = self._bounds_for(value)
            self._controller.preview_region_slice(
                self._region_id,
                start,
                stop,
                mode=self._slice_mode,
            )
            return
        if self._component_id is not None and self._parameter_name is not None:
            self._controller.preview_parameter_value(
                self._component_id,
                self._parameter_name,
                value,
                normalized=False,
            )

    def _on_slider_pressed(self) -> None:
        self._start_value = self.value()
        self._committed = False
        self._capture_region_baseline()

    def _on_spin_changed(self, value: float) -> None:
        if self._updating:
            return
        if self._start_value is None:
            self._start_value = float(value)
            self._committed = False
            self._capture_region_baseline()
        self._updating = True
        try:
            self._slider.setValue(self._value_to_slider(float(value)))
        finally:
            self._updating = False
        self._preview(float(value))

    def _on_slider_changed(self, pos: int) -> None:
        if self._updating:
            return
        value = self._slider_to_value(int(pos))
        self._updating = True
        try:
            self._spin.setValue(value)
        finally:
            self._updating = False
        self._preview(value)

    def _on_slider_released(self) -> None:
        self.commit_if_needed()
        self.editingFinished.emit()

    def _on_spin_editing_finished(self) -> None:
        self.commit_if_needed()
        self.editingFinished.emit()
