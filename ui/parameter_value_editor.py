"""Inline value + soft-range slider editor for parameter rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDoubleSpinBox, QSlider, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from .controller import ControllerWrapper

_SLIDER_STEPS = 1000

# Minimal editor-style slider: thin track, solid gray thumb.
_SLIDER_STYLE = """
QSlider::groove:horizontal {
    height: 2px;
    background: #d0d0d0;
    border: none;
    border-radius: 1px;
    margin: 0 4px;
}
QSlider::sub-page:horizontal,
QSlider::add-page:horizontal {
    background: #d0d0d0;
    border: none;
    border-radius: 1px;
}
QSlider::handle:horizontal {
    width: 10px;
    height: 10px;
    margin: -4px 0;
    border: none;
    border-radius: 5px;
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
    Parameter editor with a spin box and a soft-range slider beneath it.

    While dragging, values are previewed via the controller without undo.
    Committing records a single undo step from the drag-start value.
    """

    editingFinished = Signal()

    def __init__(
        self,
        controller: ControllerWrapper,
        *,
        component_id: str,
        parameter_name: str,
        soft_lo: float,
        soft_hi: float,
        parent: QWidget | None = None,
    ) -> None:
        """
        Initialize the editor.

        Parameters
        ----------
        controller : ControllerWrapper
            Application controller for preview/commit.
        component_id : str
            Component owning the parameter.
        parameter_name : str
            Parameter name.
        soft_lo, soft_hi : float
            Soft slider bounds.
        parent : QWidget or None, optional
            Parent widget.
        """
        super().__init__(parent)
        self._controller = controller
        self._component_id = component_id
        self._parameter_name = parameter_name
        self._soft_lo = float(soft_lo)
        self._soft_hi = float(soft_hi)
        if self._soft_hi <= self._soft_lo:
            self._soft_hi = self._soft_lo + 1.0

        self._start_value: float | None = None
        self._committed = False
        self._updating = False

        self._spin = QDoubleSpinBox(self)
        self._spin.setDecimals(4)
        self._spin.setRange(self._soft_lo, self._soft_hi)
        self._spin.setSingleStep(max((self._soft_hi - self._soft_lo) / 100.0, 1e-4))
        self._spin.setKeyboardTracking(False)
        self._spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self._spin.setFrame(False)
        self._spin.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._spin.setFixedHeight(20)

        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setRange(0, _SLIDER_STEPS)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(max(_SLIDER_STEPS // 20, 1))
        self._slider.setFixedHeight(18)
        self._slider.setStyleSheet(_SLIDER_STYLE)
        self._slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 1, 4, 2)
        layout.setSpacing(2)
        layout.addWidget(self._spin)
        layout.addWidget(self._slider)
        self.setMinimumHeight(44)
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
        self._controller.commit_parameter_preview(
            self._component_id,
            self._parameter_name,
            old_value,
            new_value,
            normalized=False,
        )
        self._committed = True
        self._start_value = new_value

    def cancel_preview(self) -> None:
        """Restore the drag-start value when the editor is dismissed without commit."""
        if self._committed or self._start_value is None:
            return
        self._controller.preview_parameter_value(
            self._component_id,
            self._parameter_name,
            float(self._start_value),
            normalized=False,
        )

    def _value_to_slider(self, value: float) -> int:
        span = self._soft_hi - self._soft_lo
        if span <= 0:
            return 0
        t = (value - self._soft_lo) / span
        return round(min(max(t, 0.0), 1.0) * _SLIDER_STEPS)

    def _slider_to_value(self, pos: int) -> float:
        t = pos / float(_SLIDER_STEPS)
        return self._soft_lo + t * (self._soft_hi - self._soft_lo)

    def _preview(self, value: float) -> None:
        self._controller.preview_parameter_value(
            self._component_id,
            self._parameter_name,
            value,
            normalized=False,
        )

    def _on_slider_pressed(self) -> None:
        # Start a new undo baseline at the value when the drag begins.
        self._start_value = self.value()
        self._committed = False

    def _on_spin_changed(self, value: float) -> None:
        if self._updating:
            return
        if self._start_value is None:
            self._start_value = float(value)
            self._committed = False
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
