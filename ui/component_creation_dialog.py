"""
Dialog for creating peaks/backgrounds with editable parameter configuration.

The dialog is used by multiple UI entry points (plot context menus, properties
context menus) and calls :class:`ui.controller.ControllerWrapper` methods to
create a new component and apply parameter configuration.
"""

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from core.math_models import ModelRegistry, ParameterSpec

from .controller import ControllerWrapper


@dataclass(frozen=True)
class _EditorState:
    """Widget bundle for a single parameter row."""

    value_edit: QLineEdit
    lower_edit: QLineEdit
    upper_edit: QLineEdit
    vary_cb: QCheckBox
    expr_edit: QLineEdit


class ComponentCreationDialog(QDialog):
    """
    Modal dialog for creating a new peak/background component.

    The dialog lets the user pick component type and model, and edit a table
    of parameter fields: ``value``, ``lower``, ``upper``, ``vary``, ``expr``.
    Only ``value`` is applied directly during creation; the other fields are
    applied right after creation via :meth:`ui.controller.ControllerWrapper.update_parameter`.

    Notes
    -----
    - Parameter bounds are interpreted as +/- infinity when fields are left
      blank (or entered as ``inf`` / ``-inf``).
    - If the region already has a background, the ``background`` component
      type is not offered (the dialog won't allow replacement).
    """

    def __init__(
        self,
        controller: ControllerWrapper,
        *,
        region_id: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add component")
        self._controller = controller
        self._region_id = region_id

        self._param_specs: tuple[ParameterSpec, ...] = ()
        self._editors_by_param: dict[str, _EditorState] = {}

        self._setup_widgets()
        self._build_layout()

        self._refresh_component_type_items()
        self._on_component_type_changed(self._component_type_combo.currentText())

    def _setup_widgets(self) -> None:
        """Create widgets and connect signals."""
        self._component_type_combo = QComboBox()
        self._model_combo = QComboBox()

        self._params_table = QTableWidget()
        self._params_table.setAlternatingRowColors(True)
        self._params_table.setColumnCount(6)
        self._params_table.setHorizontalHeaderLabels(["name", "value", "lower", "upper", "vary", "expr"])
        self._params_table.horizontalHeader().setStretchLastSection(True)

        self._type_label = QLabel("Component type")
        self._model_label = QLabel("Model name")
        self._params_label = QLabel("Parameters")

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self._on_create)
        self._buttons.rejected.connect(self.reject)

    def _build_layout(self) -> None:
        """Assemble the dialog layout."""
        layout = QVBoxLayout(self)

        type_row = QHBoxLayout()
        type_row.addWidget(self._type_label)
        type_row.addWidget(self._component_type_combo)
        layout.addLayout(type_row)

        model_row = QHBoxLayout()
        model_row.addWidget(self._model_label)
        model_row.addWidget(self._model_combo)
        layout.addLayout(model_row)

        layout.addWidget(self._params_label)
        layout.addWidget(self._params_table, stretch=1)

        layout.addWidget(self._buttons)

        # Make the model selection responsive even if the table changes focus.
        self._component_type_combo.currentTextChanged.connect(self._on_component_type_changed)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)

    def _refresh_component_type_items(self) -> None:
        """Populate type dropdown based on current region state."""
        self._component_type_combo.blockSignals(True)
        try:
            self._component_type_combo.clear()
            # If region already has a background, disable background replacement.
            background_exists = self._controller.query.get_background_id(self._region_id) is not None
            self._component_type_combo.addItem("peak")
            if not background_exists:
                self._component_type_combo.addItem("background")
            self._component_type_combo.setCurrentIndex(0)
        finally:
            self._component_type_combo.blockSignals(False)

    def _on_component_type_changed(self, component_type: str) -> None:
        """Update the model list when component type changes."""
        self._model_combo.blockSignals(True)
        try:
            self._model_combo.clear()
            if component_type == "peak":
                models = ModelRegistry.get_peak_model_names()
            else:
                models = ModelRegistry.get_background_model_names()

            for name in models:
                self._model_combo.addItem(name)

            self._model_combo.setCurrentIndex(0 if models else -1)
        finally:
            self._model_combo.blockSignals(False)

        if self._model_combo.currentText():
            self._on_model_changed(self._model_combo.currentText())

    def _on_model_changed(self, model_name: str) -> None:
        """Rebuild parameter table for the selected model."""
        model = ModelRegistry.get(model_name)
        self._param_specs = tuple(model.parameter_schema)

        self._editors_by_param.clear()

        self._params_table.setRowCount(len(self._param_specs))
        for row, spec in enumerate(self._param_specs):
            name_item = QTableWidgetItem(spec.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._params_table.setItem(row, 0, name_item)

            value_edit = QLineEdit(str(spec.default))
            lower_edit = QLineEdit("" if math.isinf(spec.lower) else str(spec.lower))
            upper_edit = QLineEdit("" if math.isinf(spec.upper) else str(spec.upper))
            vary_cb = QCheckBox()
            vary_cb.setChecked(spec.vary)
            expr_edit = QLineEdit(spec.expr or "")

            self._params_table.setCellWidget(row, 1, value_edit)
            self._params_table.setCellWidget(row, 2, lower_edit)
            self._params_table.setCellWidget(row, 3, upper_edit)
            self._params_table.setCellWidget(row, 4, vary_cb)
            self._params_table.setCellWidget(row, 5, expr_edit)

            self._editors_by_param[spec.name] = _EditorState(
                value_edit=value_edit,
                lower_edit=lower_edit,
                upper_edit=upper_edit,
                vary_cb=vary_cb,
                expr_edit=expr_edit,
            )

    @staticmethod
    def _float_equal(a: float, b: float) -> bool:
        """Compare floats with +/-inf support."""
        if math.isinf(a) and math.isinf(b):
            return (a > 0) == (b > 0)
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= 1e-9

    @staticmethod
    def _parse_bound(text: str, *, is_lower: bool) -> float:
        """
        Parse a bound value from a text field.

        Blank text means +/-infinity:
        - lower: -inf
        - upper: +inf
        """
        cleaned = text.strip()
        if not cleaned:
            return -np.inf if is_lower else np.inf

        low = cleaned.lower()
        if low in {"-inf", "-infinity"}:
            return -np.inf
        if low in {"inf", "+inf", "infinity", "+infinity"}:
            return np.inf

        return float(cleaned)

    @staticmethod
    def _parse_expr(text: str) -> str | None:
        """Parse dependency expression from an editable text field."""
        cleaned = text.strip()
        return cleaned if cleaned else None

    def _on_create(self) -> None:
        """Create component and apply parameter configuration."""
        component_type = str(self._component_type_combo.currentText()).strip()
        model_name = str(self._model_combo.currentText()).strip()
        if component_type not in {"peak", "background"}:
            raise ValueError("Invalid component type selection.")
        if not model_name:
            raise ValueError("Please select a model.")

        value_by_param: dict[str, float] = {}
        config_by_param: dict[str, tuple[float, float, bool, str | None]] = {}

        for spec in self._param_specs:
            editors = self._editors_by_param[spec.name]
            value_text = editors.value_edit.text().strip()
            if not value_text:
                raise ValueError(f"Missing value for parameter '{spec.name}'.")
            value_by_param[spec.name] = float(value_text)

            lower = self._parse_bound(editors.lower_edit.text(), is_lower=True)
            upper = self._parse_bound(editors.upper_edit.text(), is_lower=False)
            vary = editors.vary_cb.isChecked()
            expr = self._parse_expr(editors.expr_edit.text())
            config_by_param[spec.name] = (lower, upper, vary, expr)

        # Capture component identifiers before creation so we can find the created component.
        before_peak_ids = set(self._controller.query.get_peaks_ids(self._region_id))
        before_bg_id = self._controller.query.get_background_id(self._region_id)

        if component_type == "peak":
            self._controller.create_peak(self._region_id, model_name, parameters=value_by_param)
            after_peak_ids = set(self._controller.query.get_peaks_ids(self._region_id))
            new_ids = list(after_peak_ids - before_peak_ids)
            if len(new_ids) != 1:
                raise RuntimeError(
                    "Failed to identify created peak id (unexpected number of new peaks)."
                )
            component_id = new_ids[0]
        else:
            # Background replacement is disabled in this dialog when a background exists.
            if before_bg_id is not None:
                raise RuntimeError("Background already exists; replacement is disabled.")
            self._controller.create_background(self._region_id, model_name, parameters=value_by_param)
            component_id = str(self._controller.query.get_background_id(self._region_id))
            if not component_id:
                raise RuntimeError("Failed to identify created background id.")

        # Apply lower/upper/vary/expr only when the value differs from model defaults.
        for spec in self._param_specs:
            lower, upper, vary, expr = config_by_param[spec.name]

            if not self._float_equal(lower, spec.lower):
                self._controller.update_parameter(
                    component_id, spec.name, "lower", lower, normalized=False
                )
            if not self._float_equal(upper, spec.upper):
                self._controller.update_parameter(
                    component_id, spec.name, "upper", upper, normalized=False
                )
            if vary != spec.vary:
                self._controller.update_parameter(
                    component_id, spec.name, "vary", vary, normalized=False
                )
            if (expr or None) != spec.expr:
                self._controller.update_parameter(
                    component_id, spec.name, "expr", expr, normalized=False
                )

        self.accept()
