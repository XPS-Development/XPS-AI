from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.orchestration import AppParameters


class OptionsDialog(QDialog):
    """
    Dialog for viewing and editing AppParameters.

    Parameters are grouped into Core, Import, NN, Optimization, and
    Serialization sections. Changes are applied to an existing
    AppParameters instance via apply_to_params.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Application settings")

        self._automatic_methods_cb = QCheckBox()
        self._default_bg_model_edit = QLineEdit()
        self._show_spectrum_id_in_tree_cb = QCheckBox()

        self._import_use_be_cb = QCheckBox()
        self._import_use_cps_cb = QCheckBox()

        self._nn_model_path_edit = QLineEdit()
        self._nn_pred_threshold_sb = QDoubleSpinBox()
        self._nn_smooth_cb = QCheckBox()
        self._nn_interp_num_sb = QSpinBox()

        self._optimization_kwargs_edit = QPlainTextEdit()

        self._serialization_mode_edit = QLineEdit()
        self._serialization_path_edit = QLineEdit()
        self._serialization_indent_sb = QSpinBox()

        self._setup_widgets()
        self._build_layout()

    def _setup_widgets(self) -> None:
        """Configure widgets with sensible ranges and defaults."""
        self._nn_pred_threshold_sb.setRange(0.0, 1.0)
        self._nn_pred_threshold_sb.setSingleStep(0.05)
        self._nn_pred_threshold_sb.setDecimals(3)

        self._nn_interp_num_sb.setRange(1, 10_000)

        self._serialization_indent_sb.setRange(0, 16)
        self._serialization_indent_sb.setSpecialValueText("None")

    def _build_layout(self) -> None:
        """Create the main dialog layout."""
        main_layout = QVBoxLayout(self)

        core_group = QGroupBox("Core")
        core_layout = QFormLayout(core_group)
        core_layout.addRow("Automatic methods", self._automatic_methods_cb)
        core_layout.addRow("Default background model", self._default_bg_model_edit)
        core_layout.addRow("Show spectrum ID in tree", self._show_spectrum_id_in_tree_cb)

        import_group = QGroupBox("Import")
        import_layout = QFormLayout(import_group)
        import_layout.addRow("Use binding energy", self._import_use_be_cb)
        import_layout.addRow("Use CPS", self._import_use_cps_cb)

        nn_group = QGroupBox("NN")
        nn_layout = QFormLayout(nn_group)
        nn_layout.addRow("Model path", self._nn_model_path_edit)
        nn_layout.addRow("Prediction threshold", self._nn_pred_threshold_sb)
        nn_layout.addRow("Smooth mask", self._nn_smooth_cb)
        nn_layout.addRow("Interpolation points", self._nn_interp_num_sb)

        opt_group = QGroupBox("Optimization")
        opt_layout = QFormLayout(opt_group)
        opt_layout.addRow("Optimization kwargs (JSON)", self._optimization_kwargs_edit)

        ser_group = QGroupBox("Serialization")
        ser_layout = QFormLayout(ser_group)
        ser_layout.addRow("Default mode", self._serialization_mode_edit)
        ser_layout.addRow("Default path", self._serialization_path_edit)
        ser_layout.addRow("Indent (0 = None)", self._serialization_indent_sb)

        groups_grid = QGridLayout()
        groups_grid.addWidget(core_group, 0, 0)
        groups_grid.addWidget(import_group, 0, 1)
        groups_grid.addWidget(nn_group, 1, 0)
        groups_grid.addWidget(opt_group, 1, 1)
        groups_grid.addWidget(ser_group, 2, 0, 1, 2)

        main_layout.addLayout(groups_grid)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(buttons)

        main_layout.addLayout(buttons_layout)

    def load_from_params(self, params: AppParameters) -> None:
        """
        Populate widgets from an AppParameters instance.

        Parameters
        ----------
        params : AppParameters
            Source parameters.
        """
        self._automatic_methods_cb.setChecked(params.automatic_methods)
        self._default_bg_model_edit.setText(params.default_background_model_for_auto_methods)
        self._show_spectrum_id_in_tree_cb.setChecked(params.show_spectrum_id_in_tree)

        self._import_use_be_cb.setChecked(params.import_use_binding_energy)
        self._import_use_cps_cb.setChecked(params.import_use_cps)

        self._nn_model_path_edit.setText(params.nn_model_path or "")
        self._nn_pred_threshold_sb.setValue(params.nn_pred_threshold)
        self._nn_smooth_cb.setChecked(params.nn_smooth)
        self._nn_interp_num_sb.setValue(params.nn_interp_num)

        self._optimization_kwargs_edit.setPlainText(self._dict_to_pretty_json(params.optimization_kwargs))

        self._serialization_mode_edit.setText(str(params.default_serialization_mode))
        self._serialization_path_edit.setText(str(params.default_serialization_path or ""))
        if params.default_serialization_indent is None:
            self._serialization_indent_sb.setValue(0)
        else:
            self._serialization_indent_sb.setValue(params.default_serialization_indent)

    def apply_to_params(self, params: AppParameters) -> None:
        """
        Apply current widget values to an AppParameters instance.

        Parameters
        ----------
        params : AppParameters
            Target parameters to mutate.

        Raises
        ------
        ValueError
            If optimization kwargs JSON is invalid.
        """
        params.automatic_methods = self._automatic_methods_cb.isChecked()
        params.default_background_model_for_auto_methods = self._default_bg_model_edit.text()
        params.show_spectrum_id_in_tree = self._show_spectrum_id_in_tree_cb.isChecked()

        params.import_use_binding_energy = self._import_use_be_cb.isChecked()
        params.import_use_cps = self._import_use_cps_cb.isChecked()

        nn_model_path = self._nn_model_path_edit.text().strip()
        params.nn_model_path = nn_model_path or None
        params.nn_pred_threshold = float(self._nn_pred_threshold_sb.value())
        params.nn_smooth = self._nn_smooth_cb.isChecked()
        params.nn_interp_num = int(self._nn_interp_num_sb.value())

        params.optimization_kwargs = self._parse_kwargs_json(
            self._optimization_kwargs_edit.toPlainText(),
        )

        mode_text = self._serialization_mode_edit.text().strip() or "replace"
        params.default_serialization_mode = mode_text  # type: ignore[assignment]

        path_text = self._serialization_path_edit.text().strip()
        params.default_serialization_path = Path(path_text) if path_text else None

        indent_value = int(self._serialization_indent_sb.value())
        params.default_serialization_indent = None if indent_value == 0 else indent_value

    def validate_and_apply(self, params: AppParameters) -> bool:
        """
        Validate current values and apply them to params.

        Parameters
        ----------
        params : AppParameters
            Target parameters to mutate.

        Returns
        -------
        bool
            True if validation succeeded and params were updated.
        """
        try:
            self.apply_to_params(params)
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return False
        return True

    def _dict_to_pretty_json(self, data: dict[str, Any]) -> str:
        """Return a stable, pretty JSON representation of a dictionary."""
        import json

        if not data:
            return "{}"
        return json.dumps(data, indent=2, sort_keys=True)

    def _parse_kwargs_json(self, text: str) -> dict[str, Any]:
        """Parse optimization kwargs from JSON text."""
        import json

        cleaned = text.strip()
        if not cleaned:
            return {}
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid optimization kwargs JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("Optimization kwargs must be a JSON object (mapping).")
        return value
