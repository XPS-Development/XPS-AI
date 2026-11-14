"""Edit explicit track operations without accepting arbitrary formula text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ui.controller import ControllerWrapper


class MetricDialog(QDialog):
    """Choose stable track operands, operand order and an explicit reference step."""

    def __init__(
        self,
        controller: ControllerWrapper,
        table_id: str,
        parent: QWidget | None = None,
        *,
        metric_id: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("PostProcessingDialog")
        self.setWindowTitle("Analysis column")
        self.resize(600, 560)
        self._controller, self._table_id, self._metric_id = controller, table_id, metric_id
        table = controller.query.get_analytical_table(table_id).state
        existing = next((m for m in table.metrics if m.id_ == metric_id), None)
        self._operations = controller.query.get_post_processing_operations()
        self.name_edit = QLineEdit("" if existing is None else existing.name)
        self.operation = QComboBox()
        for operation in self._operations:
            self.operation.addItem(operation.label, operation.id_)
        self.track_a, self.track_b, self.reference = QComboBox(), QComboBox(), QComboBox()
        self.numerator, self.denominator = QListWidget(), QListWidget()
        tracks = {}
        for source in table.sources:
            if source.series_id in controller.query.get_post_processing_series_ids():
                series = controller.query.get_post_processing_series(source.series_id).series
                label = f"{series.name} / {controller.query.get_trend_label(source.series_id, source.track_id)}"
            else:
                label = "Unavailable source"
            tracks[source.id_] = label
        if existing is not None:
            for tid in (*existing.tracks, *existing.denominator):
                tracks.setdefault(tid, "Unavailable source")
        for tid, label in tracks.items():
            self.track_a.addItem(label, tid)
            self.track_b.addItem(label, tid)
            for widget, selected in (
                (self.numerator, () if existing is None else existing.tracks),
                (self.denominator, () if existing is None else existing.denominator),
            ):
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, tid)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if tid in selected else Qt.CheckState.Unchecked
                )
                widget.addItem(item)
        for index, row in enumerate(table.rows, 1):
            labels = [
                getattr(controller.query.get_metadata(p.spectrum_id), "name", None) or p.spectrum_id
                for p in row.points
                if p.spectrum_id
            ]
            self.reference.addItem(f"{index}. {' / '.join(labels)}", row.id_)
        if (
            existing is not None
            and existing.reference_row_id
            and self.reference.findData(existing.reference_row_id) < 0
        ):
            self.reference.addItem("Unavailable reference row", existing.reference_row_id)
        if existing is not None:
            self.operation.setCurrentIndex(self.operation.findData(existing.operation))
            self.track_a.setCurrentIndex(self.track_a.findData(existing.tracks[0]))
            if len(existing.tracks) > 1:
                self.track_b.setCurrentIndex(self.track_b.findData(existing.tracks[1]))
            if existing.reference_row_id:
                self.reference.setCurrentIndex(self.reference.findData(existing.reference_row_id))
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.form = QFormLayout()
        for label, widget in (
            ("Name", self.name_edit),
            ("Operation", self.operation),
            ("Trend A / numerator", self.track_a),
            ("Trend B / denominator", self.track_b),
            ("Sum / numerator tracks", self.numerator),
            ("Denominator tracks", self.denominator),
            ("Reference row", self.reference),
        ):
            self.form.addRow(label, widget)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(self.form)
        layout.addWidget(
            QLabel(
                "Ratios use fitted model areas without sensitivity factors.\nMissing operands stay empty."
            )
        )
        layout.addWidget(self.error)
        layout.addWidget(buttons)
        self.operation.currentIndexChanged.connect(self._update_fields)
        self._update_fields()

    def _update_fields(self) -> None:
        option = self._operations[self.operation.currentIndex()]
        for widget, visible in (
            (self.track_a, option.operands in ("one", "two", "fraction")),
            (self.track_b, option.operands == "two"),
            (self.numerator, option.operands in ("sum", "two_sets")),
            (self.denominator, option.operands in ("fraction", "two_sets")),
            (self.reference, option.reference),
        ):
            self.form.setRowVisible(widget, visible)

    @staticmethod
    def _checked(widget: QListWidget) -> tuple[str, ...]:
        return tuple(
            widget.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(widget.count())
            if widget.item(i).checkState() == Qt.CheckState.Checked
        )

    def _save(self) -> None:
        option = self._operations[self.operation.currentIndex()]
        if option.operands in ("sum", "two_sets"):
            tracks = self._checked(self.numerator)
        elif option.operands == "two":
            tracks = (self.track_a.currentData(), self.track_b.currentData())
        else:
            tracks = (self.track_a.currentData(),)
        if not self.name_edit.text().strip() or not tracks or any(t is None for t in tracks):
            self.error.setText("Enter a name and select all required operands.")
            return
        denominator = (
            self._checked(self.denominator) if option.operands in ("fraction", "two_sets") else ()
        )
        if option.operands in ("fraction", "two_sets") and not denominator:
            self.error.setText("Select the denominator tracks.")
            return
        if option.operands == "fraction" and tracks[0] not in denominator:
            self.error.setText("The selected track must be included in the denominator set.")
            return
        self._controller.save_analytical_metric(
            self._table_id,
            name=self.name_edit.text(),
            operation=option.id_,
            tracks=tracks,
            denominator=denominator,
            reference_row_id=self.reference.currentData() if option.reference else None,
            metric_id=self._metric_id,
        )
        self.accept()
