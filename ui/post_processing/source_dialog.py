"""Select qualified trends from independently prepared spectrum series."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ui.controller import ControllerWrapper


class SourceDialog(QDialog):
    """Show each trend's series, coverage and component composition before selection."""

    def __init__(
        self,
        controller: ControllerWrapper,
        parent: QWidget | None = None,
        *,
        name: str = "New analysis",
        selected: tuple[tuple[str, str], ...] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Analytical table sources")
        self.resize(720, 540)
        self.name_edit = QLineEdit(name)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Series / peak trend", "Peaks / steps", "Components"])
        for sid in controller.query.get_post_processing_series_ids():
            dto = controller.query.get_post_processing_series(sid)
            root = QTreeWidgetItem([dto.series.name + (" — recalculate" if dto.stale else "")])
            self.tree.addTopLevelItem(root)
            if dto.series.result is None:
                continue
            names = dict(dto.series.track_names)
            for track in dto.series.result.tracks:
                points = [
                    (step, peak)
                    for step in dto.series.result.series.steps
                    for peak in step.peaks
                    if peak.component_id in track.component_ids
                ]
                composition = "; ".join(
                    f"{getattr(controller.query.get_metadata(step.spectrum_id), 'name', None) or step.spectrum_id}: E={peak.position:.6g}; S={peak.area}"
                    for step, peak in points
                )
                item = QTreeWidgetItem(
                    [
                        names.get(track.anchor_id, track.id_),
                        f"{len(track.component_ids)} / {len(dto.series.spectrum_ids)}",
                        "E: " + ", ".join(f"{p.position:.6g}" for _, p in points[:4]) + " eV",
                    ]
                )
                item.setToolTip(2, composition)
                item.setData(0, Qt.ItemDataRole.UserRole, (sid, track.id_))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    0,
                    Qt.CheckState.Checked
                    if (sid, track.id_) in selected
                    else Qt.CheckState.Unchecked,
                )
                if not dto.exportable and (sid, track.id_) not in selected:
                    item.setDisabled(True)
                root.addChild(item)
            root.setExpanded(True)
        # Keep unavailable existing sources visible and removable.
        represented = set()
        for i in range(self.tree.topLevelItemCount()):
            root = self.tree.topLevelItem(i)
            if root is not None:
                for j in range(root.childCount()):
                    child = root.child(j)
                    if child is not None:
                        represented.add(child.data(0, Qt.ItemDataRole.UserRole))
        for pair in selected:
            if pair not in represented:
                item = QTreeWidgetItem([f"Unavailable: {pair[0]} / {pair[1]}"])
                item.setData(0, Qt.ItemDataRole.UserRole, pair)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked)
                self.tree.addTopLevelItem(item)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Choose peak sequences. Row alignment follows the supplied spectrum order.")
        )
        layout.addWidget(self.name_edit)
        layout.addWidget(self.tree, 1)
        layout.addWidget(buttons)

    def selections(self) -> tuple[tuple[str, str], ...]:
        """Return explicitly checked series/trend identities."""
        result = []
        for i in range(self.tree.topLevelItemCount()):
            root = self.tree.topLevelItem(i)
            if root is None:
                continue
            items = (
                [root]
                if root.data(0, Qt.ItemDataRole.UserRole)
                else [root.child(j) for j in range(root.childCount())]
            )
            result.extend(
                item.data(0, Qt.ItemDataRole.UserRole)
                for item in items
                if item is not None and item.checkState(0) == Qt.CheckState.Checked
            )
        return tuple(result)
