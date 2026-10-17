"""Adjacent-spectrum review using shared plot data, styling and component colours."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.component_colors import color_for_component
from ui.plot.plot_area import PEN_BACKGROUND, PEN_MODEL, DoubleClickAutoRangeViewBox

if TYPE_CHECKING:
    from ui.controller import ControllerWrapper


class LinkEditor(QWidget):
    """Inspect neighbouring decompositions and explicitly assign or break links."""

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.series_id = None
        self.pair, self.left, self.right = QComboBox(), QComboBox(), QComboBox()
        self.links = QListWidget()
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")
        layout = QVBoxLayout(self)
        layout.addWidget(self.pair)
        row = QHBoxLayout()
        row.addWidget(self.left, 1)
        row.addWidget(self.right, 1)
        for label, linked in (("Assign link", True), ("Break link", False)):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, value=linked: self._apply(value))
            row.addWidget(button)
        layout.addLayout(row)
        layout.addWidget(self.error)
        layout.addWidget(self.graphics, 3)
        layout.addWidget(self.links, 1)
        self.pair.currentIndexChanged.connect(self._pair_changed)
        self.left.currentIndexChanged.connect(self._draw)
        self.right.currentIndexChanged.connect(self._draw)

    def set_series(self, series_id: str | None) -> None:
        """Keep the selected neighbour pair when a command refreshes the workspace."""
        current = self.pair.currentData() if self.series_id == series_id else None
        self.series_id = series_id
        self.pair.blockSignals(True)
        self.pair.clear()
        if series_id is not None:
            series = self.controller.query.get_post_processing_series(series_id).series
            if series.result is not None:
                steps = series.result.series.steps
                for a, b in pairwise(steps):
                    labels = [
                        getattr(self.controller.query.get_metadata(s.spectrum_id), "name", None)
                        or s.spectrum_id
                        for s in (a, b)
                    ]
                    self.pair.addItem(" → ".join(labels), (a.spectrum_id, b.spectrum_id))
        found = self.pair.findData(current)
        if found >= 0:
            self.pair.setCurrentIndex(found)
        self.pair.blockSignals(False)
        self._pair_changed()

    def _pair_changed(self) -> None:
        for combo in (self.left, self.right):
            combo.blockSignals(True)
            combo.clear()
        if self.series_id and self.pair.currentData():
            result = self.controller.query.get_post_processing_series(self.series_id).series.result
            if result is not None:
                for combo, sid in zip(
                    (self.left, self.right), self.pair.currentData(), strict=True
                ):
                    step = next(s for s in result.series.steps if s.spectrum_id == sid)
                    for peak in step.peaks:
                        combo.addItem(
                            f"{peak.component_id} — E={peak.position:.6g}; S={peak.area}",
                            peak.component_id,
                        )
        for combo in (self.left, self.right):
            combo.blockSignals(False)
        self._draw()

    def _draw(self) -> None:
        self.graphics.ci.clear()
        self.links.clear()
        if not self.series_id or not self.pair.currentData():
            return
        result = self.controller.query.get_post_processing_series(self.series_id).series.result
        if result is None:
            return
        by_component = {cid: track for track in result.tracks for cid in track.component_ids}
        pair = self.pair.currentData()
        steps = [next(s for s in result.series.steps if s.spectrum_id == sid) for sid in pair]
        for index, (step, combo) in enumerate(zip(steps, (self.left, self.right), strict=True)):
            plot = self.graphics.ci.addPlot(
                row=0, col=index, viewBox=DoubleClickAutoRangeViewBox(enableMenu=False)
            )
            plot.setLabel("bottom", "Binding energy", units="eV")
            plot.setLabel("left", "Intensity")
            plot.invertX(True)
            if step.spectrum_id not in self.controller.query.get_all_spectra_ids():
                continue
            data = self.controller.query.get_spectrum_plot_data(step.spectrum_id)
            for curve in data.curves:
                if curve.kind == "residual":
                    continue
                track = by_component.get(curve.component_id)
                color = color_for_component(
                    component_id=track.anchor_id if track else curve.component_id
                )
                pen = (
                    PEN_BACKGROUND
                    if curve.kind == "background"
                    else PEN_MODEL
                    if curve.kind == "model"
                    else pg.mkPen(
                        color, width=3 if curve.component_id == combo.currentData() else 1
                    )
                    if curve.kind == "peak"
                    else pg.mkPen("#999999")
                )
                plot.plot(curve.x, curve.y, pen=pen)
        diagram = self.graphics.ci.addPlot(
            row=1, col=0, colspan=2, viewBox=DoubleClickAutoRangeViewBox(enableMenu=False)
        )
        diagram.setLabel("bottom", "Peak position", units="eV")
        diagram.getAxis("left").setTicks([[(0, "Previous"), (1, "Next")]])
        for level, step in enumerate(steps):
            for peak in step.peaks:
                diagram.plot(
                    [peak.position],
                    [level],
                    pen=None,
                    symbol="o",
                    symbolBrush=color_for_component(
                        component_id=by_component[peak.component_id].anchor_id
                    ),
                )
        source = {p.component_id: p for p in steps[0].peaks}
        target = {p.component_id: p for p in steps[1].peaks}
        for track in result.tracks:
            for link in track.links:
                if link.source_id not in source or link.target_id not in target:
                    continue
                status = "manual" if link.manual else "ambiguous" if link.ambiguous else "automatic"
                self.links.addItem(f"{link.source_id} → {link.target_id} — {status}")
                color = (
                    "#e6a817"
                    if link.ambiguous
                    else color_for_component(component_id=track.anchor_id)
                )
                diagram.plot(
                    [source[link.source_id].position, target[link.target_id].position],
                    [0, 1],
                    pen=pg.mkPen(color, width=3 if link.manual else 1),
                )

    def _apply(self, linked: bool) -> None:
        if not self.series_id or not self.left.currentData() or not self.right.currentData():
            self.error.setText("Select a component in each neighbouring spectrum.")
            return
        try:
            self.controller.set_post_processing_link(
                self.series_id, self.left.currentData(), self.right.currentData(), linked
            )
        except ValueError as error:
            self.error.setText(str(error))
        else:
            self.error.clear()
