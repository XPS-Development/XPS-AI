"""Adjacent-spectrum review using shared plot data, styling and component colours."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.component_colors import color_for_component
from ui.plot.plot_area import PEN_BACKGROUND, DoubleClickAutoRangeViewBox

if TYPE_CHECKING:
    from ui.controller import ControllerWrapper


class LinkEditor(QWidget):
    """Inspect neighbouring decompositions and explicitly assign or break links."""

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.series_id = None
        self.track_id: str | None = None
        self.pair, self.left, self.right = QComboBox(), QComboBox(), QComboBox()
        self.scope = QComboBox()
        self.scope.addItems(("Neighbouring pair", "Whole series"))
        self.scope.setCurrentIndex(1)
        self.spacing = QDoubleSpinBox()
        self.spacing.setRange(0.05, 2.0)
        self.spacing.setSingleStep(0.05)
        self.spacing.setDecimals(2)
        self.spacing.setValue(0.18)
        self.spacing.setSuffix(" of height")
        self.spacing.setToolTip(
            "Vertical step as a fraction of spectrum height; smaller values overlap more"
        )
        self.components = QCheckBox("Decomposition")
        self.show_links = QCheckBox("Peak links")
        self.normalize = QCheckBox("Normalize heights")
        self.normalize.setToolTip(
            "Scale each displayed spectrum to equal height; saved intensities stay unchanged"
        )
        self.links = QListWidget()
        self.links.setObjectName("PostProcessingTrendColumns")
        self.links.setMaximumHeight(86)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.hide()
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(self.scope)
        controls.addWidget(self.pair, 1)
        layout.addLayout(controls)
        row = QHBoxLayout()
        row.addWidget(self.left, 1)
        row.addWidget(self.right, 1)
        for label, linked in (("Assign link", True), ("Break link", False)):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, value=linked: self._apply(value))
            row.addWidget(button)
        layout.addLayout(row)
        layout.addWidget(self.error)
        display = QHBoxLayout()
        display.addWidget(QLabel("Vertical step"))
        display.addWidget(self.spacing)
        display.addWidget(self.normalize)
        display.addWidget(self.components)
        display.addWidget(self.show_links)
        display.addStretch()
        layout.addLayout(display)
        hint = QLabel(
            "Reduce the vertical step for a denser stack. Enable peak links to select peaks on the plot."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.graphics, 1)
        layout.addWidget(self.links)
        self.scope.currentIndexChanged.connect(self._draw)
        self.normalize.toggled.connect(self._draw)
        self.spacing.valueChanged.connect(self._draw)
        self.components.toggled.connect(self._draw)
        self.show_links.toggled.connect(self._draw)
        self.pair.currentIndexChanged.connect(self._pair_changed)
        self.left.currentIndexChanged.connect(self._draw)
        self.right.currentIndexChanged.connect(self._draw)

    def set_series(self, series_id: str | None) -> None:
        """Keep the selected neighbour pair when a command refreshes the workspace."""
        current = self.pair.currentData() if self.series_id == series_id else None
        if self.series_id != series_id:
            self.track_id = None
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

    def set_track(self, track_id: str) -> None:
        """Keep the selected trend and the editable peak pair in agreement."""
        self.track_id = track_id
        self._pair_changed()

    def _pair_changed(self) -> None:
        previous = (self.left.currentData(), self.right.currentData())
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
                    for index, peak in enumerate(step.peaks, 1):
                        combo.addItem(
                            f"Peak {index} · {peak.position:.6g} eV",
                            peak.component_id,
                        )
                track = next((t for t in result.tracks if t.id_ == self.track_id), None)
                for combo, old in zip((self.left, self.right), previous, strict=True):
                    selected = combo.findData(old)
                    if track is not None:
                        selected = next(
                            (
                                i
                                for i in range(combo.count())
                                if combo.itemData(i) in track.component_ids
                            ),
                            -1,
                        )
                    if track is not None or selected >= 0:
                        combo.setCurrentIndex(selected)
        for combo in (self.left, self.right):
            combo.blockSignals(False)
        self.error.clear()
        self.error.hide()
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
        step_numbers = {s.spectrum_id: i for i, s in enumerate(result.series.steps, 1)}
        component_steps = {
            p.component_id: s.spectrum_id for s in result.series.steps for p in s.peaks
        }
        steps = (
            result.series.steps
            if self.scope.currentIndex() == 1
            else tuple(s for s in result.series.steps if s.spectrum_id in pair)
        )
        plot = self.graphics.ci.addPlot(viewBox=DoubleClickAutoRangeViewBox(enableMenu=False))
        plot.setLabel("bottom", "Binding energy", units="eV")
        plot.setLabel(
            "left",
            "Intensity (normalized, offset)"
            if self.normalize.isChecked()
            else "Intensity (shared scale, offset)",
        )
        plot.getAxis("left").setStyle(showValues=False)
        plot.invertX(True)
        live_ids = set(self.controller.query.get_all_spectra_ids())
        data = {
            s.spectrum_id: self.controller.query.get_spectrum_plot_data(s.spectrum_id)
            for s in steps
            if s.spectrum_id in live_ids
        }
        bounds = {}
        for sid, spectrum in data.items():
            values = [c.y[np.isfinite(c.y)] for c in spectrum.curves if c.kind == "raw"]
            values = [v for v in values if v.size]
            if values:
                bounds[sid] = (
                    min(float(v.min()) for v in values),
                    max(float(v.max()) for v in values),
                )
        shared_scale = max((hi - lo for lo, hi in bounds.values()), default=1.0) or 1.0
        markers = {}
        selected_ids = {self.left.currentData(), self.right.currentData()}
        for index, step in enumerate(steps):
            sid = step.spectrum_id
            if sid not in bounds:
                continue
            baseline, maximum = bounds[sid]
            scale = ((maximum - baseline) or 1.0) if self.normalize.isChecked() else shared_scale
            offset = (len(steps) - index - 1) * self.spacing.value()
            active = sid in pair
            for curve in data[sid].curves:
                if curve.kind == "residual":
                    continue
                track = by_component.get(curve.component_id)
                color = color_for_component(
                    component_id=track.anchor_id if track else curve.component_id
                )
                pen = (
                    PEN_BACKGROUND
                    if curve.kind == "background"
                    else pg.mkPen("#333333", width=1.3)
                    if curve.kind == "model"
                    else pg.mkPen(color, width=2 if curve.component_id in selected_ids else 1)
                    if curve.kind == "peak"
                    else pg.mkPen("#333333", width=1.0)
                )
                item = plot.plot(curve.x, (curve.y - baseline) / scale + offset, pen=pen)
                item.setZValue(index * 10 + (0 if curve.kind == "raw" else 1))
                item.setVisible(curve.kind == "raw" or self.components.isChecked())
                if curve.kind == "raw":
                    # Each nearer spectrum masks the curves behind it, like paper sheets.
                    item.setFillLevel(offset)
                    item.setBrush("w")
                if curve.kind == "peak":
                    peak = next(
                        (p for p in step.peaks if p.component_id == curve.component_id), None
                    )
                    if peak is not None and curve.x.size:
                        nearest = int(np.argmin(np.abs(curve.x - peak.position)))
                        y = float((curve.y[nearest] - baseline) / scale + offset)
                        if np.isfinite(y):
                            markers[peak.component_id] = (peak.position, y)
            raw_x = [c.x for c in data[sid].curves if c.kind == "raw" and c.x.size]
            if raw_x:
                name = (
                    getattr(self.controller.query.get_metadata(sid), "name", None)
                    or f"Step {index + 1}"
                )
                label = pg.TextItem(
                    f"{step_numbers[sid]}. {name}",
                    color="#444444" if active else "#888888",
                    anchor=(1, 1),
                )
                label.setPos(min(float(x.min()) for x in raw_x), offset + 0.05)
                label.setZValue(len(steps) * 10 + 3)
                plot.addItem(label)
        # Connect peaks in the same coordinates as the measured spectra.
        for track in result.tracks:
            for link in track.links:
                if link.source_id not in markers or link.target_id not in markers:
                    continue
                a, b = markers[link.source_id], markers[link.target_id]
                active = link.source_id in selected_ids or link.target_id in selected_ids
                color = QColor(
                    "#e6a817"
                    if link.ambiguous
                    else color_for_component(component_id=track.anchor_id)
                )
                color.setAlpha(190 if active else 65)
                if self.show_links.isChecked():
                    line = plot.plot(
                        [a[0], b[0]],
                        [a[1], b[1]],
                        pen=pg.mkPen(color, width=1.5 if active else 1, style=Qt.PenStyle.DashLine),
                    )
                    line.setZValue(len(steps) * 10 + 1)
                if (component_steps[link.source_id], component_steps[link.target_id]) == pair:
                    status = (
                        "manual" if link.manual else "ambiguous" if link.ambiguous else "automatic"
                    )
                    self.links.addItem(f"{a[0]:.6g} → {b[0]:.6g} eV — {status}")
        for step in steps:
            for peak in step.peaks:
                if not self.show_links.isChecked() or peak.component_id not in markers:
                    continue
                x, y = markers[peak.component_id]
                color = color_for_component(component_id=by_component[peak.component_id].anchor_id)
                point = pg.ScatterPlotItem(
                    [x],
                    [y],
                    size=9 if peak.component_id in selected_ids else 6,
                    pen=pg.mkPen(color),
                    brush="w",
                    hoverable=True,
                    hoverBrush=color,
                    tip=lambda _x, _y, info: info,
                    data=[f"{peak.position:.6g} eV"],
                )
                point.setZValue(len(steps) * 10 + 2)
                point.sigClicked.connect(
                    lambda _item, _points, _event, sid=step.spectrum_id, cid=peak.component_id: (
                        QTimer.singleShot(0, self, lambda: self._select_peak(sid, cid))
                    )
                )
                plot.addItem(point)
        plot.autoRange()

    def _select_peak(self, spectrum_id: str, component_id: str) -> None:
        pair = self.pair.currentData()
        if pair is None:
            return
        if spectrum_id not in pair:
            for index in range(self.pair.count()):
                if spectrum_id in self.pair.itemData(index):
                    self.pair.setCurrentIndex(index)
                    pair = self.pair.currentData()
                    break
        combo = self.left if spectrum_id == pair[0] else self.right
        index = combo.findData(component_id)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _apply(self, linked: bool) -> None:
        if not self.series_id or not self.left.currentData() or not self.right.currentData():
            self.error.setText("Select a component in each neighbouring spectrum.")
            self.error.show()
            return
        try:
            self.controller.set_post_processing_link(
                self.series_id, self.left.currentData(), self.right.currentData(), linked
            )
        except ValueError as error:
            self.error.setText(str(error))
            self.error.show()
        else:
            self.error.clear()
            self.error.hide()
