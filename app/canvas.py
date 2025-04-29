import numpy as np
import pyqtgraph as pg

from PySide6 import QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


class PlotCanvas(pg.PlotWidget):
    # colors = [
    #     'darkblue',
    #     'crimson',
    #     'khaki',
    #     'orange',
    #     'lightgreen',
    #     'magenta',
    #     'lightpink',
    #     'deepskyblue'
    # ]

    def __init__(self, parent, workspace):        
        super().__init__(parent)

        self.parent = parent
        self.workspace = workspace

        self.showGrid(x=True, y=True)
        self.setMouseEnabled(x=True, y=True)
        self.setLabel("bottom", "Binding Energy (eV)")
        vb = self.getViewBox()
        # vb.setMouseMode(pg.ViewBox.RectMode)
        vb.setMenuEnabled(False)

        # Cursors to set region parameters
        self.c1 = self.create_cursor('start_point')
        self.c2 = self.create_cursor('end_point')
        self.cursor_pen = {'color': 'r', 'width': 2, 'style': Qt.DashLine}

        self.main_curves_color = self.palette().color(QtGui.QPalette.Text)
        self.mask_parameters = ((0, 0, 255, 100), {255, 0, 0, 255})
        self.setBackground(self.palette().color(QtGui.QPalette.Base))

        self.cursor_label = QLabel("Click Position: (x, y)")
        self.scene().sigMouseClicked.connect(self.mouse_clicked)
    
    def mouse_clicked(self, mouse_event):
            # if mouse_event.button():
            scene_pos = mouse_event.scenePos()
            vb = self.plotItem.vb
            if vb.sceneBoundingRect().contains(scene_pos):
                mouse_point = vb.mapSceneToView(scene_pos)
                x = mouse_point.x()
                y = mouse_point.y()
                self.cursor_label.setText(f"Click Position: ({x:.2f}, {y:.2f})")

    def reload_spectrum(self, spectrum):
        
        self.spectrum = spectrum
        self.setTitle(spectrum.name)

        x, y = spectrum.x, spectrum.y
        y_smooth = spectrum.y_smoothed
        regions = spectrum.regions

        self.main_curves = []
        self.regions_lines = []

        self.main_curves.extend([
            pg.PlotDataItem(x, y, pen={'color': self.main_curves_color, 'width': 2}),
            pg.PlotDataItem(x, y_smooth, pen={'color': self.main_curves_color, 'width': 2})
        ])

        self.create_masks()

        for region in regions:
            self.add_region(region)
    
    def _prepare_mask(self, mask):
        mask = mask.copy()
        v = np.lib.stride_tricks.sliding_window_view(mask, 3, writeable=True)
        f = np.array([0, 1, 0])
        v[(v == f).all(axis=1)] = np.array([[1, 1, 1]])
        return mask
    
    def create_masks(self):
        peak_mask, max_mask = self.spectrum.get_masks()
        if peak_mask is None or max_mask is None:
            return
        
        peak_mask = self._prepare_mask(peak_mask)
        max_mask = self._prepare_mask(max_mask)

        x = self.spectrum.x_interpolated
        y = self.spectrum.y_interpolated
        
        curve = pg.PlotDataItem(x, y, pen={'color': self.main_curves_color, 'width': 2})
        min_to_fill = np.zeros_like(x)

        self.main_curves.append(curve)

        if peak_mask.any():
            c1 = np.where(peak_mask, y, np.nan)
            c2 = np.where(peak_mask, min_to_fill, np.nan)
            curve_peak_1 = pg.PlotDataItem(x, c1, pen=self.mask_parameters[0])
            curve_peak_2 = pg.PlotDataItem(x, c2, pen=self.mask_parameters[0])
            fill_peak = pg.FillBetweenItem(curve_peak_1, curve_peak_2, brush=self.mask_parameters[0])
            self.main_curves.append(fill_peak)

        if max_mask.any():
            c1 = np.where(max_mask, y, np.nan)
            c2 = np.where(max_mask, min_to_fill, np.nan)
            curve_max_1 = pg.PlotDataItem(x, c1, pen=self.mask_parameters[1])
            curve_max_2 = pg.PlotDataItem(x, c2, pen=self.mask_parameters[1])
            fill_max = pg.FillBetweenItem(curve_max_1, curve_max_2, brush=self.mask_parameters[1])
            self.main_curves.append(fill_max)
    
    def delete_region(self, region_idx):
        region = self.regions_lines.pop(region_idx)
        for line in region:
            self.removeItem(line)
    
    def add_region(self, region):
        region_curves = []

        reg_x, back, s, *reg_lines = region.draw_lines()
        region_curves.extend([
            pg.PlotDataItem(reg_x, back, pen={'color': self.main_curves_color, 'width': 2, 'style': Qt.DashLine}),
            pg.PlotDataItem(reg_x, s, pen={'color': self.main_curves_color, 'width': 2, 'style': Qt.DotLine})
        ])

        for i, line in enumerate(reg_lines):
            color = pg.mkColor((i, len(reg_lines)))
            region_curves.append(
                pg.PlotDataItem(reg_x, line, pen={'color': color, 'width': 3})
            )
        self.regions_lines.append(region_curves)
    
    # def delete_line(self, region_idx, line_idx):
    #     line = self.regions_lines[region_idx].pop(2 + line_idx)
    #     self.removeItem(line)
    
    # def add_line(self, region_idx):
    #     self.regions_lines[region_idx].append(pg.PlotDataItem(0, 0, pen=next(self.colors)))
    
    def update_data(self):
        for region_curves, region in zip(self.regions_lines, self.spectrum.regions):
            reg_x, back, s, *reg_lines = region.draw_lines()
            for line_curve, data in zip(region_curves, (back, s, *reg_lines)):
                line_curve.setData(reg_x, data)
    
    def change_smoothing_plotting(self, smoothed=False):
        vis = {'color': self.main_curves_color, 'width': 2, 'alpha': 1}
        transp = {'color': self.main_curves_color, 'width': 1, 'alpha': 0.2}

        if smoothed:
            self.main_curves[0].setPen(transp)
            self.main_curves[1].setPen(vis)
        else:
            self.main_curves[0].setPen(vis)
            self.main_curves[1].setPen(transp)

    def update_plot(self, disp_type='lines', smoothed=False):
        self.clear()
        if disp_type == 'lines':
            self.addItem(self.main_curves[0])
            self.change_smoothing_plotting(smoothed)
            if smoothed:
                self.addItem(self.main_curves[1])
            for region_curves in self.regions_lines:
                for curve in region_curves:
                    self.addItem(curve)

        elif disp_type == 'labeled':
            for curve in self.main_curves[2:]:
                self.addItem(curve)

        elif disp_type == 'raw':
            self.addItem(self.main_curves[0])
            self.change_smoothing_plotting(smoothed)
            if smoothed:
                self.addItem(self.main_curves[1])

    def load_cursors(self, region):
        self.region = region
        self.set_cursors(region.start_point, region.end_point)

    def create_cursor(self, param):
        c = pg.InfiniteLine(angle=90, movable=True, pen=None)
        c.sigPositionChangeFinished.connect(lambda cursor: self.update_position(cursor.value(), param))
        return c

    def update_cursors(self, pos1, pos2):
        self.c1.setPos(pos1)
        self.c2.setPos(pos2)

    def set_cursors(self, pos1, pos2):
        self.update_cursors(pos1, pos2)
        self.c1.setPen(self.cursor_pen)
        self.c2.setPen(self.cursor_pen)
        self.addItem(self.c1)
        self.addItem(self.c2)
    
    def update_position(self, val, param):
        if self.spectrum is not None:
            x1 = self.c1.value()
            x2 = self.c2.value()
            if x1 > x2:
                x1, x2 = x2, x1
                self.workspace.change_region_parameter(self.region, self.spectrum, 'start_point', x1)
                self.workspace.change_region_parameter(self.region, self.spectrum, 'end_point', x2)
            else:
                self.workspace.change_region_parameter(self.region, self.spectrum, param, val)
        
        self.update_data()
        self.parent.sidebars.update_region_settings_tab()

