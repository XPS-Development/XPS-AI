#TODO: docs
import os

import numpy as np
from copy import deepcopy

from scipy.signal import savgol_filter
from tools._tools import interpolate, pseudo_voight


class Line():
    #TODO: проверить площадь
    def __init__(self, loc, scale, const, gl_ratio, color=None):
        self.loc = loc
        self.scale = scale
        self._c = const
        self._gl = gl_ratio

        self.area = const * (1 + gl_ratio * (np.sqrt(2) * np.log(2) - 1))
        self.height = self.f(loc)

        self.color = color
    
    @property
    def fwhm(self):
        return 2 * self.scale
    
    @fwhm.setter
    def fwhm(self, fwhm):
        self.scale = fwhm / 2
    
    @property
    def gl_ratio(self):
        return self._gl
    
    @gl_ratio.setter
    def gl_ratio(self, gl_ratio):
        self._gl = gl_ratio
        self.area = self.const * (1 + self.gl_ratio * (np.sqrt(2) * np.log(2) - 1))
    
    @gl_ratio.getter
    def gl_ratio(self):
        return self._gl
    
    @property
    def const(self):
        return self._c
    
    @const.setter
    def const(self, const):
        self._c = const
        self.area = const * (1 + self.gl_ratio * (np.sqrt(2) * np.log(2) - 1))
    
    @const.getter
    def const(self):
        return self._c

    def f(self, x):
        return pseudo_voight(x, self.loc, self.scale, self.const, self.gl_ratio)

    def get_params(self, xps_peak_like=True):
        if xps_peak_like:
            # params = 
            params = list(map(lambda x: f'{x:.3f}', [self.loc, self.area, self.fwhm]))
            params.append(f'{round(self.gl_ratio * 100)}')
            return params
        else:
            return [self.loc, self.fwhm, self.const, self.gl_ratio, self.area, self.height]

    def __repr__(self):
        return f'Line(Position={self.loc}, Scale={self.scale}, Amplitude={self.const}, GL%={self.gl_ratio})'


class Region():
    def __init__(self, x, y, y_norm, i_1=None, i_2=None, start_idx=None, end_idx=None, background_type='shirley'):
        if x[0] > x[-1]:
            x = x[::-1]
            y = y[::-1]
            y_norm = y_norm[::-1]

        self.x = x
        self.y = y
        self.y_norm = y_norm

        self.start_idx = start_idx
        self.end_idx = end_idx

        if i_1 is None and i_2 is None:
            self.i_1 = y[0]
            self.i_2 = y[-1]
        else:
            self.i_1 = i_1
            self.i_2 = i_2

        self.start_point = x[0]
        self.end_point = x[-1]

        self.background_type = background_type
        self.background = None
        self.lines = []
    
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, x):
        self._x = x
        self.start_point = x[0]
        self.end_point = x[-1]
    
    @x.getter
    def x(self):
        return self._x

    def append(self, line):
        self.lines.append(line)

    def add_line(self, loc, scale, const, gl_ratio, color=None):
        line = Line(loc, scale, const, gl_ratio, color=color)
        self.append(line)
    
    def delete_line(self, idx=None):
        if idx is not None:
            self.lines.pop(idx)
    
    def draw_lines(self):
        lines = [self.x, self.background, self.peak_sum()]
        lines.extend([line.f(self.x) + self.background for line in self.lines])
        return lines

    def peak_sum(self):
        s = deepcopy(self.background)
        for l in self.lines:
            s += l.f(self.x)
        return s
    
    def export_params(self, xps_peak_like=True):
        params = []
        for line in self.lines:
            params.append(line.get_params(xps_peak_like=xps_peak_like))
        return params

    def __repr__(self):
        s = f'Region(start={self.x[0]}, end={self.x[-1]})'
        return s

class Spectrum():
    """Initialize tool for saving spectrum info."""
    def __init__(
            self, energies, intensities, 
            name=None, file=None, group=None,
            window_length=10, poly_order=2
    ):

        self.name = name
        self.file = file
        self.group = group

        if energies[0] > energies[-1]:
            energies = energies[::-1].copy()
            intensities = intensities[::-1].copy()

        self.x = energies
        self.y = intensities
        self.is_predicted = False
        self.is_analyzed = False

        if len(self.x) > 500:
            self.is_survey = True
        else:
            self.is_survey = False

        self.peak = None
        self.max = None

        self.charge_correction = 0
        self.regions = []
        self.preproc()
        self.add_smoothing(window_length=window_length, poly_order=poly_order)
    
    def add_smoothing(self, window_length=10, poly_order=2):
        self.y_smoothed = savgol_filter(self.y, window_length, poly_order)
        self.y_norm_smoothed = savgol_filter(self.y_norm, window_length, poly_order)

    def preproc(self):
        x, y = self.x, self.y
        min_value = y.min()
        max_value = y.max()
        y_norm = (y - min_value) / (max_value - min_value)
        self.y_norm = y_norm
        self.norm_coefs = (min_value, max_value)

        x_interpolated, y_interpolated = interpolate(x, y_norm)

        if x[0] > x[-1]:
            # copy to prevent negative stride error in torch
            x_interpolated = x_interpolated[::-1].copy()
            y_interpolated = y_interpolated[::-1].copy()

        self.x_interpolated = x_interpolated
        self.y_interpolated = y_interpolated

    def add_masks(self, peak_mask, max_mask, init_mask=False):
        self.is_predicted = True
        if init_mask:
            self.init_peak = peak_mask
            self.init_max = max_mask
        self.peak = peak_mask
        self.max = max_mask

    def get_masks(self):
        return self.peak, self.max

    def add_region(self, region):
        region.norm_coefs = self.norm_coefs
        self.regions.append(region)

    def create_region(self, start_idx, end_idx, background_type='shirley'):
        self.is_analyzed = True # prevent post processing for manually analyzed spectra
        region = Region(
            self.x[start_idx:end_idx], self.y[start_idx:end_idx], self.y_norm[start_idx:end_idx], \
            self.y_smoothed[start_idx], self.y_smoothed[end_idx-1], start_idx, end_idx, background_type
        )
        self.add_region(region)
        return region

    def delete_region(self, r):
        if isinstance(r, Region):
            self.regions.remove(r)
        elif isinstance(r, int):
            self.regions.pop(r)
    
    def change_region_range(self, region, start_idx, end_idx):
        region.start_idx = start_idx
        region.end_idx = end_idx
        region.x = self.x[start_idx:end_idx]
        region.y = self.y[start_idx:end_idx]
        region.y_norm = self.y_norm[start_idx:end_idx]
        region.i_1 = self.y_smoothed[start_idx]
        region.i_2 = self.y_smoothed[end_idx-1]
    
    def set_charge_correction(self, delta):
        self.charge_correction = delta
        self.x += delta
        self.x_interpolated += delta
        for region in self.regions:
            for line in region.lines:
                line.loc += delta
    
    def remove_charge_correction(self):
        self.set_charge_correction(-self.charge_correction)
        self.charge_correction = 0

    def save_spectrum(self, file_name, drop_empty=True):
        back_to_save = np.zeros_like(self.x)
        peak_sum_to_save = np.zeros_like(self.x)
        new_lines = []

        for region in self.regions:
            start_idx = region.start_idx
            end_idx = region.end_idx
            reg_x, back, s, *lines = region.draw_lines()
            back_to_save[start_idx:end_idx] = back
            peak_sum_to_save[start_idx:end_idx] = s

            for line in lines:
                new_line = np.zeros_like(self.x)
                new_line[start_idx:end_idx] = line
                new_lines.append(new_line)

        peak_names = [f'Peak {n}' for n in range(1, len(new_lines) + 1)]
        column_names = ['B.E.(eV)', 'Raw Intensity', 'Peak Sum', 'Background']
        column_names.extend(peak_names)
        header = '{:<20}' * (len(column_names) - 1) + '{}'
        header = header.format(*column_names) 
        array_to_save = np.vstack((self.x, self.y, peak_sum_to_save, back_to_save, *new_lines)).T

        if drop_empty:
            array_to_save = array_to_save.round(3)
            array_to_save = array_to_save.astype(str)
            array_to_save[array_to_save == '0.0'] = ' '
            np.savetxt(file_name, array_to_save, header=header, fmt='%-19s', comments='')
        else:
            np.savetxt(file_name, array_to_save, header=header, fmt='%-19.3f', comments='')

    def export_params(self, file_name: str, xps_peak_like=True):        
        params = []
        for region in self.regions:
            params.extend(region.export_params(xps_peak_like=xps_peak_like))
        for n, line_param in enumerate(params):
            line_param.insert(0, str(n))

        if xps_peak_like:
            pattern = '{:<14}{:<14}{:<14}{:<14}{}'
            header = pattern.format('Peak', 'Position (eV)', 'Area', 'FWHM (eV)', '%GL (%)')
            with open(file_name, 'w') as f:
                f.write(header + '\n')
                for line in params:
                    line = pattern.format(*line)
                    f.write(line + '\n')
        else:
            header = ['Peak', 'Position (eV)', 'Scale', 'Amplitude', '%GL (%)', 'Area', 'Height']
            np.savetxt(file_name, params, delimiter=' ', header=header, fmt='%<14.3f')
    
    def get_params(self, xps_peak_like=True):
        params = []
        for region in self.regions:
            params.extend(region.export_params(xps_peak_like=xps_peak_like))
        return params

    def view_data(self, *args, **kwargs):
        self._matplotlib_view(*args, **kwargs)

    def _matplotlib_view(self, ax=None, disp_type='lines', show=False, norm=False, smoothed=False):
        from matplotlib import pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()

        if norm or disp_type == 'labels':
            mask_params = ({'color': 'b', 'alpha': 0.2}, {'color': 'r'})
            x = self.x_interpolated
            y = self.y_interpolated
            ax.plot(x, y, 'k')
            if disp_type == 'labels':
                min_to_fill = self.y_interpolated.min()
                for mask, mask_param in zip(self.get_masks(), mask_params):
                    ax.fill_between(x, y, min_to_fill, where=mask > 0, **mask_param) # masks

        else:
            if smoothed:
                ax.plot(self.x, self.y_smoothed, 'k')
                ax.plot(self.x, self.y, 'k', alpha=0.3)
            else:
                ax.plot(self.x, self.y, 'k')

            if disp_type == 'lines':
                for region in self.regions:
                    reg_x, back, s, *lines = region.draw_lines()
                    ax.plot(reg_x, back, 'k--') # background
                    ax.plot(reg_x, s, 'k', alpha=0.7) # peak sum
                    for line in lines:
                        ax.plot(reg_x, line) # lines

        if show:
            plt.show()
        return ax
    
    def __repr__(self):
        s = f'Spectrum(name={self.name})'
        return s
