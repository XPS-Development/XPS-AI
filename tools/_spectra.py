#TODO: docs
import numpy as np

from tools._tools import interpolate, pseudo_voight


class Line():
    def __init__(self, loc, scale, const, gl_ratio, name=None):
        self.name = name
        self.loc = loc
        self.scale = scale
        self.const = const
        self.gl_ratio = gl_ratio

        self.fwhm = 2 * scale
        self.area = const * (1 + gl_ratio * (np.sqrt(2) * np.log(2) - 1))
        self.height = self.f(loc)

    def f(self, x):
        return pseudo_voight(x, self.loc, self.scale, self.const, self.gl_ratio)

    def __repr__(self):
        return f'Line(name={self.name}, loc={self.loc}, scale={self.scale}, const={self.const}, gl_ratio={self.gl_ratio})'


class Region():
    def __init__(self, x, y, y_norm, start_idx, end_idx):
        if x[0] > x[-1]:
            x = x[::-1]
            y = y[::-1]
            y_norm = y_norm[::-1]

        self.start_idx = start_idx
        self.end_idx = end_idx

        self.x = x
        self.y = y
        self.y_norm = y_norm

        self.background = None
        self.lines = []

    def append(self, line):
        self.lines.append(line)

    def add_line(self, loc, scale, const, gl_ratio, name=None):
        line = Line(loc, scale, const, gl_ratio, name=name)
        self.append(line)
    
    def delete_line(self, idx=None):
        if idx is not None:
            self.lines.pop(idx)
    
    def draw_lines(self):
        lines = [self.x, self.background]
        lines.extend([line.f(self.x) + self.background for line in self.lines])
        return lines

    def __repr__(self):
        s = f'Region(start={self.start_idx}, end={self.end_idx}'
        for line in self.lines:
            s += f'\n\t{line}'
        return s


class Spectrum():
    """Initialize tool for saving spectrum info."""
    def __init__(self, energies, intensities, name=None):
        self.name = name
        if energies[0] > energies[-1]:
            energies = energies[::-1].copy()
            intensities = intensities[::-1].copy()
        self.x = energies
        self.y = intensities
        self.regions = []
        self.preproc()

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

    def add_masks(self, peak_mask, max_mask):
        self.peak = peak_mask
        self.max = max_mask

    def get_masks(self):
        return self.peak, self.max

    def add_region(self, region):
        region.norm_coefs = self.norm_coefs
        self.regions.append(region)
    
    def create_region(self, start_idx, end_idx):
        region = Region(
            self.x[start_idx:end_idx].copy(), self.y[start_idx:end_idx].copy(), self.y_norm[start_idx:end_idx].copy(), start_idx, end_idx
        )
        self.add_region(region)
        return region
    
    def draw_spectrum(self):
        lines = []
        for region in self.regions:
            lines.extend(region.draw_lines())
        return self.x, self.y, *lines
    
    def charge_correction(self, delta=0):
        self.x += delta
        self.x_interpolated += delta
        for region in self.regions:
            region.x += delta
            for line in region.lines:
                line.loc += delta
    
    def __repr__(self):
        s = f'Spectrum(name={self.name})'
        for region in self.regions:
            s += f'\n\t{region}'
        return s
