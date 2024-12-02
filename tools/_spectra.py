#TODO: отрефакторить папку

#TODO: docs

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

from tools.parsers.vamas import VAMAS
from tools.parsers.specs import SPECS
from tools._utils import interpolate, peak_sum, pseudo_voight


class Line():
    def __init__(self, loc, scale, const, gl_ratio, name=None):
        self.name = name
        self.loc = loc
        self.scale = scale
        self.const = const
        self.gl_ratio = gl_ratio

        self.fwhm = 2 * scale
        self.area = const * (1 + gl_ratio * (np.sqrt(2) * np.log(2) - 1))

    def draw_line(self, x):
        return pseudo_voight(x, self.loc, self.scale, self.const, self.gl_ratio)

    def __repr__(self):
        return f'Line(loc={self.loc}, scale={self.scale}, const={self.const}, gl_ratio={self.gl_ratio})'

#TODO: сделать self.x, self.y как сырые, сделать препроцессинг
class Spectrum():
    """Initialize tool for saving spectrum info."""
    def __init__(self, energies, intensities, name=None):
        self.name = name
        self.x = energies
        self.y = intensities
        self.lines = []
        self.background = None
        self.preproc()

    #TODO: функция для инициализации переводных коэффициентов
    def preproc(self):
        # preproc
        x, y = interpolate(self.x, self.y)
        min_value = y.min()
        max_value = y.max()
        y = (y - min_value)/(max_value - min_value)
        self.norm_coefs = (min_value, max_value)

        if x[0] > x[-1]:
            # copy to prevent negative stride error in torch
            x = x[::-1].copy()
            y = y[::-1].copy()

        self.norm_x = x
        self.norm_y = y
    
    def add_line(self, loc, scale, const, gl_ratio, name=None, norm=True):
        line = Line(loc, scale, const, gl_ratio, name=name)
        self.lines.append(line)

    def add_masks(self, peak_mask, max_mask):
        self.peak = peak_mask
        self.max = max_mask

    def get_masks(self):
        return self.peak, self.max

    def create_lines(self):
        main_y = np.zeros_like(self.x, dtype=np.float32) + self.background
        y_lines = [main_y]
        for line in self.lines:
            y_line = line.draw_line(self.x)
            y_lines[0] += y_line
            y_lines.append(y_line + self.background)
        y_lines.append(self.background)
        return y_lines


class Groupe():
    def __init__(self, name=None):
        self.name = name
        self.spectra = []

    def add_spectrum(self, x, y, name=None):
        spectrum = Spectrum(x, y, name=name)
        self.spectra.append(spectrum)

    @classmethod
    def load_casa(cls, *files):
        groupe = cls()
        for f in files:
            with open(f, 'r') as f:
                name = f.readline().strip()
                data = np.loadtxt(f, delimiter='\t', skiprows=3, usecols=(1, 3))
            groupe.add_spectrum(data[:, 1], data[:, 0], name=name)
        return groupe

    @classmethod
    def load_vamas(cls, file):
        obj = VAMAS(file)
        groupe = cls()
        for b in obj.blocks:
            name = b.name
            x = np.array(b.binding_axis, dtype=np.float32)
            y = np.array(b.data[0], dtype=np.float32)
            groupe.add_spectrum(x, y, name=name)
        return groupe

    @classmethod
    def load_specs2(cls, file):
        obj = SPECS(file)
        groups = []
        for i, g in enumerate(obj.groups):
            groupe = cls(f'Groupe {i}')
            for r in g.regions:
                name = r.name
                x = r.binding_axis
                y = r.counts
                groupe.add_spectrum(x, y, name=name)
            groups.append(groupe)
        return groups

    def __getitem__(self, index):
        return self.spectra[index]

    def __len__(self):
        return len(self.spectra)
