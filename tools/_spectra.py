#TODO: docs
import numpy as np

from tools.parsers.vamas import VAMAS
from tools.parsers.specs import SPECS
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

    def f(self, x):
        return pseudo_voight(x, self.loc, self.scale, self.const, self.gl_ratio)

    def __repr__(self):
        return f'Line(loc={self.loc}, scale={self.scale}, const={self.const}, gl_ratio={self.gl_ratio})'


class Region():
    def __init__(self, start, end, i_1, i_2):
        if start > end:
            start, end = end, start
        if i_1 > i_2:
            i_1, i_2 = i_2, i_1

        self.start_x = start
        self.end_x = end
        self.i_1 = i_1
        self.i_2 = i_2
        self.background = None
        self.lines = []

    def add_line(self, loc, scale, const, gl_ratio, name=None):
        line = Line(loc, scale, const, gl_ratio, name=name)
        self.lines.append(line)

    def draw_background(self, x):
        return self.background
    
#TODO: сделать self.x, self.y как сырые, сделать препроцессинг
class Spectrum():
    """Initialize tool for saving spectrum info."""
    def __init__(self, energies, intensities, name=None):
        self.name = name
        self.x = energies
        self.y = intensities
        self.regions = []
        self.preproc()

    #TODO: функция для инициализации переводных коэффициентов
    def preproc(self):
        x, y = self.x, self.y
        min_value = y.min()
        max_value = y.max()
        y_norm = (y - min_value)/(max_value - min_value)
        self.y_norm = y_norm
        self.norm_coefs = (min_value, max_value)

        x_interpolated, y_interpolated = interpolate(x, y_norm)

        if x[0] > x[-1]:
            # copy to prevent negative stride error in torch
            x_interpolated = x_interpolated[::-1].copy()
            y_interpolated = y_interpolated[::-1].copy()

        self.x_interpolated = x
        self.y_interpolated = y

    def add_masks(self, peak_mask, max_mask):
        self.peak = peak_mask
        self.max = max_mask

    def get_masks(self):
        return self.peak, self.max

    def creae_region(self, start, end, i_1, i_2):
        region = Region(start, end, i_1, i_2)
        self.regions.append(region)
        return region

    def create_lines(self):
        main_y = np.zeros_like(self.x, dtype=np.float32) + self.background
        y_lines = [main_y]
        for line in self.lines:
            y_line = line.draw_line(self.x)
            y_lines[0] += y_line
            y_lines.append(y_line + self.background)
        y_lines.append(self.background)
        return y_lines


class Workspace():
    def __init__(self, analyzer, name=None):
        self.name = name
        self.analyzer = analyzer
        self.groups = {}

    def create_groupe(self, name=None):
        if name is None:
            name = f'Groupe {len(self.groups)}'
        self.groups[name] = []

    def add_spectrum(self, x, y, groupe_name=None, name=None):
        spectrum = Spectrum(x, y, name=name)
        if groupe_name is None:
            groupe_name = f'Groupe {len(self.groups)}'
        if groupe_name not in self.groups:
            self.create_groupe(groupe_name)
        self.groups[groupe_name].append(spectrum)

    def load_casa(self, *files, group_name=None):
        for f in files:
            with open(f, 'r') as f:
                name = f.readline().strip()
                data = np.loadtxt(f, delimiter='\t', skiprows=3, usecols=(1, 3))
            self.add_spectrum(data[:, 1], data[:, 0], groupe_name=group_name, name=name)

    def load_vamas(self, file, groupe_name=None):
        obj = VAMAS(file)
        for b in obj.blocks:
            name = b.name
            x = np.array(b.binding_axis, dtype=np.float32)
            y = np.array(b.data[0], dtype=np.float32)
            self.add_spectrum(x, y, name=name, groupe_name=groupe_name)

    def load_specs2(self, file):
        obj = SPECS(file)
        for i, g in enumerate(obj.groups):
            groupe_name = f'Groupe {i}'
            for r in g.regions:
                name = r.name
                x = r.binding_axis
                y = r.counts
                self.add_spectrum(x, y, name=name, groupe_name=groupe_name)

    def load_files(self, *files):
        for f in files:
            if f.suffix == '.txt':
                self.load_casa(f)
            elif f.suffix == '.vms':
                self.load_vamas(f)
            elif f.suffix == '.xml':
                self.load_specs2(f)

    def rename_group(self, groupe_name, new_groupe_name):
        self.groups[new_groupe_name] = self.groups.pop(groupe_name)
    
    def move_spectrum(self, groupe_name, idx, new_groupe_name):
        self.groups[new_groupe_name].append(self.groups[groupe_name].pop(idx))

    def merge_groups(self, groupe_1, groupe_2):
        self.groups[groupe_1] = self.groups[groupe_1] + self.groups[groupe_2]
        self.delete_group(groupe_2)

    def delete_group(self, group):
        self.groups.pop(group)

    def delete_spectrum(self, groupe_name, idx):
        self.groups[groupe_name].pop(idx)    
    
    def analyze(self, groupe_name=None, idxs=None):
        pass

    def __repr__(self):
        return f'Workspace(name={self.name}, groups={self.groups})'
