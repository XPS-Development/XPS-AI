from pathlib import Path

import numpy as np

from tools.parsers.vamas import VAMAS
from tools.parsers.specs import SPECS
from tools._spectra import Spectrum
from tools._analyzer import Analyzer


class Workspace():
    def __init__(self, model):
        self.analyzer = Analyzer(model)
        self.pred_threshold = 0.5
        self.charge_correction = 0
        self.groups = {}
    
    def _replace_none_name(self, name=None):
        if name is None:
            return f'group {len(self.groups)}'
        else:
            return name

    def create_group(self, name=None):
        name = self._replace_none_name(name)
        self.groups[name] = []

    def add_spectrum(self, x, y, group_name=None, name=None):
        spectrum = Spectrum(x, y, name=name)
        name = self._replace_none_name(name)
        if group_name not in self.groups:
            self.create_group(group_name)
        self.groups[group_name].append(spectrum)

    def load_txt(self, *files, group_name=None, type='casa'):
        name = self._replace_none_name(group_name)
        for f in files:
            with open(f, 'r') as f:
                name = f.readline().strip()
                data = np.loadtxt(f, delimiter='\t', skiprows=3, usecols=(1, 3))
            self.add_spectrum(data[:, 1], data[:, 0], group_name=group_name, name=name)

    def load_vamas(self, file, group_name=None):
        obj = VAMAS(file)
        if group_name is None:
            group_name = f'group {len(self.groups)}'
        for b in obj.blocks:
            name = b.name
            x = np.array(b.binding_axis, dtype=np.float32)
            y = np.array(b.data[0], dtype=np.float32)
            self.add_spectrum(x, y, name=name, group_name=group_name)

    def load_specs2(self, file, join_groups=True, group_name=None):
        obj = SPECS(file)
        l = len(self.groups)
        for i, g in enumerate(obj.groups):
            if join_groups:
                group_name = self._replace_none_name(group_name)
            else:
                group_name = f'group {i + l}'
            for r in g.regions:
                name = r.name
                x = r.binding_axis
                y = r.counts
                self.add_spectrum(x, y, name=name, group_name=group_name)

    def load_files(self, *files, join_groups=True, group_name=None):
        for f in files:
            if isinstance(f, str):
                f = Path(f)
            elif not isinstance(f, Path):
                print(f'File {f} must be str or Path')
                continue

            if f.suffix == '.txt':
                self.load_txt(f, group_name=group_name)
            elif f.suffix == '.vms':
                self.load_vamas(f, group_name=group_name)
            elif f.suffix == '.xml':
                self.load_specs2(f, join_groups=join_groups, group_name=group_name)

    def rename_group(self, group_name, new_group_name):
        self.groups[new_group_name] = self.groups.pop(group_name)
    
    def move_spectrum(self, spectrum_idx, group_name, new_group_name):
        self.groups[new_group_name].append(self.groups[group_name].pop(spectrum_idx))

    def merge_groups(self, group_1, *other_groups):
        for group in other_groups:
            self.groups[group_1] += self.groups[group]
            self.delete_group(group)

    def delete_group(self, group):
        self.groups.pop(group)

    def delete_spectrum(self, group_name, idx):
        self.groups[group_name].pop(idx)
    
    # def find_
    
    def aggregate_spectra(self, groups=None, idxs=None):
        spectra = []
        if idxs is None and not groups :
            for group in self.groups:
                spectra.extend(self.groups[group])
        elif idxs is None and isinstance(groups, str):
            spectra.extend(self.groups[groups])
        elif idxs is None and isinstance(groups, list):
            for group in groups:
                spectra.extend(self.groups[group])
        elif isinstance(idxs, int):
            spectra.append(self.groups[groups][idxs])
        elif isinstance(idxs, list):
            for idx in idxs:
                spectra.append(self.groups[groups][idx])
        return spectra
    
    def predict(self, groups=None, idxs=None, spectra=None):
        if spectra is None:
            spectra = self.aggregate_spectra(groups, idxs)
        self.analyzer.predict(*spectra, pred_threshold=self.pred_threshold)
    
    def post_process(self, groups=None, idxs=None, spectra=None):
        if spectra is None:
            spectra = self.aggregate_spectra(groups, idxs)
        self.analyzer.post_process(*spectra)

    def set_charge_correction(self, groups=None, idxs=None, spectra=None, current_line_energy=0, desired_line_energy=0):
        if spectra is None:
            spectra = self.aggregate_spectra(groups, idxs)

        delta = desired_line_energy - current_line_energy

        if self.charge_correction != 0:
            # revert charge correction
            for s in spectra:
                s.charge_correction(-self.charge_correction)

        self.charge_correction = delta
        for s in spectra:
            s.charge_correction(delta)

    def add_line(self, group=None, idx=None, region_idx=None, region=None, loc=0, scale=0, const=0, gl_ratio=0, name=None, line=None):
        if region is None:
            region = self.groups[group][idx].regions[region_idx]
        if line is not None:
            region.append(line)
        else:
            region.add_line(loc, scale, const, gl_ratio, name=name)

    def delete_line(self, group, idx, region_idx, line_idx):
        self.groups[group][idx].regions[region_idx].delete_line(line_idx)

    def change_line_parameter(
            self, group=None, idx=None, region_idx=None, line_idx=None, line=None, loc=None, scale=None, const=None, gl_ratio=None, name=None
    ):

        if line is None:
            line = self.groups[group][idx].regions[region_idx].lines[line_idx]
        
        if loc is not None:
            line.loc = loc
        if scale is not None:
            line.scale = scale
        if const is not None:
            line.const = const
        if gl_ratio is not None:
            line.gl_ratio = gl_ratio
        if name is not None:
            line.name = name

    def create_new_region(self, start, end, group=None, idx=None, spectrum=None, name=None, use_idxs=False, background_type='shirley'):
        if spectrum is None:
            spectrum = self.groups[group][idx]
        if not use_idxs:
            start = self.recalculate_point(start, group, idx, spectrum)
            end = self.recalculate_point(end, group, idx, spectrum)
        
        region = spectrum.create_region(start, end)
        self.recalculate_background(region=region)
        
    
    def recalculate_point(self, point_val, group=None, idx=None, spectrum=None):
        # find the closest point to the given value in the spectrum
        if spectrum is None:
            spectrum = self.groups[group][idx]
        return (np.abs(spectrum.x  - point_val)).argmin()

    def recalculate_background(self, group=None, idx=None, region_idx=None, region=None):
        if region is None:
            region = self.groups[group][idx].regions[region_idx]
        self.analyzer.calculate_region_background(region)
    
    def refit(
            self, group=None, idx=None, region_idx=None, region=None, use_norm_y=True, fixed_params=[], full_refit=False, tol=0.1, fit_alg='differential evolution'
    ):
        if region is None:
            region = self.groups[group][idx].regions[region_idx]
        self.analyzer.refit_region(region, use_norm_y, fixed_params, full_refit, tol, fit_alg)

    #TODO: build trend
    def build_trend(self, param, lines, x):
        params = self.analyzer.aggregate_params(param, lines)

    def __repr__(self):
        return f'Workspace(groups={self.groups})'
