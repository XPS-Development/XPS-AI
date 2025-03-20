import pickle as pkl
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
        self.groups = {}

    def set_prediction_threshold(self, threshold):
        self.pred_threshold = threshold
        spectra = [s for s in self.aggregate_spectra() if s.is_predicted]
        self.analyzer.restrict_mask(*spectra, threshold=threshold)

    def create_group(self, group_name):
        self.groups[group_name] = []

    def add_spectrum(self, x, y, group_name=None, name=None):
        spectrum = Spectrum(x, y, name=name)
        if group_name not in self.groups:
            self.create_group(group_name)
        self.groups[group_name].append(spectrum)

    def load_txt(self, file: Path, group_name=None, type='casa'):
        if group_name is None:
            group_name = f.name
        with open(file, 'r') as f:
            name = f.readline().strip()
            data = np.loadtxt(f, delimiter='\t', skiprows=3, usecols=(1, 3))
        self.add_spectrum(data[:, 1], data[:, 0], group_name=group_name, name=name)

    def load_vamas(self, file: Path, group_name=None):
        obj = VAMAS(file)
        if group_name is None:
            group_name = file.name
        for b in obj.blocks:
            name = b.name
            x = np.array(b.binding_axis, dtype=np.float32)
            y = np.array(b.data[0], dtype=np.float32)
            self.add_spectrum(x, y, name=name, group_name=group_name)

    def load_specs2(self, file: Path, group_name=None):
        obj = SPECS(file)
        if group_name is None:
            group_name = file.name
        for i, g in enumerate(obj.groups):
            for r in g.regions:
                name = r.name
                x = r.binding_axis
                y = r.counts
                self.add_spectrum(x, y, name=name, group_name=group_name)

    def load_files(self, *files, group_name=None):
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
                self.load_specs2(f, group_name=group_name)

    def save_workspace(self, file: str):
        file = Path(file).with_suffix('.pkl')
        with file.open('wb') as f:
            pkl.dump(self.groups, f)
    
    def load_workspace(self, file: str):
        with open(file, 'rb') as f:
            self.groups.update(pkl.load(f))
    
    def save_spectra(self, save_dir: str, spectra):
        save_dir_path = Path(save_dir)
        for s in spectra:
            n = 0
            file_name = save_dir_path / f'{s.name}.dat'
            while file_name.exists():
                file_name = save_dir_path / f'{s.name}_{n}.dat'
                n += 1
            s.save_spectrum(file_name)
    
    def export_params(self, save_dir: str, spectra, xps_peak_like=True):
        save_dir_path = Path(save_dir)
        for s in spectra:
            n = 0
            file_name = save_dir_path / f'{s.name}.csv'
            while file_name.exists():
                file_name = save_dir_path / f'{s.name}_{n}.csv'
                n += 1
            s.export_params(file_name, xps_peak_like=xps_peak_like)

    def rename_group(self, group_name, new_group_name):
        self.groups[new_group_name] = self.groups.pop(group_name)
    
    def move_spectrum(self, spectrum_idx, group_name, new_group_name):
        self.groups[new_group_name].append(self.groups[group_name].pop(spectrum_idx))

    def merge_groups(self, new_group_name, other_groups):
        
        self.create_group(new_group_name)
        for group in other_groups:
            self.groups[new_group_name] += self.groups[group]
            self.delete_group(group)

    def delete_group(self, group_name):
        self.groups.pop(group_name)

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
    
    def predict(self, spectra):
        self.analyzer.predict(*spectra, pred_threshold=self.pred_threshold)
    
    def post_process(self, spectra):
        for s in spectra:
            self.analyzer.post_process(s)
            s.is_analyzed = True
            yield

    def set_charge_correction(self, spectra, current_line_energy=0, desired_line_energy=0):
        delta = desired_line_energy - current_line_energy
        for s in spectra:
            s.remove_charge_correction()
            s.set_charge_correction(delta)

    def add_line(self, region, loc=0, scale=1, const=1, gl_ratio=1, name=None, line=None):
        if line is not None:
            region.append(line)
        else:
            region.add_line(loc, scale, const, gl_ratio, name=name)

    def delete_line(self, region, line_idx):
        region.delete_line(line_idx)

    def change_line_parameter(
            self, line, param, value
    ):
        setattr(line, param, value)

    def create_new_region(self, start, end, spectrum, use_idxs=False, background_type='shirley'):
        if not use_idxs:
            start = self.find_closest_idx(start, spectrum)
            end = self.find_closest_idx(end, spectrum)
        
        region = spectrum.create_region(start, end)
        self.recalculate_background(region)
        return region

    def delete_region(self, region, spectrum):
        spectrum.delete_region(region)
    
    def change_region_parameter(self, region, spectrum, param, value):
        if param == 'start_point':
            start_idx = self.find_closest_idx(float(value), spectrum)
            end_idx = region.end_idx
            spectrum.change_region_range(region, start_idx, end_idx)
        elif param == 'end_point':
            start_idx = region.start_idx
            end_idx = self.find_closest_idx(float(value), spectrum)
            spectrum.change_region_range(region, start_idx, end_idx)
        elif param == 'background_type':
            region.background_type = value
        self.recalculate_background(region)

    def find_closest_idx(self, val, spectrum):
        # find the closest idx to the given value in the spectrum
        return (np.abs(spectrum.x  - val)).argmin()

    def recalculate_background(self, region):
        self.analyzer.calculate_region_background(region)

    def refit(
            self, region, use_norm_y=True, fixed_params=[], full_refit=False, tol=0.1, fit_alg='differential evolution',
            loc_tol=None
    ):
        self.analyzer.refit_region(region, use_norm_y, fixed_params, full_refit, tol, fit_alg, loc_tol=loc_tol)

    #TODO: build trend
    def build_trend(self, param, lines, x):
        params = self.analyzer.aggregate_params(param, lines)

    def __repr__(self):
        return f'Workspace(groups={self.groups})'
