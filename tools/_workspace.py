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
        self.spectra = []

    def set_prediction_threshold(self, threshold):
        self.pred_threshold = threshold
        for s in self.aggregate_spectra():
            if s.is_predicted:
                self.analyzer.restrict_mask(s, threshold=threshold)
    
    def check_survey(self, spectra, check=True):
        for s in spectra:
            s.is_survey = check

    def aggregate_spectra(self, files=[], groups=[], skip_survey=True):
        spectra = []
        if len(files) == 0 and len(groups) == 0:
            spectra.extend(self.spectra)
        else:
            for s in self.spectra:
                if s.file is None:
                    s.file = 'Unsorted'
                if s.group is None:
                    s.group = 'Unsorted'
                if s.file in files or s.group in groups:
                    spectra.append(s)

        if skip_survey:
            spectra = [s for s in spectra if not s.is_survey]

        return spectra

    def aggregate_unique_spectra(self, spectra=[], files=[], groups=[], skip_survey=True):
        if spectra and not files and not groups:
            if skip_survey:
                return [s for s in spectra if not s.is_survey]
            else:
                return spectra
        else:
            agg_spectra = self.aggregate_spectra(files, groups, skip_survey=skip_survey)
            for s in spectra:
                if s not in agg_spectra:
                    agg_spectra.append(s)
            return agg_spectra

    def add_spectrum(self, x, y, name=None, file=None, group=None):
        spectrum = Spectrum(x, y, name=name, file=file, group=group)
        self.spectra.append(spectrum)
    
    def delete_spectrum(self, spectrum):
        self.spectra.remove(spectrum)
    
    def rename_spectrum(self, spectrum, new_name):
        setattr(spectrum, 'name', new_name)
    
    def move_spectrum(self, spectrum, new_group_name):
        setattr(spectrum, 'group', new_group_name)
    
    def move_spectra(self, spectra, new_group_name):
        for s in spectra:
            self.move_spectrum(s, new_group_name)

    def rename_group(self, group_name, new_group_name):
        spectra = self.aggregate_spectra(groups=[group_name], skip_survey=False)
        self.move_spectra(spectra, new_group_name)

    def merge_groups(self, new_group_name, other_groups):
        spectra = self.aggregate_spectra(groups=other_groups, skip_survey=False)
        for s in spectra:
            self.move_spectrum(s, new_group_name)
    
    def delete_spectra(self, spectra=[], files=[], groups=[]):
        spectra = self.aggregate_unique_spectra(spectra, files, groups, skip_survey=False)
        for s in spectra:
            self.delete_spectrum(s)
    
    def paste_region(self, region_to_copy, other_spectra, keep_old_regions=False):
        init_min_coef, init_max_coef = region_to_copy.norm_coefs
        s_idx = region_to_copy.start_idx
        e_idx = region_to_copy.end_idx
        init_lines = []
        for l in region_to_copy.lines:
            loc = l.loc
            scale = l.scale
            norm_const = (l.const - init_min_coef) / (init_max_coef - init_min_coef)
            gl_ratio = l.gl_ratio
            init_lines.append((loc, scale, norm_const, gl_ratio))

        for s in other_spectra:
            min_coef, max_coef = s.norm_coefs
            if not keep_old_regions:
                s.regions = []
            new_r = s.create_region(s_idx, e_idx)
            self.recalculate_background(new_r)
            for l in init_lines:
                loc = l[0]
                scale = l[1]
                const = l[2] * (max_coef - min_coef) + min_coef
                gl_ratio = l[3]
                new_r.add_line(loc, scale, const, gl_ratio)
    
    def paste_spectra(self, spectrum, other_spectra, keep_old_regions=False):
        reg = spectrum.regions[0]
        self.paste_region(reg, other_spectra, keep_old_regions)
        for reg in spectrum.regions[1:]:
            self.paste_region(reg, other_spectra, keep_old_regions=True)

    def load_txt(self, file: Path):
        file_name = file.parent.name
        with open(file, 'r') as f:
            s = f.readline().split()
            if len(s) == 1: # casa format
                name = s[0]
                data = np.loadtxt(f, delimiter='\t', skiprows=3, usecols=(1, 3))
                self.add_spectrum(data[:, 1], data[:, 0], name=name, file=file_name)
            elif len(s) == 2: # numpy format
                name = file.stem
                data = np.loadtxt(f)
                self.add_spectrum(data[:, 0], data[:, 1], name=name, file=file_name)

    def load_vamas(self, file: Path):
        obj = VAMAS(file)
        file_name = file.name
        for b in obj.blocks:
            group = b.sample
            name = b.name
            x = np.array(b.binding_axis, dtype=np.float32)
            y = np.array(b.data[0], dtype=np.float32)
            self.add_spectrum(x, y, name=name, file=file_name, group=group)

    def load_specs2(self, file: Path):
        obj = SPECS(file)
        file_name = file.name
        for g in obj.groups:
            group = g.name
            for r in g.regions:
                name = r.name
                x = r.binding_axis
                y = r.counts
                self.add_spectrum(x, y, name=name, file=file_name, group=group)

    def load_files(self, *files):
        for f in files:
            if isinstance(f, str):
                f = Path(f)
            elif not isinstance(f, Path):
                print(f'File {f} must be str or Path')
                continue

            if f.suffix == '.txt' or f.suffix == '.dat' or f.suffix == '.csv':
                self.load_txt(f)
            elif f.suffix == '.vms':
                self.load_vamas(f)
            elif f.suffix == '.xml':
                self.load_specs2(f)

    def save_workspace(self, file: str):
        file = Path(file).with_suffix('.pkl')
        with file.open('wb') as f:
            pkl.dump(self.spectra, f)
    
    def load_workspace(self, file: str):
        self.spectra = []
        with open(file, 'rb') as f:
            obj = pkl.load(f)
        if isinstance(obj, list):
            self.spectra = obj
        elif isinstance(obj, dict): # compatibility with older versions
            for obj in obj.values():
                self.spectra.extend(obj)
                for s in obj:
                    s.file = 'Unsorted'
                    s.group = 'Unsorted'
    
    def save_spectra(self, save_dir: str, spectra):
        save_dir_path = Path(save_dir)
        for s in spectra:
            n = 1
            file_name = save_dir_path / f'{s.name}.dat'
            while file_name.exists():
                file_name = save_dir_path / f'{s.name}_{n}.dat'
                n += 1
            s.save_spectrum(file_name)
    
    def export_params(self, save_dir: str, spectra, xps_peak_like=True):
        save_dir_path = Path(save_dir)
        for s in spectra:
            n = 1
            file_name = save_dir_path / f'{s.name}.csv'
            while file_name.exists():
                file_name = save_dir_path / f'{s.name}_{n}.csv'
                n += 1
            s.export_params(file_name, xps_peak_like=xps_peak_like)

    def aggregate_and_export(self, file: str, spectra, xps_peak_like=True):
        params = []
        for s in spectra:
            params.extend(s.get_params(xps_peak_like=False))
        peak_nums = list(range(len(params)))
        params = np.array(params)
        params = np.insert(params, 0, peak_nums, axis=1)
        params[:, 0] += 1

        if xps_peak_like:
            header = ['Peak', 'Position (eV)', 'Area', 'FWHM (eV)', '%GL (%)']
            pattern = '{:<14}' * (len(header) - 1) + '{}'
            header = pattern.format(*header)
            params = params[:, [0, 1, 5, 2, 4]]
            fmt = ('%-13d', '%-13.3f', '%-13.3f', '%-13.3f', '%-13.3f')
        else:
            header = ['Peak', 'Position (eV)', 'Scale', 'Amplitude', '%GL (%)', 'Area', 'Height']
            pattern = '{:<14}' * (len(header) - 1) + '{}'
            header = pattern.format(*header)
            fmt = ('%-13d', '%-13.3f', '%-13.3f', '%-13.3f', '%-13.3f', '%-13.3f', '%-13.3f')

        np.savetxt(file, params, delimiter=' ', header=header, fmt=fmt, comments='')
    
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

    def add_line(self, region, loc=0, scale=1, const=1, gl_ratio=1, color=None, line=None):
        if line is not None:
            region.append(line)
        else:
            region.add_line(loc, scale, const, gl_ratio, color=color)

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
    def build_trend(self, lines, selected_option):
        if selected_option == "Relative area":
            areas = np.array([line.area for line in lines])
            sum_area = sum(areas)
            rel_areas = areas / sum_area
            return rel_areas
        elif selected_option == "Position":
            attr = "loc"
        elif selected_option == "Area":
            attr = "area"
        elif selected_option == "FWHM":
            attr = "fwhm"
        elif selected_option == "GL":
            attr = "gl"
        return self.analyzer.aggregate_params(attr, lines)

    def __repr__(self):
        return f'Workspace(groups={self.groups})'
