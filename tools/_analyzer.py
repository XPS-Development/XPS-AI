import numpy as np
from numpy import trapz
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import savgol_filter

import torch

from tools._tools import peak_sum


class Analyzer():
    """Tool for spectra analyzing."""
    def __init__(self, model):
        model.eval()
        self.model = model

    @torch.no_grad()
    def prepare_input(self, y):
        """Prepare input tensor for the model by normalizing and stacking the input data."""
        t_y = torch.tensor(y, dtype=torch.float32, device='cpu')
        t_y_log = torch.log(10*t_y + 1)
        t_y_log = (t_y_log - t_y_log.min())/(t_y_log.max() - t_y_log.min())
        t_inp = torch.stack((t_y, t_y_log), dim=0)
        return t_inp

    def batch_data(self, *spectra):
        """Convert multiple Spectrum objects into a batch of input tensors."""
        tensors = tuple(self.prepare_input(s.y_interpolated) for s in spectra)
        return torch.stack(tensors, dim=0)

    @torch.no_grad()
    def predict(self, *spectra, pred_threshold=0.5):
        """Add predicted masks to spectra."""
        inp = self.batch_data(*spectra)
        out = self.model(inp)

        peak = out[:, 0, :].detach().numpy()
        max = out[:, 1, :].detach().numpy()
        pred_peak_mask = (peak > pred_threshold)
        pred_max_mask = (max > pred_threshold)

        for i, s in enumerate(spectra):
            s.add_masks(pred_peak_mask[i], pred_max_mask[i])

    def _find_borders(self, mask):
        """Return idxs of borders in mask"""
        separators = np.diff(mask)
        idxs = np.argwhere(separators == True).reshape(-1)
        return idxs

    # def non_max

    def static_shirley(self, x, y, i_1, i_2, iters=8):
        """Calculate iterative Shirley background."""
        # i_1 < i_2
        #TODO: calc point with numpy vectors
        background = np.zeros_like(x, dtype=np.float32)
        for _ in range(iters):
            y_adj = y - background
            s_adj = trapz(y_adj, x)
            shirley_to_i = lambda i: i_1 + (i_2 - i_1) * trapz(y_adj[:i+1], x[:i+1]) / s_adj
            points = [shirley_to_i(i) for i in range(len(x))]
            background = points
        return points

    # def calc_background(self, spectrum, method='defaul_shirley'):
    #     x, y = spectrum.get_data()
    #     y_filtered = savgol_filter(y, 40, 3)

    #     if method == 'defaul_shirley':
    #         return self._default_shirley(x, y, )
    
    # # def _init_params():

    #TODO: initial guess params, function to finding initial params
    #TODO: filter too small peaks
    def init_params(self, x, y, n_peaks, locations):
        # lambda-function with L2 loss for differential_evolution alg
        g = lambda p: np.sqrt(np.sum((y - peak_sum(n_peaks)(x, *p)) ** 2))
        bounds = []
        for i in range(n_peaks):
            bounds.append((locations[2*i], locations[2*i + 1]))
            bounds.append((0.2, 5))
            bounds.append((0, 10))
            bounds.append((0, 1))
        res = differential_evolution(g, bounds, maxiter=1000)

        a, b = bounds.shape
        bounds = bounds.T.reshape(b, a)

        return res.x, bounds

    def fit(self, x, y, n_peaks, initial_params, bounds=None, active_background_fitting=False, initial_background=None):
        """Fitting line shapes for the spectrum"""

        if active_background_fitting:
            pass
        else:
            background = initial_background
            y = y - background

            popt, _ = curve_fit(peak_sum(n_peaks), x, y, initial_params, bounds=bounds)
            return popt
    
    def fit_by_mask(self, x, y, max_mask, active_background_fitting=False, initial_background=None):
        # find idxs of max regions in each peak region
        borders_idxs = self._find_borders(max_mask) # idxs in max_mask
        n_peaks = len(borders_idxs) // 2
        # find borders location
        max_borders = x[borders_idxs]
        for i in range(n_peaks):
            max_borders[2*i] -= 0.05
            max_borders[2*i + 1] += 0.05
        init_params, bounds = self.init_params(x, y, n_peaks, max_borders)

        params = self.fit(x, y, n_peaks, init_params, bounds, active_background_fitting, initial_background)
        return params

    def find_nearest_idxs(self, array_1, array_2, idxs):
        vals = array_1[idxs]
        new_idxs = []
        for val in vals:
            idx = (np.abs(array_2 - val)).argmin()
            new_idxs.append(idx)
        return new_idxs

    #TODO: recalc to non interpolated spectrum
    def parse_masks_to_regions(self, x, y, x_int, y_int, peak_mask, max_mask):
        y_smooth = savgol_filter(y, 40, 2)

        # find region borders in peak_mask
        peak_borders = self._find_borders(peak_mask)
        peak_borders_joint = [0]
        for b in peak_borders:
            if b - peak_borders_joint[-1] > 15:
                peak_borders_joint.append(b)
        
        # find max borders in max_mask
        max_borders = self._find_borders(max_mask) # idxs in max_mask

        for i in range(len(peak_borders_joint) - 1):
            f = peak_borders_joint[i]
            t = peak_borders_joint[i + 1]

            # check if region is empty then skip
            if not np.any(max_mask[f:t]):
                continue
            
            # recalculate masks borders to non interpolated data
            f, t = self.find_nearest_idxs(x_int, x, (f, t))

            n_peaks = len(borders_idxs) // 2
            # find borders location
            max_borders = x[borders_idxs]

            reg_y = y[f:t]
            reg_y_smooth = y_smooth[f:t]

            borders_idxs = self._find_borders(max_mask) # idxs in max_mask

            yield reg_x, reg_y, reg_max_mask, reg_y_smooth[0], reg_y_smooth[-1]

    #TODO: active shirley and static shirley
    #TODO: разбить функцию а то слишком большая
    def post_process(self, spectrum, active_background_fitting=False):
        # fit normalized spectrum
        x, y = spectrum.x, spectrum.y_norm
        x_int, y_int = spectrum.x_interpolated, spectrum.y_interpolated
        min_value, max_value = spectrum.norm_coefs

        for reg_x, reg_y, reg_max_mask, i_1, i_2 in self.parse_masks_to_regions(
            x, y, x_int, y_int, spectrum.peak_mask, spectrum.max_mask
        ):

            reg_background = self.static_shirley(
                x, y, i_1, i_2
            )

            params = self.fit_by_mask(
                reg_x, reg_y, reg_max_mask, 
                active_background_fitting=active_background_fitting,
                initial_background=reg_background
            )

            for idx in range(len(params) // 4):
                spectrum.add_line(
                    loc=params[4 * idx],
                    scale=params[4 * idx + 1],
                    const=(max_value - min_value)*params[4 * idx + 2],
                    gl_ratio=params[4 * idx + 3]
                )
            #TODO: recalculate background to non interpolated spectrum
            # t = (np.abs(x - reg_x[-1])).argmin()
            f = interp1d(x, reg_background, kind='linear')
            reg = spectrum.create_region(reg_x[0], reg_x[-1], i_1, i_2)
            reg.background = reg_background
            background[f:t] = (max_value - min_value) * reg_background + min_value

        # recalculate background
        background = (max_value - min_value) * background + min_value
        f = interp1d(x, background, kind='linear')
        spectrum.background = f(spectrum.x)
