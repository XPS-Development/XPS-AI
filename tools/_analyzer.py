import numpy as np
from numpy import trapz
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import savgol_filter

import torch
from tools._utils import peak_sum


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
        tensors = tuple(self.prepare_input(s.norm_y) for s in spectra)
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

    def _init_borders(self, peak_mask):
        peak_borders_idx = self._find_borders(peak_mask)
        b = [0]
        b.extend(peak_borders_idx.tolist())
        b.append(255)
        self.region_borders = b

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
    def fit(self, x, y, y_smooth, max_mask, initial_params=None, active_background_fitting=False):
        """Fitting line shapes for the spectrum"""

        # initial params for background
        i_1 = y_smooth[0]
        i_2 = y_smooth[-1]

        if active_background_fitting:
            pass
        else:
            background = self.static_shirley(x, y, i_1, i_2)
            y = y - background
            # find idxs of max regions in each peak region
            max_borders = self._find_borders(max_mask) # idxs in max_mask
            n_peaks = len(max_borders) // 2
            # find borders location
            max_borders = x[max_borders]

            if not initial_params:
                # lambda-function with L2 loss for differential_evolution alg
                g = lambda p: np.sqrt(np.sum((y - peak_sum(n_peaks)(x, *p)) ** 2))
                bounds = []
                for i in range(n_peaks):
                    bounds.append((max_borders[2*i] - 0.1, max_borders[2*i + 1] + 0.1))
                    bounds.append((0.2, 5))
                    bounds.append((0, 10))
                    bounds.append((0, 1))
                res = differential_evolution(g, bounds, maxiter=1000)
                initial_params = res.x

            bounds = np.array(bounds)
            a, b = bounds.shape
            bounds = bounds.T.reshape(b, a)
            popt, _ = curve_fit(peak_sum(n_peaks), x, y, initial_params, bounds=bounds)
            return popt, background

    #TODO: active shirley and static shirley
    #TODO: разбить функцию а то слишком большая
    def process(self, spectrum, active_background_fitting=False):
        # fit normalized spectrum
        x, y = spectrum.norm_x, spectrum.norm_y
        min_value, max_value = spectrum.norm_coefs
        y_smooth = savgol_filter(y, 40, 3)
        background = y_smooth.copy()
        peak_mask, max_mask = spectrum.get_masks()
        peak_borders_idx = self._find_borders(peak_mask)

        # join nearest borders if distance < 12
        borders_joint = [0]
        for b in peak_borders_idx:
            if b - borders_joint[-1] > 15:
                borders_joint.append(b)

        for i in range(len(borders_joint) - 1):
            f = borders_joint[i]
            t = borders_joint[i + 1]

            reg_x = x[f:t]
            reg_y = y[f:t]
            reg_y_smooth = y_smooth[f:t]
            reg_max_mask = max_mask[f:t]
            
            # check if region is empty then skip
            if not np.any(reg_max_mask):
                continue

            params, reg_background = self.fit(
                reg_x, reg_y, reg_y_smooth, reg_max_mask, 
                initial_params=None,
                active_background_fitting=active_background_fitting
            )

            for idx in range(len(params) // 4):
                spectrum.add_line(
                    loc=params[4 * idx],
                    scale=params[4 * idx + 1],
                    const=(max_value - min_value)*params[4 * idx + 2],
                    gl_ratio=params[4 * idx + 3]
                )

            background[f:t] = reg_background

        # recalculate background
        background = (max_value - min_value)* background + min_value
        f = interp1d(x, background, kind='linear')
        spectrum.background = f(spectrum.x)
