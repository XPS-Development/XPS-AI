import numpy as np
from numpy import trapz
from scipy.optimize import curve_fit, differential_evolution

import torch

from tools._spectra import Region, Line
from tools._tools import peak_sum

#TODO: проблемы с границами региона
#TODO: пересчет фона ширли, указание новых параметров для региона
class Analyzer():
    """Tool for spectra analyzing."""
    def __init__(self, model):
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
        mask = np.pad(mask, (1, 1), 'constant', constant_values=(0, 0))
        mask = np.diff(mask, append=0)
        return np.argwhere(np.abs(mask)).reshape(-1)

    def prepare_max_mask(self, max_mask, range=5):
        """Find idxs of medians in max_mask."""
        medians = []
        borders = self._find_borders(max_mask)
        for f, t in zip(borders[0::2], borders[1::2]):
            medians.append((t + f) // 2)
        return np.array(medians)

    def static_shirley(self, x, y, i_1, i_2, iters=8):
        """Calculate iterative Shirley background."""
        # i_1 < i_2
        background = np.zeros_like(x, dtype=np.float32)
        for _ in range(iters):
            y_adj = y - i_1 - background
            k = (i_2 - i_1) / trapz(y_adj, x)
            shirley_to_i = lambda i: k * trapz(y_adj[:i+1], x[:i+1])
            background = np.array([shirley_to_i(i) for i in range(len(x))])
        return background + i_1

    def calculate_background(self, x, y, background_type='shirley', **kwargs):
        """Calculate background."""
        if background_type == 'shirley':
            return self.static_shirley(x, y, **kwargs)
        else:
            raise ValueError(f'Unknown background type: {background_type}')

    def calculate_region_background(self, region, use_norm=False, **kwargs):
        """Calculate background for a region."""
        if use_norm:
            x, y = region.x, region.y_norm
            background_type = region.background_type
            min_coef, max_coef = region.norm_coefs
            i_1, i_2 = region.i_1, region.i_2
            # normalize i_1 and i_2
            i_1 = (i_1 - min_coef) / (max_coef - min_coef)
            i_2 = (i_2 - min_coef) / (max_coef - min_coef)
            background = self.calculate_background(x, y, background_type=background_type, i_1=i_1, i_2=i_2, **kwargs)
            # denormalize background
            region.background = background * (max_coef - min_coef) + min_coef
        else:
            x, y = region.x, region.y
            background_type = region.background_type
            i_1, i_2 = region.i_1, region.i_2
            region.background = self.calculate_background(x, y, background_type=background_type, i_1=i_1, i_2=i_2, **kwargs)
    
    # def prepare_fit_function(self, n_peaks):
    def _diff_ev_fit(self, x, y, bounds, maxiter=200):
        """
        Use differential_evolution to fit peaks to data.

        Parameters
        ----------
        x : numpy.array
            Energy values.
        y : numpy.array
            Intensity values.
        bounds : list of tuples
            Bounds for parameters in the format [(min, max), ...].
        maxiter : int, optional
            Maximum number of iterations in the optimization algorithm. Default is 200.

        Returns
        -------
        params : numpy.array
            Best fit parameters.
        """
        # lambda-function with L2 loss for differential_evolution alg
        f = lambda p: np.sum((y - peak_sum(len(bounds) // 4)(x, *p)) ** 2)
        res = differential_evolution(f, bounds=bounds, maxiter=maxiter)
        return res.x

    #TODO: filter too small peaks
    def init_params_by_locations(self, x, y, locations):
        # use normilized data
        """
        Initialize parameters for fitting peaks by locations.

        Parameters
        ----------
        x : numpy.array
            Energy values.
        y : numpy.array
            Intensity values.
        locations : list
            List of locations for peaks.

        Returns
        -------
        params : numpy.array
            Initial parameters for fitting.
        bounds : numpy.array
            Bounds for parameters in the shape (2, 4*n).
        """
        bounds = []
        for l in locations:
            bounds.append((l - 0.1, l + 0.1)) # loc
            bounds.append((0.1, 8)) # scale
            bounds.append((0, 5)) # const
            bounds.append((0, 1)) # gl_ratio

        params = self._diff_ev_fit(x, y, bounds)
        
        bounds = np.array(bounds)
        a, b = bounds.shape
        bounds = bounds.T.reshape(b, a)

        return params, bounds

    def fit(self, x, y, n_peaks, initial_params, bounds=None, active_background_fitting=False, initial_background=None, function=None):
        """Fitting line shapes for the spectrum"""
        if function is None:
            function = peak_sum(n_peaks)

        if active_background_fitting:
            pass
        else:
            background = initial_background
            y = y - background

            popt, _ = curve_fit(function, x, y, initial_params, bounds=bounds)
            return popt
    
    def _construct_bounds(self, val, tol, max_bound=None, min_bound=None):
        if min_bound is not None:
            b_min = val - tol if val - tol > min_bound else min_bound
        else:
            b_min = val - tol
        if max_bound is not None:
            b_max = val + tol if val + tol < max_bound else max_bound
        else:
            b_max = val + tol
        return (b_min, b_max)
    
    #TODO: refit fixed number of params
    def _refit(
            self,
            x,
            y,
            n_peaks,
            initial_params,
            background,
            tol=0.1,
            fixed_params=[],
            fit_alg='differential evolution'
    ):

        bounds = []
        for num, param in enumerate(initial_params):
            if num in fixed_params:
                bounds.append(
                    self._construct_bounds(param, param / 1000)
                )
            elif num % 4 == 0: # num % 4 != 0 for loc
                bounds.append(
                    self._construct_bounds(param, tol)
                )
            elif num % 4 == 3: # num % 4 != 3 for gl_ratio
                bounds.append(
                    self._construct_bounds(param, tol, min_bound=0, max_bound=1)
                )
            else: # num % 4 == 1 or num % 4 == 2 for scale or const
                bounds.append(
                    self._construct_bounds(param, tol, min_bound=0)
                )

        if fit_alg == 'differential evolution':
            y = y - background
            params = self._diff_ev_fit(x, y, bounds)
        elif fit_alg == 'least squares':
            bounds = np.array(bounds)
            a, b = bounds.shape
            bounds = bounds.T.reshape(b, a)
            params = self.fit(x, y, n_peaks, initial_params, bounds=bounds, initial_background=background)
        else:
            raise ValueError(f'Unknown fit_alg: {fit_alg}')

        return params

    def refit_region(
        self, region, use_norm_y=True, fixed_params=[], full_refit=False, tol=0.1, fit_alg='differential evolution'
    ):
        x = region.x
        n_peaks = len(region.lines)

        if full_refit:
            y = region.y_norm
            norm_coefs = region.norm_coefs
            background = (region.background  - norm_coefs[0]) / (norm_coefs[1] - norm_coefs[0])

            initial_params, bounds = self.init_params_by_locations(x, y, [l.loc for l in region.lines])
            params = self.fit(x, y, n_peaks, initial_params, bounds=bounds, initial_background=background)
            lines = self.params_to_lines(params, norm_coefs)
            region.lines = lines
            return

        if use_norm_y:
            y = region.y_norm
            norm_coefs = region.norm_coefs
            background = (region.background  - norm_coefs[0]) / (norm_coefs[1] - norm_coefs[0])
            initial_params = self.lines_to_params(region.lines, norm_coefs)
        else:
            y = region.y
            norm_coefs = (0, 1)
            background = region.background
            initial_params = self.lines_to_params(region.lines, norm_coefs)

        params = self._refit(
            x, y, n_peaks, initial_params, background, tol, fixed_params, fit_alg
        )

        lines = self.params_to_lines(params, norm_coefs)
        region.lines = lines
    
    def recalculate_idx(self, idx, array_1, array_2):
        if idx >= len(array_1):
            return len(array_2)
        val = array_1[idx]
        return (np.abs(array_2 - val)).argmin()

    def parse_masks_to_regions(self, x, y, x_int, y_smoothed, peak_mask, max_mask):
        # find region borders in peak_mask
        peak_borders = self._find_borders(peak_mask)
        # find maxima idxs in max_mask
        max_idxs = self.prepare_max_mask(max_mask)
        # recaculate idxs to non interpolated data
        peak_borders = np.array([self.recalculate_idx(idx, x_int, x) for idx in peak_borders])
        max_idxs = np.array([self.recalculate_idx(idx, x_int, x) for idx in max_idxs])

        # delete close borders
        connected_peak_borders = []
        for b in peak_borders:
            if len(connected_peak_borders) == 0:
                connected_peak_borders.append(b)
            elif b - connected_peak_borders[-1] < 5:
                connected_peak_borders.pop()
            else:
                connected_peak_borders.append(b)
        connected_peak_borders = np.array(connected_peak_borders)

        # split borders to regions
        for f, t in zip(peak_borders[0::2], peak_borders[1::2]):
            # choose max_idxs in region
            local_max_idxs = max_idxs[(max_idxs > f) & (max_idxs < t)]

            max_locations = x[local_max_idxs]
            reg_x = x[f:t]
            reg_y = y[f:t]
            reg_y_smoothed = y_smoothed[f:t]

            if local_max_idxs.size != 0:
                yield f, t, reg_x, reg_y, reg_y_smoothed[0], reg_y_smoothed[-1], max_locations
            else:
                continue
            
    def params_to_lines(self, params, norm_coefs=(0, 1)):
        lines = []
        min_value, max_value = norm_coefs
        for idx in range(len(params) // 4):
            l = Line(
                loc=params[4 * idx],
                scale=params[4 * idx + 1],
                const=(max_value - min_value)*params[4 * idx + 2],
                gl_ratio=params[4 * idx + 3]
            )
            lines.append(l)
        return lines

    def lines_to_params(self, lines, norm_coefs=(0, 1)):
        params = []
        min_value, max_value = norm_coefs
        for line in lines:
            params.extend([line.loc, line.scale, line.const / (max_value - min_value), line.gl_ratio])
        return np.array(params)
    
    def aggregate_params(self, param, lines):
        match param:
            case 'loc':
                return [l.loc for l in lines]
            case 'scale':
                return [l.scale for l in lines]
            case 'const':
                return [l.const for l in lines]
            case 'gl_ratio':
                return [l.gl_ratio for l in lines]
            case 'fwhm':
                return [l.fwhm for l in lines]
            case 'area':
                return [l.area for l in lines]
            case 'height':
                return [l.height for l in lines]
            case _:
                raise ValueError(f'Unknown param: {param}')

    #TODO: active shirley and static shirley
    def post_process(self, *spectra, active_background_fitting=False):
        for spectrum in spectra:
            # fit normalized spectrum
            x, y_norm = spectrum.x, spectrum.y_norm
            x_int, y_smoothed = spectrum.x_interpolated, spectrum.y_norm_smoothed
            min_value, max_value = spectrum.norm_coefs

            for start_idx, end_idx, reg_x, reg_y, i_1, i_2, max_locs in self.parse_masks_to_regions(
                x, y_norm, x_int, y_smoothed, spectrum.peak, spectrum.max
            ):  
                # create region and add background
                region = spectrum.create_region(start_idx, end_idx)
                reg_background = self.static_shirley(
                    reg_x, reg_y, i_1, i_2
                )
                region.background = reg_background * (max_value - min_value) + min_value

                # calculate initial params by max_locations from the mask
                init_params, bounds = self.init_params_by_locations(reg_x, reg_y - reg_background, max_locs)
                # accurate fitting
                params = self.fit(
                    reg_x, reg_y, len(max_locs), init_params, bounds, active_background_fitting, reg_background
                )

                # convert params to lines
                lines = self.params_to_lines(params, norm_coefs=spectrum.norm_coefs)
                region.lines = lines
