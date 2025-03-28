import numpy as np
from numpy import trapz
from scipy.optimize import curve_fit, differential_evolution

import onnxruntime as ort

from tools._spectra import Region, Line
from tools._tools import peak_sum

class Analyzer():
    """Tool for spectra analyzing."""
    def __init__(self, model):
        self.ort_session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

    def prepare_input(self, y):
        """Prepare input tensor for the model by normalizing and stacking the input data."""
        y_log = np.log(10*y + 1)
        y_log = (y_log - y_log.min())/(y_log.max() - y_log.min())
        y_inp = np.stack((y, y_log), axis=0, dtype=np.float32)
        return y_inp[np.newaxis, :, :]

    def _predict(self, spectrum):
        """Add predicted masks to spectra."""
        inp = {'l_x_': self.prepare_input(spectrum.y_interpolated)}
        out = self.ort_session.run(None, inp)[0]

        peak = out[0, 0, :]
        max = out[0, 1, :]

        spectrum.add_masks(peak, max, init_mask=True)
    
    def smooth_mask(self, mask, window_length=10):
        """Smooth mask using moving average."""
        return np.convolve(mask, np.ones(window_length)/window_length, mode='same')
    
    def restrict_mask(self, spectrum, threshold=0.5):
        """Restrict masks to the peaks with the highest probability."""

        peak = (self.smooth_mask(spectrum.init_peak) > threshold)
        max = (spectrum.init_max > threshold)
        spectrum.add_masks(peak, max)

    def predict(self, *spectra, pred_threshold=0.5):
        """Add predicted masks to spectra."""
        for s in spectra:
            self._predict(s)
            self.restrict_mask(s, threshold=pred_threshold)

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
    def init_params_by_locations(self, x, y, locations, maxiter=200):
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

        params = self._diff_ev_fit(x, y, bounds, maxiter=maxiter)
        
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
    
    def _construct_bounds(self, val, tol, max_bound=None, min_bound=None, abs_tol=False):
        if not abs_tol:
            tol = val * tol
        if min_bound is not None:
            b_min = val - tol if val - tol > min_bound else min_bound
        else:
            b_min = val - tol
        if max_bound is not None:
            b_max = val + tol if val + tol < max_bound else max_bound
        else:
            b_max = val + tol
        return (b_min, b_max)

    def _refit(
            self,
            x,
            y,
            n_peaks,
            initial_params,
            background,
            tol=0.1,
            fixed_params=[],
            fit_alg='differential evolution',
            abs_tol=False,
            loc_tol=1
    ):
        bounds = []
        for num, param in enumerate(initial_params):
            if num in fixed_params:
                bounds.append(
                    self._construct_bounds(param, 1e-6)
                )
            elif num % 4 == 0: # loc
                if loc_tol is not None:
                    bounds.append(
                        self._construct_bounds(param, loc_tol, abs_tol=True)
                    )
                else:
                    bounds.append(
                        self._construct_bounds(param, tol, abs_tol=True)
                    )
            elif num % 4 == 3: # gl_ratio
                bounds.append(
                    self._construct_bounds(param, tol, min_bound=0, max_bound=1, abs_tol=abs_tol)
                )
            else: # num % 4 == 1 or num % 4 == 2 for scale or const
                bounds.append(
                    self._construct_bounds(param, tol, min_bound=0, abs_tol=abs_tol)
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
        self, region, use_norm_y=True, fixed_params=[], full_refit=False, tol=0.1, fit_alg='differential evolution',
        loc_tol=None
    ):
        x = region.x
        n_peaks = len(region.lines)

        if full_refit:
            y = region.y_norm
            norm_coefs = region.norm_coefs
            background = (region.background  - norm_coefs[0]) / (norm_coefs[1] - norm_coefs[0])

            initial_params, bounds = self.init_params_by_locations(x, y-background, [l.loc for l in region.lines])
            params = self.fit(x, y, n_peaks, initial_params, bounds=bounds, initial_background=background)
            self.update_lines(params, region.lines, norm_coefs)
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
            x, y, n_peaks, initial_params, background, tol, fixed_params, fit_alg, loc_tol=loc_tol
        )
        self.update_lines(params, region.lines, norm_coefs)
    
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
        for f, t in zip(connected_peak_borders[0::2], connected_peak_borders[1::2]):
            # choose max_idxs in region
            local_max_idxs = max_idxs[(max_idxs > f) & (max_idxs < t)]

            if local_max_idxs.size != 0:    
                max_locations = x[local_max_idxs]
                reg_x = x[f:t]
                reg_y = y[f:t]
                reg_y_smoothed = y_smoothed[f:t]
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

    def update_lines(self, params, lines, norm_coefs=(0, 1)):
        min_value, max_value = norm_coefs
        for idx in range(len(params) // 4):
            lines[idx].loc = params[4 * idx]
            lines[idx].scale = params[4 * idx + 1]
            lines[idx].const = (max_value - min_value)*params[4 * idx + 2]
            lines[idx].gl_ratio = params[4 * idx + 3]

    def lines_to_params(self, lines, norm_coefs=(0, 1)):
        params = []
        min_value, max_value = norm_coefs
        for line in lines:
            params.extend([line.loc, line.scale, line.const / (max_value - min_value), line.gl_ratio])
        return np.array(params)
    
    def aggregate_params(self, param, lines):
        return [getattr(l, param) for l in lines]

    #TODO: active shirley and static shirley
    def post_process(self, spectrum, active_background_fitting=False, ):
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
            init_params, bounds = self.init_params_by_locations(reg_x, reg_y - reg_background, max_locs, maxiter=200)
            # accurate fitting
            params = self.fit(
                reg_x, reg_y, len(max_locs), init_params, bounds, active_background_fitting, reg_background
            )

            # convert params to lines
            lines = self.params_to_lines(params, norm_coefs=spectrum.norm_coefs)
            region.lines = lines

