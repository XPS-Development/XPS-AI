from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import random, normal, choice
from scipy import stats
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset

from tools import interpolate


class XPSDataset(Dataset):
    def __init__(self, path):
        super().__init__()

        self.data = []

        for f in Path(path).iterdir():
            array = np.loadtxt(f, delimiter=',')
            self.data.append(array)
        
    def __getitem__(self, index):
        array = self.data[index]

        x = array[:, 0]
        x = torch.tensor(x, dtype=torch.float32)

        x_log = array[:, 1]
        x_log = torch.tensor(x_log, dtype=torch.float32)

        peak_mask = array[:, 2]
        peak_mask = torch.tensor(peak_mask, dtype=torch.float32)

        max_mask = array[:, 3]
        max_mask = torch.tensor(max_mask, dtype=torch.float32)
        
        return torch.stack((x, x_log), dim=0), torch.stack((peak_mask, max_mask), dim=0)
    
    def __len__(self):
        return len(self.data)        


def gauss(x, loc, scale):
    return np.exp(-(x-loc)**2 / (2*scale**2))


def lorentz(x, loc, scale):
    return np.exp(-(x-loc)**2 / (2*scale**2))


def pseudo_voigt(x, loc, scale, c, r):
    return c * (r * gauss(x, loc, scale) + (1 - r) * lorentz(x, loc, scale))


def create_peak(x, loc, scale, c_peak, r, c_base=None):
    y = pseudo_voigt(x, loc, scale, c_peak, r)
    if c_base:
        y += c_base * stats.norm(loc=loc, scale=scale).cdf(x)
    return y


def create_mask(x, from_x, to_x):  
    zeros = np.zeros_like(x)
    zeros[(x > from_x) & (x < to_x)] = 1
    return zeros


def view_labeled_data(
        x, 
        y, 
        masks=(),
        mask_params=({'color': 'b', 'alpha': 0.2}, {'color': 'r'}),
        peak_params=(),
        debug=False
):

    plt.plot(x, y, 'k')

    min_to_fill = y.min()
    for mask, mask_param in zip(masks, mask_params):
        plt.fill_between(x, y, min_to_fill, where=mask > 0, **mask_param)
        
    for p in peak_params:
        if debug:
            print(p)
        plt.plot(x, create_peak(x, *p))

    plt.show()


#TODO: docs
class SynthGenerator():
    """Tool for spectra generation."""
    def __init__(self, params, seed=None) -> None:
        """Initialize the parameters from params.yaml."""

        self.params = params

        if seed:
            np.random.seed(seed)

        spectrum_len = self.params['spectrum_params']['len']
        self.x = np.arange(0, spectrum_len, dtype=np.float32)

    def gen_peak_params(self, peak_type) -> list:
        """Pars the parameters for the peak generation."""
        peak_params = (vals['val'] + random() * vals['var'] 
                    for vals in self.params['peak_types'][peak_type].values())
        return peak_params
    
    def peaks_to_gen(self, without_satellites=False) -> list:
        """Generate number of peaks for spectrum."""
        peaks_to_gen = []
        n_peaks = self.params['spectrum_params']['n_of_peaks']
        for p_type, n in n_peaks.items():
            if without_satellites and p_type == 'satellite':
                continue
            from_n, to_n = map(int, n.split('-'))
            n_to_choice = np.arange(from_n, to_n+1, step=1)
            peaks = [p_type] * choice(n_to_choice)
            peaks_to_gen.extend(peaks)
        return peaks_to_gen

    def gen_noise(self, debug=False):
        """Generate noise."""
        params = self.params['spectrum_params']['noise']
        noise_level = random() * params['val']
        noise_size = int(params['size'] + params['var'] * random())
        noise = normal(0, noise_level, (noise_size, ))
        if debug:
            print(f'noise_level: {noise_level},\nnoise_size: {noise_size}')
        _, noise = interpolate(np.arange(noise_size), noise, 256)
        return noise, noise_level

    def gen_shakeup(self, x):
        x = np.arange(256)
        """Generate parameter for background shake-up."""
        mult = choice([-1, 1])
        shakeup_coef = random() * self.params['spectrum_params']['shakeup']
        shakeup_start = choice(x)

        mask = mult * x < mult * shakeup_start
        x = x - shakeup_start
        x[mask] = 0

        return mult * shakeup_coef * x

    def gen_spectrum(self, debug=False) -> tuple:
        """Generate labeled spectrum for model training."""
        x = self.x
        x_to_loc_overlapped = np.arange(48, 206, dtype=np.float32)
        x_to_loc = np.arange(16, 240, dtype=np.float32)
        y = np.zeros_like(x)
        peak_mask = np.zeros_like(x)
        max_mask = np.zeros_like(x)
        peak_params = []

        peak_const = self.params['labeling']['peak_area']
        max_const = self.params['labeling']['max_area']

        noise, noise_level = self.gen_noise(debug)
        y += noise
        if noise_level > 0.0115:
            p_list = self.peaks_to_gen(without_satellites=True)
        else:
            p_list = self.peaks_to_gen()
        
        for p in p_list:
            scale, c, gl, back, overlapping = self.gen_peak_params(p)

            if overlapping:
                loc = choice(x_to_loc_overlapped)
            elif len(x_to_loc) == 0:
                continue
            else:
                loc = choice(x_to_loc)
            
            x_to_loc_overlapped = x_to_loc_overlapped[(x_to_loc_overlapped < loc - max_const * 5) | (x_to_loc_overlapped > loc + max_const * 5)]
            x_to_loc = x_to_loc[(x_to_loc < loc - 2.5 * scale) | (x_to_loc > loc + 2.5 * scale)]

            y += create_peak(x, loc, scale, c, gl, back)
            peak_mask += create_mask(x, from_x=loc-scale*peak_const, to_x=loc+scale*peak_const)
            max_mask += create_mask(x, from_x=loc-max_const, to_x=loc+max_const)
            peak_params.append((loc, scale, c, gl, back))

        y += self.gen_shakeup(x)
        y = (y - y.min()) / (y.max() - y.min())

        y_log = np.log(10*y+1)
        y_log = (y_log - y_log.min()) / (y_log.max() - y_log.min())
        
        peak_mask[peak_mask > 0] = 1
        max_mask[max_mask > 0] = 1

        masks = (peak_mask, max_mask)

        if debug:
            view_labeled_data(x, y, masks, peak_params=peak_params, debug=debug)
            plt.plot(x, y_log)
            plt.show()

        return x, y, y_log, masks, peak_params

    def gen_dataset(self, path):
        p = Path(path)
        size = self.params['dataset_size']
        p.mkdir(exist_ok=True, parents=True)
        for i in range(size):
            x, y, y_log, (peak_mask, max_mask), peak_params = self.gen_spectrum()
            data = pd.DataFrame(np.stack((y, y_log, peak_mask, max_mask), axis=1))
            p_i = p.joinpath(f'{i}.csv')
            data.to_csv(p_i, header=False, index=False)
