from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import seed, random, normal, choice
from matplotlib import pyplot as plt
from scipy import stats
from ruamel.yaml import YAML

from utils import interpolate, view_labeled_data, create_mask

def gauss(x, loc, scale):
    return 1/(scale * np.sqrt(2*np.pi)) * np.exp(-(x-loc)**2 / (2*scale**2))


def lorentz(x, loc, scale):
    return 1/(np.pi * scale * (1 + ((x - loc) / scale) ** 2))


def pseudo_voigt(x, loc, scale, c, r):
    return c * (r * gauss(x, loc, scale) + (1 - r) * lorentz(x, loc, scale))


def create_peak(x, loc, scale, c_peak, r, c_base=None):
    y = pseudo_voigt(x, loc, scale, c_peak, r)
    if c_base:
        y += c_base * stats.norm(loc=loc, scale=scale).cdf(x)
    return y


#TODO: docs
class SynthGenerator():
    """Tool for spectra generation."""
    def __init__(self) -> None:
        """Initialize the parameters from params.yaml."""
        yaml_loader = YAML(typ='safe', pure=True)
        params = yaml_loader.load(Path('model/params.yaml'))['synth_data']
        self.params = params
        seed(params['seed'])

        spectrum_len = self.params['spectrum_params']['len']
        self.x = np.arange(0, spectrum_len, dtype=np.float32)
        

    def gen_peak_params(self, peak_type) -> list:
        """Pars the parameters for the peak generation."""
        peak_params = (vals['val'] + random() * vals['var'] 
                    for vals in self.params['peak_types'][peak_type].values())
        return peak_params
    
    def peaks_to_gen(self) -> list:
        """Generate number of peaks for spectrum."""
        peaks_to_gen = []
        n_peaks = self.params['spectrum_params']['n_of_peaks']
        for p_type, n in n_peaks.items():
            from_n, to_n = map(int, n.split('-'))
            n_to_choice = np.arange(from_n, to_n+1, step=1)
            peaks = [p_type] * choice(n_to_choice)
            peaks_to_gen.extend(peaks)
        return peaks_to_gen

    def gen_noise(self):
        """Generate noise."""
        params = self.params['spectrum_params']['noise']
        noise_level = random() * params['val']
        noise_size = int(params['size'] + params['var'] * random())
        noise = normal(0, noise_level, (noise_size, ))
        _, noise = interpolate(np.arange(noise_size), noise, 256)
        return noise

    def gen_shakeup(self) -> float:
        """Generate parameter for background shake-up."""
        shakeup = random() * self.params['spectrum_params']['shakeup']
        return shakeup

    def gen_spectrum(self) -> tuple:
        """Generate labeled spectrum for model training."""
        x = self.x
        x_to_loc = np.arange(48, 206, dtype=np.float32)
        y = np.zeros_like(x)
        peak_mask = np.zeros_like(x)
        max_mask = np.zeros_like(x)
        peak_params = []

        peak_const = self.params['labeling']['peak_area']
        max_const = self.params['labeling']['max_area']

        p_list = self.peaks_to_gen()
        for p in p_list:
            scale, c, gl, back = self.gen_peak_params(p)
            loc = choice(x_to_loc)
            x_to_loc = x_to_loc[(x_to_loc < loc - max_const * 3) | (x_to_loc > loc + max_const * 3)]
            y += create_peak(x, loc, scale, c, gl, back)
            peak_mask += create_mask(x, from_x=loc-scale*peak_const, to_x=loc+scale*peak_const)
            max_mask += create_mask(x, from_x=loc-max_const, to_x=loc+max_const)
            peak_params.append((loc, scale, c, gl, back))

        y += self.gen_noise()
        y += self.gen_shakeup() * x

        y = (y - y.min()) / (y.max() - y.min())
        peak_mask[peak_mask > 0] = 1
        max_mask[max_mask > 0] = 1

        return x, y, peak_mask, max_mask, peak_params

    def gen_dataset(self, path):
        size = self.params['dataset_size']
        for i in range(size):
            x, y, peak_mask, max_mask, peak_params = self.gen_spectrum()
            data = pd.DataFrame(np.stack((y, peak_mask, max_mask), axis=1))
            data.to_csv(f'{path}/{i}.csv', header=False, index=False)
            

if __name__ == '__main__':
    gen = SynthGenerator()
    gen.gen_dataset('model/train/data/dataset')
