from collections import namedtuple

import numpy as np
from numpy.random import random, normal, choice
from scipy import stats

from utils import interpolate, view_labeled_data, create_mask


def create_gauss(x, loc, scale, c_peak, c_base=None):
    gauss = stats.norm(loc=loc, scale=scale)
    y = c_peak * gauss.pdf(x)
    if c_base:
        y += c_base * gauss.cdf(x)
    return y


Peak = namedtuple('Peak', ['loc', 'scale', 'c_peak', 'c_base'])

#TODO: docs
class SynthGenerator():
    """Tool for spectra generation"""
    def __init__(self, peak_const=5, max_const=0.3) -> None:
        self.x = np.arange(0, 256, dtype=np.float32)
        self.peak_const = peak_const
        self.max_const = max_const
    
    def gen_peaks(self, num):
        peaks = []

        x = self.x
        scale = 5 * random() + 10
        b1 = (np.abs(x - x[0] - self.peak_const * scale)).argmin()
        b2 = (np.abs(x - x[-1] + self.peak_const * scale)).argmin()
        loc = choice(x[b1:b2])
        c_peak = 1
        c_base = random() * 0.003

        peaks.append(Peak(loc, scale, c_peak, c_base))
        mult = -1 if loc >= len(self.x) / 2 else 1

        for _ in range(num - 1):
            c_peak *= random() * 0.25 + 1
            scale *= random() * 0.5 + 0.5
            loc += mult * (random() * 10 + 20)
            peaks.append(Peak(loc, scale, c_peak, c_base))

        return peaks
    
    def gen_noise(self, noise_level):
        noise_level = random() * noise_level
        noise_size = int(120 + 20*random())
        noise = normal(0, noise_level, (noise_size,))
        _, noise = interpolate(np.arange(noise_size), noise, len(x))
        return noise
    
    def gen_shakeup(self):
        c_shakeup = random() / 5e4
        return -c_shakeup * self.x

    #TODO: complete
    def gen_satellites(self, num, main_peaks):
        satellites = []
        for _ in range(num):
            mult = choice([1, -1])
            loc_adj = mult * (random() * 50 + 20)
            scale_mult = random() * 0.3 + 0.5
            c_mult = random() * 0.03 + 0.05
            for peak in main_peaks:
                satellites.append(Peak(peak.loc + loc_adj, peak.scale * scale_mult, peak.c_peak*c_mult, None))
        return satellites
    
    def gen_spectrum(self, num_of_peaks=2, noise=True, shakeup=True, num_of_satellites=0):
        y = np.zeros_like(self.x)
        peak_mask = np.zeros_like(self.x)
        max_mask = np.zeros_like(self.x)

        if noise:
            y += self.gen_noise(noise)
        
        if shakeup:
            y += self.gen_shakeup()

        if type(num_of_peaks) is tuple:
            num_of_peaks = choice(range(num_of_peaks[0], num_of_peaks[-1] + 1))
            peaks = self.gen_peaks(num_of_peaks)
        else:
            peaks = self.gen_peaks(num_of_peaks)

        if type(num_of_satellites) is tuple:
            num_of_satellites = choice(range(num_of_satellites[0], num_of_satellites[-1] + 1))
            satellites = self.gen_satellites(num_of_satellites, peaks)
            peaks.extend(satellites)
        else:
            satellites = self.gen_satellites(num_of_satellites, peaks)
            peaks.extend(satellites)

        for peak in peaks:
            loc, scale, c_peak, c_base = peak
            if loc - self.peak_const*scale < self.x[0] or loc + self.peak_const*scale > self.x[-1]:
                continue
            y += create_gauss(self.x, loc, scale, c_peak, c_base)
            peak_mask += create_mask(x, from_x=loc-self.peak_const*scale, to_x=loc+self.peak_const*scale)
            max_mask += create_mask(x, from_x=loc-self.max_const*scale, to_x=loc+self.max_const*scale)
        
        y = (y - y.min()) / (y.max() - y.min())
        peak_mask[peak_mask > 0] = 1
        max_mask[max_mask > 0] = 1

        return y, peak_mask, max_mask


if __name__ == '__main__':
    import pandas as pd
    x = np.arange(0, 256, dtype=np.float32)
    g = SynthGenerator()
    for i in range(0, 2000):
        y, peak_mask, max_mask = g.gen_spectrum((1, 2), noise=1/6e2, shakeup=True, num_of_satellites=(0, 1))
        view_labeled_data(x, y, (peak_mask, max_mask))
    #     array = np.stack((y, peak_mask, max_mask), axis=1)
    #     df = pd.DataFrame(array)
    #     df.to_csv(f'data/data_to_train/synth_{i}.csv', index=False, header=False)
    # for i in range(0, 200):
    #     y, peak_mask, max_mask = g.gen_spectrum((1, 2), noise=1/6e2, shakeup=True, num_of_satellites=(0, 1))
    #     array = np.stack((y, peak_mask, max_mask), axis=1)
    #     df = pd.DataFrame(array)
    #     df.to_csv(f'data/data_to_val/synth_{i}.csv', index=False, header=False)
