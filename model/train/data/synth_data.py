from collections import namedtuple

import numpy as np
from numpy.random import random, normal, choice
from scipy import stats

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


Peak = namedtuple('Peak', ['loc', 'scale', 'c_peak', 'r', 'c_base'])

#TODO: docs
class SynthGenerator():
    """Tool for spectra generation"""
    def __init__(self, peak_const=3, max_const=0.3) -> None:
        self.x = np.arange(0, 256, dtype=np.float32)
        self.peak_const = peak_const
        self.max_const = max_const
    
    def gen_peaks(self, num):
        peaks = []

        x = self.x
        scale = 6 * random() + 10
        b1 = (np.abs(x - x[0] - self.peak_const * scale)).argmin()
        b2 = (np.abs(x - x[-1] + self.peak_const * scale)).argmin()
        loc = choice(x[b1:b2])
        c_peak = 1
        r = random() / 10 + 0.7
        c_base = random() * 0.003

        mult = -1 if loc >= len(self.x) / 2 else 1
        peaks.append(Peak(loc, scale, c_peak, r, c_base))

        for _ in range(num - 1):
            c_peak *= random() * 0.2 + 1
            scale *= random() * 0.5 + 0.5
            r = random() / 20 + 0.9
            loc += mult * (random() * 10 + 25)
            peaks.append(Peak(loc, scale, c_peak, r, c_base))

        return peaks
    
    def gen_noise(self, noise_level):
        noise_level = random() * noise_level
        noise_size = int(120 + 20*random())
        noise = normal(0, noise_level, (noise_size,))
        _, noise = interpolate(np.arange(noise_size), noise, len(self.x))
        return noise
    
    def gen_shakeup(self):
        c_shakeup = random() / 5e4
        return -c_shakeup * self.x

    #TODO: complete
    def gen_satellites(self, num, main_peaks):
        loc = np.mean([peak.loc for peak in main_peaks])
        satellites = []

        mult = -1 if loc >= len(self.x) / 2 else 1
        for peak in main_peaks:
            for _ in range(num):
                loc_adj = mult * (random() * 50 + 40)
                scale_mult = random() * 0.2 + 0.3
                c_mult = random() * 0.02 + 0.03
                r = 1
                satellites.append(Peak(loc + loc_adj, peak.scale * scale_mult, peak.c_peak*c_mult, r, None))
        return satellites
    
    def gen_spectrum(self, num_of_peaks=2, num_of_satellites=0, noise=True, shakeup=True):
        x = self.x
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

        n = 0
        y_draft = y.copy()

        for peak in peaks:
            loc, scale, c_peak, r, c_base = peak
            y_draft += create_peak(self.x, loc, scale, c_peak, r, c_base)
            if (loc - (self.peak_const + 1)*scale) < self.x[0] or (loc + (self.peak_const + 1)*scale) > self.x[-1] or (n > 1 and abs(peak.loc - last_peak.loc) < 12):
                continue
            n += 1
            y += create_peak(self.x, loc, scale, c_peak, r, c_base)
            peak_mask += create_mask(x, from_x=loc-self.peak_const*scale, to_x=loc+self.peak_const*scale)
            max_mask += create_mask(x, from_x=loc-self.max_const*scale, to_x=loc+self.max_const*scale)
            last_peak = peak
        
        y = (y - y.min()) / (y_draft.max() - y.min())
        peak_mask[peak_mask > 0] = 1
        max_mask[max_mask > 0] = 1
        
        return y, peak_mask, max_mask

#TODO: delete
if __name__ == '__main__':
    import pandas as pd
    from matplotlib import pyplot as plt
    x = np.arange(0, 256, dtype=np.float32)
    g = SynthGenerator()
    for i in range(3500, 4000):
        y, peak_mask, max_mask = g.gen_spectrum((1, 2), noise=1/10e2, shakeup=True, num_of_satellites=1)
        # view_labeled_data(x, y, (peak_mask, max_mask))
        # plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 16
        plt.plot(x, y, 'k')
        plt.text(0, 0.97, '(b)', fontsize=16)
        plt.ylabel('Normalized intensity', fontsize=20)
        plt.xlabel('Binding energy (eV)', fontsize=20)
        min_to_fill = y.min()
        plt.fill_between(x, y, min_to_fill, where=peak_mask > 0, color='b', alpha=0.2, label='Область пика')
        plt.fill_between(x, y, min_to_fill, where=max_mask > 0, color='r', alpha=0.8, label='Область максимума пика')
        plt.savefig('synth_2.png', format='png', bbox_inches='tight', dpi=1200)
        plt.show()
                # array = np.stack((y, peak_mask, max_mask), axis=1)
        # df = pd.DataFrame(array)
        # df.to_csv(f'data/data_to_train/synth_{i}.csv', index=False, header=False)
    # for i in range(0, 400):
    #     y, peak_mask, max_mask = g.gen_spectrum((1, 2), noise=1/8e2, shakeup=True, num_of_satellites=(0, 1))
    #     array = np.stack((y, peak_mask, max_mask), axis=1)
    #     df = pd.DataFrame(array)
    #     df.to_csv(f'data/data_to_test/synth_{i}.csv', index=False, header=False)
