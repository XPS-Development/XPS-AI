#TODO: rename module
from collections import namedtuple
from time import time

from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
from numpy import trapz
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader

from model import XPSModel
from dataset import XPSDataset
from utils import view_labeled_data, interpolate

PATH_TO_VAL = 'data/data_to_val'
TRESHOLD = 0.5

def find_borders(mask):
    """Return idxs of borders in mask"""
    separators = np.diff(mask)
    idxs = np.argwhere(separators == True).reshape(-1)
    return idxs


def gauss(x, loc, scale):
    return 1/(scale * np.sqrt(2*np.pi)) * np.exp(-(x-loc)**2 / (2*scale**2))

def lorentz(x, loc, scale):
    return 1/(np.pi * scale * (1 + ((x - loc) / scale) ** 2))

def pseudo_voigt(x, loc, scale, c, r):
    return c * (r * gauss(x, loc, scale) + (1 - r) * lorentz(x, loc, scale))

def peak_sum(n):
    def f(x, *p):
        y = np.zeros_like(x, dtype=np.float32)
        for i in range(n):
            y += pseudo_voigt(x, p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3])
        return y
    return f


def calc_shirley_background(x, y, i_1, i_2, iters=8):
        """Calculate iterative Shirley background/
        
        Parameters:
        ---
        x : array_like
            Energies
        y : array_like
            Intestites
        i_1 : float
        i_2 : float
            Intesities of background
        """
        # i_1 > i_2
        background = np.zeros_like(x, dtype=np.float32)
        for _ in range(iters):
            y_adj = y - background
            s_adj = trapz(y_adj, x)
            shirley_to_i = lambda i: i_2 + (i_1 - i_2) * trapz(y_adj[:i+1], x[:i+1]) / s_adj
            points = [shirley_to_i(i) for i in range(len(x))]
            background = points
        return points





Area = namedtuple('Area', ['x', 'background', 'n_peaks', 'params'])


class Spectrum():
    """"""
    def __init__(self, energies, intensities) -> None:
        self._x = energies
        self._y = intensities

        #preproc
        self.x, y = interpolate(self._x, self._y)
        min_value = y.min()
        max_value = y.max()
        y = (y - min_value)/(max_value - min_value)
        self.norm_coefs = (min_value, max_value)
        self.y = y
        self._sort()

        self.areas = []
    
    @classmethod
    def load_from_file(cls, f, delimiter=None):
        if delimiter:
            data = np.loadtxt(f, delimiter=delimiter)
        else:
            data = np.loadtxt(f)
        
        return cls(data[:, 0], data[:, 1])

    def _sort(self):
        if self.x[0] > self.x[-1]:
            # .copy to prevent negative stride error in torch
            self.x = self.x[::-1].copy()
            self.y = self.y[::-1].copy()
            
    def get_data(self):
        return self.x, self.y
    
    def get_init_data(self):
        return self._x, self._y

    def add_masks(self, peak_mask, max_mask):
        self.peak = peak_mask
        self.max = max_mask
    
    def get_masks(self):
        return self.peak, self.max


class Analyzer():
    """Initialize tool for spectra analyzing"""
    def __init__(self, model, pred_threshold=0.5):
        self.model = model
        self.pred_threshold = pred_threshold
    
    def predict(self, *spectra):
        """Add predicted masks to spectra"""
        for s in spectra:
            x, y = s.get_data()
            t_y = torch.tensor(y, dtype=torch.float32, device='cpu').view(1, 1, -1)
            out = self.model(t_y)
            peak = out[0].view(-1).detach().numpy()
            max = out[1].view(-1).detach().numpy()
            pred_peak_mask = (peak > self.pred_threshold)
            pred_max_mask = (max > self.pred_threshold)
            s.add_masks(pred_peak_mask, pred_max_mask)
    
    def find_borders(self, mask):
        """Return idxs of borders in mask"""
        separators = np.diff(mask)
        idxs = np.argwhere(separators == True).reshape(-1)
        return idxs
    
    def calc_shirley_background(self, x, y, i_1, i_2, iters=8):
        """Calculate iterative Shirley background/
        
        Parameters:
        ---
        x : array_like
            Energies
        y : array_like
            Intestites
        i_1 : float
        i_2 : float
            Intesities of background
        """
        # i_1 < i_2
        background = np.zeros_like(x, dtype=np.float32)
        for _ in range(iters):
            y_adj = y - background
            s_adj = trapz(y_adj, x)
            shirley_to_i = lambda i: i_1 + (i_2 - i_1) * trapz(y_adj[:i+1], x[:i+1]) / s_adj
            points = [shirley_to_i(i) for i in range(len(x))]
            background = points
        return points
    
    #TODO: initial guess params
    def fit(self, x, y, max_mask, initial_guess=None):
        """"""

        # find idxs of max regions in each peak region
        max_borders = self.find_borders(max_mask) # idxs in max_mask
        n_peaks = len(max_borders) // 2
        # find x-borders from max_mask
        max_borders = x[max_borders]
        if not max_borders.any():
            return 0, None
        g = lambda p, *args: np.sqrt(np.sum((y - peak_sum(n_peaks)(x, *p, *args)) ** 2))
        bounds = []
        for i in range(n_peaks):
            bounds.append((max_borders[2*i] - 0.1, max_borders[2*i + 1] + 0.1))
            bounds.append((0.4, 1.5))
            bounds.append((0.1, 1.5))
            bounds.append((0, 1))
        res = differential_evolution(g, bounds, maxiter=2000)

        # create initial values for each gaussian
        p0 = []
        for i in range(n_peaks):
            # values: loc=mean(borders), scale=1, c=0.5
            # p0.extend([(max_borders[2*i] + max_borders[2*i + 1])/2, 0.25, 0.8])
            p0.extend(res.x)

        popt, _ = curve_fit(peak_sum(n_peaks), x, y, p0)
        return n_peaks, popt
    
    def processing(self, spectrum):
        x, y = spectrum.get_data()
        y_filtered = savgol_filter(y, 40, 3)
        
        peak_mask, max_mask = spectrum.get_masks()
        peak_borders_idx = self.find_borders(peak_mask)
        #TODO:
        b = peak_borders_idx.tolist()
        b.insert(0, 0)
        b.append(255)
        i = []
        for n in range(len(b) // 2):
            f = b[2 * n]
            t = b[2 * n + 1]
            i.append(np.mean(y_filtered[f:t]))

        for n in range(len(peak_borders_idx) // 2):
            f = peak_borders_idx[2 * n]
            t = peak_borders_idx[2 * n + 1] + 1
            l = t - f
            # skip too small regions
            if l < 20:
                continue

            curr_y = y[f - 20:t + 20]
            curr_x = x[f - 20:t + 20]
            # curr_y_filtered = y_filtered[f:t]
            curr_max_mask = max_mask[f - 20:t + 20]

            background = self.calc_shirley_background(curr_x, curr_y, i[n], i[n + 1])


            n_peaks, params = self.fit(curr_x, curr_y - background, curr_max_mask)
            spectrum.areas.append(Area(curr_x, background, n_peaks, params))

            if not n_peaks:
                continue

            S = []
            for j in range(n_peaks):
                s = trapz(pseudo_voigt(x, params[4*j], params[4*j+1], params[4*j+2], params[4*j+3]), x)
                S.append(s)
            print('Areas: ', S)
            # print('delta E: ', abs(params[0] - params[4]))


if __name__ == '__main__':
    from utils import load_data_from_casa

    array = pd.read_table('data/short_ag_cl_val/Cl2p_1.dat', sep='\s\s+', decimal=',').iloc[:, :2].to_numpy()

    # array = load_data_from_casa(f'data/full_ag_cl_val/Cl2p_5.txt')
    # array = array[:, 2:4]

    s = Spectrum(array[:, 0], array[:, 1])

    model = XPSModel()
    model.load_state_dict(torch.load('weights/v1.0_best_iou'))#, map_location=torch.device('cpu')))
    model.eval()
    a = Analyzer(model)
    a.predict(s)
    # view_labeled_data(x, y, s.get_masks())


    x, y = s.get_data()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.ylabel('Интенсивность', fontsize=20)
    plt.xlabel('Энергия связи, эВ', fontsize=20)

    # peak_mask, max_mask = s.get_masks()
    # plt.plot(x, y, 'k')
    # min_to_fill = y.min()
    # plt.fill_between(x, y, min_to_fill, where=peak_mask > 0, color='b', alpha=0.2, label='Область пика')
    # plt.fill_between(x, y, min_to_fill, where=max_mask > 0, color='r', alpha=0.8, label='Область максимума пика')

    a.processing(s)
    plt.plot(x, y, color='k', linewidth=1)
    for area in s.areas:
        p = area.params
        plt.plot(area.x, area.background, color='k', linewidth=1)
        if area.n_peaks:
            for i in range(area.n_peaks):
                plt.plot(area.x, pseudo_voigt(area.x, p[4*i], p[4*i+1], p[4*i+2], p[4*i+3]) + area.background)
            plt.plot(area.x, peak_sum(area.n_peaks)(area.x, *p) + area.background)
    plt.savefig('cl.png', format='png', bbox_inches='tight', dpi=1200)
    plt.show()
