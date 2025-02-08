import numpy as np
from scipy import stats
import torch

from tools import Spectrum, Region, Line, Analyzer
from model.models.model_deeper import XPSModel

MODEL = XPSModel()
MODEL.load_state_dict(torch.load('model/trained_models/model.pt', map_location=torch.device('cpu'), weights_only=True))
MODEL.eval()

ANALYZER = Analyzer(MODEL)


def two_arrays_equal(a, b):
    return np.all(a == b)


def two_arrays_almost_equal(a, b, tol=1e-5):
    return np.all(np.abs(a - b) < tol)


class SpectrumCase:
    def load_one_norm_like_peak(self):
        x = np.linspace(-5, 5, 100)
        p_1 = stats.norm(loc=0, scale=1)
        y = p_1.pdf(x)

        params = np.array([0, np.sqrt(2*np.log(2)), 1, 1])

        return x, y, params
    
    def load_two_norm_like_peaks(self):
        x = np.linspace(-20, 20, 100)
        p_1 = stats.norm(loc=-2, scale=1)
        p_2 = stats.norm(loc=4, scale=2)
        y = p_1.pdf(x) + p_2.pdf(x)
        params = np.array([-2, np.sqrt(2*np.log(2)), 1, 1, 4, 2*np.sqrt(2*np.log(2)), 1, 1])

        return x, y, params

    def load_one_peak(self):
        x = np.linspace(-5, 5, 100)
        p_1 = stats.norm(loc=0, scale=1)
        y = 10 * p_1.pdf(x)

        params = np.array([0, np.sqrt(2*np.log(2)), 10, 1])

        return x, y, params

    def load_one_peak_with_background(self):
        x = np.linspace(-5, 5, 100)
        p_1 = stats.norm(loc=0, scale=1)
        background = 0.1 * p_1.cdf(x)
        y = 10 * p_1.pdf(x) + background

        params = np.array([0, np.sqrt(2*np.log(2)), 10, 1])
        params.tolist()
        return x, y, background, params

    def load_two_peaks(self):
        x = np.linspace(-20, 20, 100)
        p_1 = stats.norm(loc=-2, scale=1)
        p_2 = stats.norm(loc=4, scale=2)
        y = 10 * p_1.pdf(x) + 5 * p_2.pdf(x)
        params = np.array([-2, np.sqrt(2*np.log(2)), 10, 1, 4, np.sqrt(2*np.log(2)), 5, 1])

        return x, y, params
    
    def load_two_peaks_with_background(self):
        x = np.linspace(-20, 20, 100)
        p_1 = stats.norm(loc=-2, scale=1)
        p_2 = stats.norm(loc=4, scale=2)
        background = 0.1 * p_1.cdf(x) + 0.05 * p_2.cdf(x)
        y = 10 * p_1.pdf(x) + 5 * p_2.pdf(x) + background
        params = np.array([-2, np.sqrt(2*np.log(2)), 10, 1, 4, np.sqrt(2*np.log(2)), 5, 1])

        return x, y, background, params

    def convert_x_y_to_spectrum(self, x, y, params):
        spectrum = Spectrum(x, y, name='test')
        region = Region(x, y, y, 0, 1)

        for n in range(len(params) // 4):
            line = Line(*params[n*4:(n+1)*4].tolist())
            region.lines.append(line)

        spectrum.add_region(region)

        return spectrum, region, line
