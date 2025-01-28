import unittest

import numpy as np
from scipy import stats
import torch

from model.models.model_deeper import XPSModel
from tools import Analyzer, Spectrum


MODEL = XPSModel()
MODEL.load_state_dict(torch.load('model/trained_models/model.pt', map_location=torch.device('cpu'), weights_only=True))
MODEL.eval()


def two_arrays_equal(a, b):
    return np.all(a == b)


def two_arrays_almost_equal(a, b, tol=1e-5):
    return np.all(np.abs(a - b) < tol)


class TestAnalyzer(unittest.TestCase):    
    def test_prepare_input(self):
        analyzer = Analyzer(MODEL)
        x = np.random.random((1000, 1))
        y = np.random.random((1000, 1))

        t_x = analyzer.prepare_input(x)
        t_y = analyzer.prepare_input(y)

        self.assertEqual(t_x.shape, (2, 1000, 1))
        self.assertEqual(t_y.shape, (2, 1000, 1))

    def test_batch_data(self):
        analyzer = Analyzer(MODEL)
        x = np.random.random((100,))
        y = np.random.random((100,))

        t_x = analyzer.batch_data(Spectrum(x, y), Spectrum(x, y))
        self.assertEqual(t_x.shape, (2, 2, 256))
    
    def test_predict(self):
        analyzer = Analyzer(MODEL)
        x = np.linspace(-5, 5, 100)
        y = stats.norm(loc=0, scale=1).pdf(x)
        spectrum = Spectrum(x, y, name='test')
        
        analyzer.predict(spectrum)

        self.assertEqual(spectrum.get_masks()[0].shape, (256,))        

    def test_find_borders(self):
        analyzer = Analyzer(MODEL)

        mask = np.ones(10)
        idxs = analyzer._find_borders(mask)
        self.assertEqual(len(idxs), 2)

        mask = np.zeros(10)
        idxs = analyzer._find_borders(mask)
        self.assertEqual(len(idxs), 0)

        mask = np.array([0, 1, 0])
        idxs = analyzer._find_borders(mask)
        self.assert_(two_arrays_equal(idxs, np.array([1, 2])))

        mask = np.array([1, 0, 1])
        idxs = analyzer._find_borders(mask)
        self.assert_(two_arrays_equal(idxs, np.array([0, 1, 2, 3])))

    def test_prepare_max_mask(self):
        analyzer = Analyzer(MODEL)

        mask = np.array([0, 1, 0])
        idxs = analyzer.prepare_max_mask(mask)
        self.assert_(two_arrays_equal(idxs, np.array([1])))

        mask = np.array([1, 0, 1])
        idxs = analyzer.prepare_max_mask(mask)
        self.assert_(two_arrays_equal(idxs, np.array([0, 2])))

        mask = np.array([1, 1, 1])
        idxs = analyzer.prepare_max_mask(mask)
        self.assert_(two_arrays_equal(idxs, np.array([1])))
    
    def test_init_params_by_locations(self):
        analyzer = Analyzer(MODEL)

        x = np.linspace(-10, 10, 100)

        # one peak
        y = stats.norm(loc=0, scale=1).pdf(x)
        locations = np.array([0])
        params, bounds = analyzer.init_params_by_locations(x, y, locations)
        self.assert_(
            two_arrays_almost_equal(params, np.array([0, np.sqrt(2*np.log(2)), 1, 1])),
            msg=f'params: {params}'
        )

        # two peaks
        y = stats.norm(loc=-2, scale=1).pdf(x) + stats.norm(loc=3, scale=1).pdf(x)
        locations = np.array([-2, 3])
        params, bounds = analyzer.init_params_by_locations(x, y, locations)
        self.assert_(two_arrays_almost_equal(
            params, np.array([-2, np.sqrt(2*np.log(2)), 1, 1, 3, np.sqrt(2*np.log(2)), 1, 1])),
            msg=f'params: {params}'
        )
    
    def test_static_shirley(self):
        analyzer = Analyzer(MODEL)

        # one peak
        x = np.linspace(-5, 5, 100)
        y = stats.norm(loc=0, scale=1).pdf(x)
        background = 0.1 * stats.norm(loc=0, scale=1).cdf(x)
        y += background
        
        test_background = analyzer.static_shirley(x, y, y[0], y[-1])
        self.assert_(two_arrays_almost_equal(test_background, background, tol=0.1))

        # two peaks
        x = np.linspace(-20, 20, 100)
        p_1 = stats.norm(loc=-2, scale=1)
        p_2 = stats.norm(loc=4, scale=2)
        y = p_1.pdf(x) + 0.5 * p_2.pdf(x)
        background = 0.05 * p_1.cdf(x) + 0.025 * p_2.cdf(x)
        y += background
        
        test_background = analyzer.static_shirley(x, y, y[0], y[-1])
        self.assert_(two_arrays_almost_equal(test_background, background, tol=0.1))
    
    def test_post_process(self):
        analyzer = Analyzer(MODEL)

        # one peak
        x = np.linspace(-5, 5, 100)
        p_1 = stats.norm(loc=0, scale=1)
        y = 10 * p_1.pdf(x) + 0.5 * p_1.cdf(x) + 1

        spectrum = Spectrum(x, y, name='test')
        spectrum.add_masks(
            np.pad(np.ones(154), (51, 51), 'constant'),
            np.pad(np.ones(4), (126, 126), 'constant')
        )
        analyzer.post_process(spectrum)
        
        line = spectrum.regions[0].lines[0]
        true = np.array([0, np.sqrt(2*np.log(2)), 10, 1])
        test = np.array([line.loc, line.scale, line.const, line.gl_ratio])
        self.assert_(two_arrays_almost_equal(test, true, tol=0.1), msg=f'test: {test}, true: {true}')

    def test_refit(self):
        analyzer = Analyzer(MODEL)

        # one peak
        x = np.linspace(-5, 5, 100)
        p_1 = stats.norm(loc=0.2, scale=1)
        y = 10 * p_1.pdf(x)

        true = np.array([0.2, np.sqrt(2*np.log(2)), 10, 1])
        test = analyzer.refit(x, y, 1, [0, np.sqrt(2*np.log(2)), 10, 1], 0, 0.2)
        self.assert_(two_arrays_almost_equal(test, true, tol=0.05), msg=f'test: {test}, true: {true}')
