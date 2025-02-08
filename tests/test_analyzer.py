import unittest

import numpy as np
from scipy import stats

from tools import Analyzer, Spectrum, Region

from testing_tools import two_arrays_almost_equal, two_arrays_equal, MODEL, SpectrumCase


spec_generator = SpectrumCase()


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

        x, y, _ = spec_generator.load_one_peak()
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
        x, y, true_params = spec_generator.load_one_norm_like_peak()
        locations = np.array([0])
        params, bounds = analyzer.init_params_by_locations(x, y, locations)
        self.assert_(
            two_arrays_almost_equal(params, true_params),
            msg=f'params: {params}'
        )

        # two peaks
        x, y, true_params = spec_generator.load_two_norm_like_peaks()
        locations = np.array([-2, 4])
        params, bounds = analyzer.init_params_by_locations(x, y, locations)
        self.assert_(
            two_arrays_almost_equal(params, true_params),
            msg=f'test: {params}, true: {true_params}'
        )
    
    def test_static_shirley(self):
        analyzer = Analyzer(MODEL)

        # one peak
        x, y, background, true_params = spec_generator.load_one_peak_with_background()
        
        test_background = analyzer.static_shirley(x, y, y[0], y[-1])
        self.assert_(two_arrays_almost_equal(test_background, background, tol=0.1))

        # two peaks
        x, y, background, true_params = spec_generator.load_two_peaks_with_background()
        
        test_background = analyzer.static_shirley(x, y, y[0], y[-1])
        self.assert_(two_arrays_almost_equal(test_background, background, tol=0.1))
    
    def test_post_process(self):
        analyzer = Analyzer(MODEL)

        # one peak
        x, y, background, true_params = spec_generator.load_one_peak_with_background()
        spectrum = Spectrum(x, y, name='test')
        spectrum.add_masks(
            np.pad(np.ones(154), (51, 51), 'constant'),
            np.pad(np.ones(4), (126, 126), 'constant')
        )
        analyzer.post_process(spectrum)
        
        line = spectrum.regions[0].lines[0]
        test = np.array([line.loc, line.scale, line.const, line.gl_ratio])
        self.assert_(two_arrays_almost_equal(test, true_params, tol=0.2), msg=f'test: {test}, true: {true_params}')

    def test_refit(self):
        analyzer = Analyzer(MODEL)

        # one peak
        x, y, true_params = spec_generator.load_one_peak()

        # test differential evolution
        test = analyzer._refit(x, y, 1, [0, np.sqrt(2*np.log(2)), 10, 1], 0, 0.2)
        self.assert_(two_arrays_almost_equal(test, true_params, tol=0.05), msg=f'test: {test}, true: {true_params}')

        # fix loc
        initial = np.array([0.001, np.sqrt(2), 8, 0.8])
        test = analyzer._refit(x, y, 1, initial, 0, 4, fixed_params=[0])
        self.assert_(two_arrays_almost_equal(test, true_params, tol=0.2), msg=f'test: {test}, true: {true_params}')

        # test least squares
        # fix loc and gl_ratio
        initial = np.array([0.001, np.sqrt(2), 8, 1])
        test = analyzer._refit(x, y, 1, initial, 0, 4, fixed_params=[0, 3], fit_alg='least squares')
        self.assert_(two_arrays_almost_equal(test, true_params, tol=0.2), msg=f'test: {test}, true: {true_params}')

    def test_refit_region(self):
        analyzer = Analyzer(MODEL)
        # one peak
        x, y, true_params = spec_generator.load_one_norm_like_peak()

        # full refit
        region = Region(x, y, y, 0, 1)
        region.add_line(0, 1, 1, 1)
        region.background = 0
        region.norm_coefs = (0, 1)
        analyzer.refit_region(region, use_norm_y=False, full_refit=True)
        line = region.lines[0]
        test = np.array([line.loc, line.scale, line.const, line.gl_ratio])
        self.assert_(two_arrays_almost_equal(test, true_params, tol=0.2), msg=f'test: {test}, true: {true_params}')

        # partial refit
        region = Region(x, y, y, 0, 1)
        region.add_line(0, 1, 1, 1)
        region.background = 0
        region.norm_coefs = (0, 1)
        analyzer.refit_region(region, use_norm_y=False, full_refit=False, tol=1, fixed_params=[0, 3])
        line = region.lines[0]
        test = np.array([line.loc, line.scale, line.const, line.gl_ratio])
        self.assert_(two_arrays_almost_equal(test, true_params, tol=0.2), msg=f'test: {test}, true: {true_params}')
