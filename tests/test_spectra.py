import unittest

import numpy as np
from scipy import stats 

from tools import Line, Region, Spectrum


def two_arrays_equal(a, b):
    return np.all(a == b)


class TestLine(unittest.TestCase):
    def test_fwhm(self):
        line = Line(0, 1, 1, 0)
        self.assertEqual(line.fwhm, 2)

        line = Line(0, 1, 1, 0.5)
        self.assertEqual(line.fwhm, 2)

        line = Line(0, 1, 1, 1)
        self.assertEqual(line.fwhm, 2)

    def test_area(self):
        line = Line(0, 1, 1, 0)
        self.assertAlmostEqual(line.area, 1, 1)

        line = Line(0, 1, 1, 0.5)
        self.assertAlmostEqual(line.area, 1, 1)

        line = Line(0, 1, 1, 1)
        self.assertAlmostEqual(line.area, 1, 1)

    def test_f(self):
        line = Line(0, np.sqrt(2*np.log(2)), 1, 1)
        x = np.linspace(-10, 10, 200)
        y = line.f(x)
        
        y_check = stats.norm(loc=0, scale=1).pdf(x)
        self.assertEqual(len(y), len(y_check))
        self.assertEqual(y.max(), y_check.max())
    

class TestRegion(unittest.TestCase):
    def test_repr(self):
        region = Region([1, 2, 3], [4, 5, 6], [7, 8, 9], 0, 10)
        region.add_line(0, 1, 1, 0)
        self.assertIn('Region(start=0, end=10', str(region))
        self.assertIn('Line(name=None, loc=0, scale=1, const=1, gl_ratio=0)', str(region))

    def test_add_line(self):
        region = Region([1, 2, 3], [4, 5, 6], [7, 8, 9], 0, 10)
        region.add_line(0, 1, 1, 0)
        self.assertEqual(len(region.lines), 1)

    def test_draw_lines(self):
        x = np.linspace(-10, 10, 200)
        y_init = stats.norm(loc=0, scale=1).pdf(x)
        
        region = Region(x, y_init, y_init, 0, 1)
        region.background = np.zeros(200)
        region.add_line(0, np.sqrt(2*np.log(2)), 1, 1)

        y = region.draw_lines()[-1]

        self.assertEqual(len(y), len(x))
        self.assertEqual(len(y), len(y_init))
        self.assertEqual(y.max(), y_init.max())
    

class TestSpectrum(unittest.TestCase):

    def test_init_and_preproc(self):
        # create test spectrum
        x = np.linspace(-5, 5, 100)
        y = stats.norm(loc=0, scale=1).pdf(x)
        spectrum = Spectrum(x, y, name='test')

        self.assert_(two_arrays_equal(spectrum.x, x))
        self.assert_(two_arrays_equal(spectrum.y, y))
        self.assertEqual(len(spectrum.x_interpolated), 256)
        self.assertEqual(len(spectrum.y_interpolated), 256)
        self.assertEqual(spectrum.y_interpolated.max(), 1)

    def test_add_region(self):
        # create test spectrum
        x = np.linspace(-5, 5, 100)
        y = stats.norm(loc=0, scale=1).pdf(x)
        spectrum = Spectrum(x, y, name='test')

        # create test region
        region = Region(x, y, y, 0, 1)
        region.background = np.zeros(100)

        spectrum.add_region(region)

        self.assertEqual(len(spectrum.regions), 1)
        self.assertEqual(len(spectrum.regions[0].x), len(x))
    
