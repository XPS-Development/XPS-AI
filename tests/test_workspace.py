import unittest
import numpy as np

from tools._workspace import Workspace
from tools._spectra import Spectrum, Line

from testing_tools import two_arrays_equal, two_arrays_almost_equal, MODEL, SpectrumCase


spec_generator = SpectrumCase()

class TestWorkspace(unittest.TestCase):
    def check_valid_spectrum(self, spectrum):
        self.assertIsInstance(spectrum, Spectrum)
        self.assertEqual(len(spectrum.x_interpolated), 256)
        self.assertEqual(len(spectrum.y_interpolated), 256)
        self.assertEqual(spectrum.y_norm.max(), 1)
        self.assertEqual(spectrum.y_norm.min(), 0)

    def test_create_group(self):
        workspace = Workspace(None)
        workspace.create_group('test')
        self.assertIn('test', workspace.groups)
        self.assertEqual(workspace.groups['test'], [])

    def test_add_spectrum(self):
        workspace = Workspace(None)
        x = np.linspace(0, 100, 200)
        y = np.linspace(0, 100, 200)
        workspace.add_spectrum(x, y, group_name='test')
        self.assertIn('test', workspace.groups)
        self.assertEqual(len(workspace.groups['test']), 1)
    
    def test_rename_group(self):
        workspace = Workspace(None)
        workspace.create_group('test')
        workspace.rename_group('test', 'new_test')
        self.assertNotIn('test', workspace.groups)
        self.assertIn('new_test', workspace.groups)

    def test_move_spectrum(self):
        workspace = Workspace(None)
        workspace.create_group('test')
        x = np.linspace(0, 100, 200)
        y = np.linspace(0, 100, 200)
        workspace.add_spectrum(x, y, group_name='test')
        workspace.create_group('new_test')
        workspace.move_spectrum(0, 'test', 'new_test')
        self.assertEqual(len(workspace.groups['test']), 0)
        self.assertEqual(len(workspace.groups['new_test']), 1)

    def test_merge_groups(self):
        workspace = Workspace(None)
        workspace.create_group('test1')
        workspace.create_group('test2')
        x_1 = np.linspace(0, 100, 200)
        y_1 = np.linspace(0, 100, 200)
        x_2 = np.linspace(0, 100, 200)
        y_2 = np.linspace(0, 100, 200)
        workspace.add_spectrum(x_1, y_1, group_name='test1')
        workspace.add_spectrum(x_2, y_2, group_name='test2')
        workspace.merge_groups('test1', 'test2')
        self.assertNotIn('test2', workspace.groups)
        self.assertIn('test1', workspace.groups)
        self.assertEqual(len(workspace.groups['test1']), 2)

    def test_delete_group(self):
        workspace = Workspace(None)
        workspace.create_group('test')
        workspace.delete_group('test')
        self.assertNotIn('test', workspace.groups)

    def test_delete_spectrum(self):
        workspace = Workspace(None)
        workspace.create_group('test')
        x = np.linspace(0, 100, 200)
        y = np.linspace(0, 100, 200)
        workspace.add_spectrum(x, y, group_name='test')
        workspace.delete_spectrum('test', 0)
        self.assertEqual(len(workspace.groups['test']), 0)

    def test_load_vamas(self):
        workspace = Workspace(None)
        workspace.load_vamas('tests/test_data/test_1_spec.vms', 'test_1')
        self.assertIn('test_1', workspace.groups)
        self.assertEqual(len(workspace.groups['test_1']), 1)
        spec = workspace.groups['test_1'][0]
        self.check_valid_spectrum(spec)

        workspace.load_vamas('tests/test_data/test_18_spec.vms', 'test_2')
        self.assertIn('test_2', workspace.groups)
        self.assertEqual(len(workspace.groups['test_2']), 18)
        spec = workspace.groups['test_2'][0]
        self.check_valid_spectrum(spec)

    def test_load_specs2(self):
        workspace = Workspace(None)
        workspace.load_specs2('tests/test_data/test_1_spec.xml', group_name='test')
        self.assertIn('test', workspace.groups)
        self.assertEqual(len(workspace.groups['test']), 1)
        spec = workspace.groups['test'][0]
        self.check_valid_spectrum(spec)

    def test_load_txt(self):
        workspace = Workspace(None)
        workspace.load_txt('tests/test_data/test_1_spec.txt', group_name='test')
        self.assertIn('test', workspace.groups)
        self.assertEqual(len(workspace.groups['test']), 1)
        spec = workspace.groups['test'][0]
        self.check_valid_spectrum(spec)

    def test_load_files(self):
        workspace = Workspace(None)
        workspace.load_files('tests/test_data/test_1_spec.txt', 'tests/test_data/test_18_spec.vms', group_name='test')
        self.assertIn('test', workspace.groups)
        self.assertEqual(len(workspace.groups['test']), 19)

    def test_predict(self):
        workspace = Workspace(MODEL)
        workspace.load_txt('tests/test_data/test_1_spec.txt', group_name='test')
        workspace.predict('test')
        for spec in workspace.groups['test']:
            self.assertEqual(spec.max.max(), 1)
            self.assertEqual(spec.max.min(), 0)
        
    def test_post_process(self):
        workspace = Workspace(MODEL)
        workspace.create_group('test')
        x, y, params = spec_generator.load_one_norm_like_peak()
        spectrum = Spectrum(x, y, name='test')
        workspace.groups['test'].append(spectrum)
        spectrum.add_masks(
            np.pad(np.ones(180), (38, 38), 'constant'),
            np.pad(np.ones(4), (126, 126), 'constant')
        )
        workspace.post_process('test')
        line = spectrum.regions[0].lines[0]
        test = np.array([line.loc, line.scale, line.const, line.gl_ratio])
        self.assert_(two_arrays_almost_equal(test, params, tol=0.2), msg=f'test: {test}, true: {params}')

    def test_charge_correction(self):
        workspace = Workspace(MODEL)
        workspace.create_group('test')

        x, y, params = spec_generator.load_one_norm_like_peak()
        spectrum = Spectrum(x, y, name='test')
        region = spectrum.create_region(0, len(x))
        region.lines.append(Line(0, 1, 0, 1))
        workspace.groups['test'].append(spectrum)

        workspace.set_charge_correction(current_line_energy=0, desired_line_energy=10)
        line = spectrum.regions[0].lines[0]
        self.assertEqual(line.loc, 10)
        self.assertEqual(region.x[0], 5)
        self.assertEqual(spectrum.x_interpolated[0], 5)
        self.assertEqual(spectrum.x[0], 5)
