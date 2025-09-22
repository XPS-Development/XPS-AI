import numpy as np
import pytest
from uuid import uuid4

from lib.spectra import PeakParameter, Peak


class TestPeakParameter:
    def test_initial_value_within_bounds(self):
        p = PeakParameter("amp", value=5, min_val=0, max_val=10)
        assert p.value == 5

    def test_value_clipped_below_min(self):
        p = PeakParameter("amp", value=-5, min_val=0, max_val=10)
        assert p.value == 0

    def test_value_clipped_above_max(self):
        p = PeakParameter("amp", value=20, min_val=0, max_val=10)
        assert p.value == 10

    def test_set_value_within_bounds(self):
        p = PeakParameter("amp", value=1, min_val=0, max_val=10)
        p.set(value=7)
        assert p.value == 7

    def test_set_value_clipped(self):
        p = PeakParameter("amp", value=1, min_val=0, max_val=10)
        p.set(value=20)
        assert p.value == 10

    def test_set_min_and_max_updates_value(self):
        p = PeakParameter("amp", value=5, min_val=0, max_val=10)
        p.set(min=6)
        assert p.value == 6
        p.set(max=4)
        assert p.value == 4

    def test_set_expr_and_vary(self):
        p = PeakParameter("amp", value=1)
        p.set(expr="2 * other", vary=False)
        assert p.expr == "2 * other"
        assert not p.vary


class TestPeak:
    def test_peak_has_unique_id_and_region_id(self):
        region_id = uuid4().hex
        peak = Peak(region_id)
        assert peak.region_id == region_id
        assert isinstance(peak.id, str)

    def test_peak_parameters_are_peakparameters(self):
        peak = Peak("region1")
        assert isinstance(peak.amp_par, PeakParameter)
        assert isinstance(peak.cen_par, PeakParameter)
        assert isinstance(peak.sig_par, PeakParameter)
        assert isinstance(peak.frac_par, PeakParameter)

    def test_getattr_returns_parameter_values(self):
        peak = Peak("region1")
        peak.amp_par.set(value=3.5)
        assert peak.amp == 3.5

    def test_setattr_updates_parameter_values(self):
        peak = Peak("region1")
        peak.amp = 2.0
        assert peak.amp_par.value == 2.0

    def test_set_method_updates_parameter(self):
        peak = Peak("region1")
        peak.set("amp", value=4.2, min=0, max=10)
        assert peak.amp == 4.2
        assert peak.amp_par.max == 10

    def test_fwhm_property_getter_and_setter(self):
        peak = Peak("region1")
        peak.sig = 2.0
        assert peak.fwhm == 4.0
        peak.fwhm = 10.0
        assert np.isclose(peak.sig, 5.0)

    def test_pvoigt_evaluation(self):
        x = np.linspace(-5, 5, 100)
        peak = Peak("region1")
        y = peak.f(x)
        assert isinstance(y, np.ndarray)
        assert y.shape == x.shape
