from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from .funcs import pvoigt


class PeakParameter:
    """
    Single peak parameter with bounds, variation flag, and optional dependency expression.

    Parameters
    ----------
    name : str
        Parameter name (e.g., "pos", "amp", "fwhm").
    value : float
        Initial value of the parameter. Will be clipped to [min_val, max_val].
    min_val : float, default=-np.inf
        Minimum allowed value.
    max_val : float, default=np.inf
        Maximum allowed value.
    vary : bool, default=True
        Whether the parameter can be varied during optimization.
    expr : str | None, default=None
        Optional string expression for dependency on other parameters.
        Should reference other parameters by their unique names.
        E.g., '2 * p0123' reference to the same parameter (pos/amp/...) in peak with id 0123.

    Attributes
    ----------
    value : float
        Current value of the parameter, clipped to [min, max].
    min : float
        Minimum allowed value.
    max : float
        Maximum allowed value.
    vary : bool
        Variation flag for optimization.
    expr : str | None
        Dependency expression.
    """

    def __init__(
        self,
        name: str,
        value: float,
        min_val: float = -np.inf,
        max_val: float = np.inf,
        vary: bool = True,
        expr: str | None = None,
    ) -> None:
        self.name: str = name
        self.min: float = min_val
        self.max: float = max_val
        self.vary: bool = vary
        self.expr: str | None = expr

        # Clip initial value
        self.value: float = self._clip_value(value)

    def _clip_value(self, value: float) -> float:
        """
        Clip value to [min, max].

        Parameters
        ----------
        value : float
            Input value.

        Returns
        -------
        float
            Clipped value within the bounds [min, max].
        """
        if value < self.min:
            return self.min
        elif value > self.max:
            return self.max
        return value

    def set(self, **kwargs) -> None:
        """
        Update parameter attributes.

        Parameters
        ----------
        value : float, optional
            New value for the parameter (will be clipped to [min, max]).
        min : float, optional
            New minimum bound.
        max : float, optional
            New maximum bound.
        vary : bool, optional
            Whether the parameter can be varied during optimization.
        expr : str, optional
            Dependency expression referencing other parameters.
        """
        for key, value in kwargs.items():
            if key == "value":
                self.value = self._clip_value(float(value))
            elif key in ("min", "max"):
                setattr(self, key, float(value))
                self.value = self._clip_value(self.value)
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return (
            f"<PeakParameter {self.name}: "
            f"value={self.value}, min={self.min}, max={self.max}, "
            f"vary={self.vary}, expr={self.expr}>"
        )


class Peak:
    """
    Represents a single pseudo-Voigt peak with parameters and FWHM synchronization.

    Parameters
    ----------
    region_id : str
        Identifier of the parent region to which this peak belongs.

    Attributes
    ----------
    id : str
        Unique identifier of the peak (UUID4 hex string).
    region_id : str
        Identifier of the parent region.
    amp_par : PeakParameter
        Amplitude of the peak.
    cen_par : PeakParameter
        Center position of the peak.
    sig_par : PeakParameter
        Gaussian/Lorentzian width parameter of the peak.
    frac_par : PeakParameter
        Fraction of Lorentzian component in pseudo-Voigt.
    """

    def __init__(self, region_id: str) -> None:
        self.id: str = uuid4().hex
        self.region_id: str = region_id

        self.amp_par: PeakParameter = PeakParameter(
            "amp", value=1, min_val=0, max_val=np.inf
        )
        self.cen_par: PeakParameter = PeakParameter(
            "cen", value=0, min_val=-np.inf, max_val=np.inf
        )
        self.sig_par: PeakParameter = PeakParameter(
            "sig", value=1, min_val=0, max_val=np.inf
        )
        self.frac_par: PeakParameter = PeakParameter(
            "frac", value=1, min_val=0, max_val=1
        )

    @property
    def fwhm(self) -> float:
        """Full width at half maximum of the peak (2 * sig)."""
        return 2 * self.sig

    @fwhm.setter
    def fwhm(self, value: float) -> None:
        """Set FWHM, automatically updating `sig`."""
        self.sig = value / 2

    def __getattr__(self, name: str) -> Any:
        """Delegate access to PeakParameter values."""
        par_attr = f"{name}_par"
        param = self.__dict__.get(par_attr)
        if isinstance(param, PeakParameter):
            return param.value
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate setting of PeakParameter values; handle fwhm separately."""
        if name == "fwhm":
            super().__setattr__(name, value)
        else:
            par_attr = f"{name}_par"
            param = self.__dict__.get(par_attr)
            if isinstance(param, PeakParameter):
                param.set(value=value)
            else:
                super().__setattr__(name, value)

    def set(self, name: str, **kwargs: Any) -> None:
        """
        Update attributes of a PeakParameter.

        Parameters
        ----------
        name : str
            Name of the parameter without '_par' suffix.
        **kwargs : dict
            Arguments to pass to PeakParameter.set(), e.g., value, min_val, max_val, vary, expr.
        """
        par_attr = f"{name}_par"
        param = self.__dict__.get(par_attr)
        if isinstance(param, PeakParameter):
            param.set(**kwargs)

    def f(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Evaluate the pseudo-Voigt function at given x values.

        Parameters
        ----------
        x : NDArray[np.float64]
            Array of x-values.

        Returns
        -------
        NDArray[np.float64]
            Evaluated y-values of the pseudo-Voigt peak.
        """
        return pvoigt(x, self.amp, self.cen, self.sig, self.frac)

    def __repr__(self) -> str:
        return (
            f"<Peak {self.id}: "
            f"amp={self.amp}, cen={self.cen}, sig={self.sig}, frac={self.frac}>"
        )


class Region:
    def __init__(self, spectrum_id: str):
        self.id: str = uuid4().hex
        self.spectrum_id: str = spectrum_id
        self.peaks: list[Peak] = []


class Spectrum:
    def __init__(self):
        self.id: str = uuid4().hex
        self.regions: list[Region] = []


class SpectrumCollection:
    def __init__(self):
        self.peaks_index = {}  # {id: object}
        self.region_index = {}
        self.spectra_index = {}

    def register(self, obj):
        if isinstance(obj, Peak):
            self.peaks_index[obj.id] = obj
        elif isinstance(obj, Region):
            self.region_index[obj.id] = obj
        elif isinstance(obj, Spectrum):
            self.spectra_index[obj.id] = obj

    def add_spectrum(self, spectrum):
        self.register(spectrum)
        for region in spectrum.regions:
            self.register(region)
            for peak in region.peaks:
                self.register(peak)

    def get_spectrum(self, id):
        return self.spectra_index[id]

    def get_peak(self, id):
        return self.peaks_index[id]

    def get_region(self, id):
        return self.region_index[id]
