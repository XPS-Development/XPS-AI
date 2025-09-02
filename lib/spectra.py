from dataclasses import dataclass, field
from typing import Any, Optional, Union, List, Tuple
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


@dataclass
class Spectrum:
    """Spectrum data container with optional processed attributes.

    Holds the raw spectrum data and optional preprocessed data, along with
    regions of interest. Supports charge correction and region management.

    Parameters
    ----------
    x : NDArray
        Energy axis of the spectrum.
    y : NDArray
        Intensity axis of the spectrum.
    name : Optional[str], default=None
        Human-readable name for the spectrum.
    file : Optional[str], default=None
        Source file path for the spectrum.
    group : Optional[str], default=None
        Group label for dataset organization.

    Attributes
    ----------
    id : str
        Unique identifier for the spectrum (UUID4 hex).
    regions : List[Region]
        List of Region objects associated with this spectrum.
    charge_correction : float
        Applied shift to the energy axis.
    y_norm : Optional[NDArray]
        Normalized intensity (0-1), filled after normalization.
    norm_coefs : Optional[Tuple[float, float]]
        Minimum and maximum used for normalization.
    x_interpolated : Optional[NDArray]
        Interpolated x-axis for uniform sampling.
    y_interpolated : Optional[NDArray]
        Interpolated intensity corresponding to x_interpolated.
    y_smoothed : Optional[NDArray]
        Smoothed intensity using, e.g., Savitzky-Golay filter.
    y_norm_smoothed : Optional[NDArray]
        Smoothed normalized intensity.
    """

    x: NDArray
    y: NDArray
    name: Optional[str] = None
    file: Optional[str] = None
    group: Optional[str] = None

    id: str = field(default_factory=lambda: uuid4().hex)
    regions: List[Region] = field(default_factory=list)
    charge_correction: float = 0.0

    # Optional fields for processed data
    y_norm: Optional[NDArray] = None
    norm_coefs: Optional[Tuple[float, float]] = None
    x_interpolated: Optional[NDArray] = None
    y_interpolated: Optional[NDArray] = None
    y_smoothed: Optional[NDArray] = None
    y_norm_smoothed: Optional[NDArray] = None

    def add_region(self, region: Region) -> None:
        """Attach a region to the spectrum."""
        self.regions.append(region)

    def create_region(
        self, start_idx: int, end_idx: int, background_type: str = "shirley"
    ) -> Region:
        """
        Create a region from spectrum data and attach it.

        Parameters
        ----------
        start_idx : int
            Start index of the region in the spectrum arrays.
        end_idx : int
            End index of the region in the spectrum arrays.
        background_type : str, default 'shirley'
            Type of background for the region.

        Returns
        -------
        Region
            The newly created region attached to this spectrum.

        Raises
        ------
        ValueError
            If the spectrum has not been normalized and smoothed.
        """
        if self.y_norm is None or self.y_smoothed is None:
            raise ValueError(
                "Spectrum must be normalized and smoothed before creating regions."
            )

        region = Region(
            self.x[start_idx:end_idx],
            self.y[start_idx:end_idx],
            self.y_norm[start_idx:end_idx],
            self.y_smoothed[start_idx],
            self.y_smoothed[end_idx - 1],
            start_idx,
            end_idx,
            background_type,
        )
        self.add_region(region)
        return region

    def delete_region(self, r: Union[int, Region]) -> None:
        """
        Delete a region by index or instance.

        Parameters
        ----------
        r : int or Region
            Index of the region in the regions list or the Region instance.
        """
        if isinstance(r, Region):
            self.regions.remove(r)
        elif isinstance(r, int):
            self.regions.pop(r)

    def set_charge_correction(self, delta: float) -> None:
        """
        Apply a constant shift to the spectrum energy axis and all regions.

        Parameters
        ----------
        delta : float
            Shift to apply to the x-axis.
        """
        self.charge_correction += delta
        self.x += delta
        if self.x_interpolated is not None:
            self.x_interpolated += delta
        for region in self.regions:
            region.x += delta

    def remove_charge_correction(self) -> None:
        """Reset the charge correction to zero."""
        self.set_charge_correction(-self.charge_correction)
        self.charge_correction = 0.0

    def summary(self) -> str:
        """
        Return a concise summary of the spectrum.

        Returns
        -------
        str
            Summary including size of data arrays, number of regions,
            charge correction, and status of preprocessing.
        """
        lines = [
            f"Spectrum ID: {self.id}",
            f"Name: {self.name or 'N/A'}, File: {self.file or 'N/A'}, Group: {self.group or 'N/A'}",
            f"Data points: {self.x.size}",
            f"Regions: {len(self.regions)}",
            f"Charge correction: {self.charge_correction:.3f}",
            f"Normalized: {'Yes' if self.y_norm is not None else 'No'}",
            f"Smoothed: {'Yes' if self.y_smoothed is not None else 'No'}",
            f"Interpolated: {'Yes' if self.x_interpolated is not None else 'No'}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<Spectrum id={self.id[:8]} name={self.name or 'N/A'} "
            f"points={len(self.x)} regions={len(self.regions)} "
        )


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
