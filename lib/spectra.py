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
        self.id: str = f"p{uuid4().hex}"
        self.region_id: str = region_id

        self.amp_par: PeakParameter = PeakParameter("amp", value=1, min_val=0, max_val=np.inf)
        self.cen_par: PeakParameter = PeakParameter("cen", value=0, min_val=-np.inf, max_val=np.inf)
        self.sig_par: PeakParameter = PeakParameter("sig", value=1, min_val=0, max_val=np.inf)
        self.frac_par: PeakParameter = PeakParameter("frac", value=1, min_val=0, max_val=1)

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
        return f"<Peak {self.id}: " f"amp={self.amp}, cen={self.cen}, sig={self.sig}, frac={self.frac}>"


@dataclass
class Region:
    """
    Region of interest within a spectrum.

    A `Region` stores a subsection of spectrum data (X/Y arrays),
    background information, and associated peaks. Regions may belong
    to a parent `Spectrum` and can be managed by a `SpectrumCollection`.

    Parameters
    ----------
    spectrum_id : str
        UUID of the parent spectrum this region belongs to.
    x : Optional[NDArray], default=None
        X-axis values (energy/binding energy) of the region.
    y : Optional[NDArray], default=None
        Intensity values of the region.
    y_norm : Optional[NDArray], default=None
        Normalized intensity values (0–1).
    i_1 : Optional[float], default=None
        Background intensity at the start of the region.
    i_2 : Optional[float], default=None
        Background intensity at the end of the region.
    background_type : str, default="shirley"
        Type of background applied (e.g., "shirley", "linear").
    collection : Optional[SpectrumCollection], default=None
        Reference to the collection for automatic registration.

    Attributes
    ----------
    id : str
        Unique identifier of the region (UUID4 hex).
    peaks : list[Peak]
        Peaks associated with this region.
    """

    spectrum_id: str
    x: Optional[NDArray] = None
    y: Optional[NDArray] = None
    y_norm: Optional[NDArray] = None
    i_1: Optional[float] = None
    i_2: Optional[float] = None
    background_type: str = "shirley"
    collection: Optional["SpectrumCollection"] = None

    id: str = field(default_factory=lambda: f"r{uuid4().hex}")
    peaks: List["Peak"] = field(default_factory=list)

    def add_peak(self, peak: "Peak") -> None:
        """
        Attach a peak to the region and notify collection if present.

        Parameters
        ----------
        peak : Peak
            Peak instance to add.
        """
        self.peaks.append(peak)
        if self.collection is not None:
            self.collection.register(peak)

    def remove_peak(self, peak: Union["Peak", str]) -> None:
        """
        Remove a peak from the region by instance or ID and notify collection.

        Parameters
        ----------
        peak : Peak or str
            Peak instance or its UUID to remove.
        """
        if isinstance(peak, Peak):
            self.peaks = [p for p in self.peaks if p != peak]
            if self.collection is not None:
                self.collection.peaks_index.pop(peak.id, None)
        elif isinstance(peak, str):
            self.peaks = [p for p in self.peaks if p.id != peak]
            if self.collection is not None:
                self.collection.peaks_index.pop(peak, None)

    def update_range(
        self,
        x: NDArray,
        y: NDArray,
        y_norm: Optional[NDArray] = None,
        i_1: Optional[float] = None,
        i_2: Optional[float] = None,
    ) -> None:
        """
        Update the data arrays and background values of the region.

        Parameters
        ----------
        x : NDArray
            New X-axis values.
        y : NDArray
            New Y-axis values.
        y_norm : Optional[NDArray], default=None
            New normalized Y values.
        i_1 : Optional[float], default=None
            Background intensity at the start of the region.
        i_2 : Optional[float], default=None
            Background intensity at the end of the region.
        """
        self.x = x
        self.y = y
        if y_norm is not None:
            self.y_norm = y_norm
        if i_1 is not None:
            self.i_1 = i_1
        if i_2 is not None:
            self.i_2 = i_2

    def __repr__(self) -> str:
        return (
            f"<Region id={self.id[:8]} spectrum_id={self.spectrum_id[:8]} "
            f"points={self.x.size if self.x is not None else 0} "
            f"peaks={len(self.peaks)} background_type={self.background_type}>"
        )


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
    collection: Optional[SpectrumCollection], default=None
        Collection to which the spectrum belongs.

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

    id: str = field(default_factory=lambda: f"s{uuid4().hex}")
    regions: List[Region] = field(default_factory=list)
    charge_correction: float = 0.0

    _collection: Optional["SpectrumCollection"] = field(default=None, repr=False, init=False)

    # Optional fields for processed data
    y_norm: Optional[NDArray] = None
    norm_coefs: Optional[Tuple[float, float]] = None
    x_interpolated: Optional[NDArray] = None
    y_interpolated: Optional[NDArray] = None
    y_smoothed: Optional[NDArray] = None
    y_norm_smoothed: Optional[NDArray] = None

    @property
    def collection(self) -> Optional["SpectrumCollection"]:
        return self._collection

    @collection.setter
    def collection(self, collection: Optional["SpectrumCollection"]) -> None:
        self._collection = collection
        for region in self.regions:
            region.collection = collection

    def add_region(self, region: Region) -> None:
        """Attach region to spectrum and notify collection if present."""
        self.regions.append(region)
        if self.collection is not None:
            self.collection.register(region)

    def create_region(self, start_idx: int, end_idx: int, background_type: str = "shirley") -> Region:
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
            raise ValueError("Spectrum must be normalized and smoothed before creating regions.")

        region = Region(
            spectrum_id=self.id,
            x=self.x[start_idx:end_idx],
            y=self.y[start_idx:end_idx],
            y_norm=self.y_norm[start_idx:end_idx],
            i_1=self.y_smoothed[start_idx],
            i_2=self.y_smoothed[end_idx - 1],
            background_type=background_type,
        )
        self.add_region(region)
        return region

    def remove_region(self, region: Union[Region, str]) -> None:
        """
        Remove a region from the spectrum by instance or ID and notify collection.

        Parameters
        ----------
        region : Region or str
            Region instance or its UUID to remove.
        """
        if isinstance(region, Region):
            self.regions = [r for r in self.regions if r.id != region.id]
            if self.collection is not None:
                self.collection.region_index.pop(region.id, None)
                for peak in region.peaks:
                    self.collection.peaks_index.pop(peak.id, None)
        elif isinstance(region, str):
            self.regions = [r for r in self.regions if r.id != region]
            if self.collection is not None:
                self.collection.region_index.pop(region, None)
                # удалить все пики региона тоже:
                reg = self.collection.region_index.get(region)
                if reg:
                    for peak in reg.peaks:
                        self.collection.peaks_index.pop(peak.id, None)

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
            f"points={self.x.size} regions={len(self.regions)} "
        )


class SpectrumCollection:
    """
    Container for managing and indexing spectra, regions, and peaks.

    SpectrumCollection maintains fast lookup tables (indexes) that map
    object IDs to their corresponding objects. This allows retrieval
    of any registered :class:`Spectrum`, :class:`Region`, or :class:`Peak`
    by its UUID.

    Notes
    -----
    - All objects (`Spectrum`, `Region`, `Peak`) must have an `id` attribute
      that is unique within the collection.
    - When adding a spectrum via :meth:`add_spectrum`, its regions and peaks
      are automatically registered.
    - Removing objects is typically managed through
      :meth:`Spectrum.remove_region` and :meth:`Region.remove_peak`
      which also keep the collection in sync.

    Attributes
    ----------
    peaks_index : dict[str, Peak]
        Mapping of peak UUIDs to peak objects.
    region_index : dict[str, Region]
        Mapping of region UUIDs to region objects.
    spectra_index : dict[str, Spectrum]
        Mapping of spectrum UUIDs to spectrum objects.
    """

    def __init__(self):
        self.peaks_index = {}  # {id: Peak}
        self.region_index = {}  # {id: Region}
        self.spectra_index = {}  # {id: Spectrum}

    def register(self, obj: Union[Spectrum, Region, Peak]):
        """
        Register an object (spectrum, region, or peak) in the collection.

        Parameters
        ----------
        obj : Spectrum or Region or Peak
            Object to register in the collection.
        """
        if isinstance(obj, Peak):
            self.peaks_index[obj.id] = obj
        elif isinstance(obj, Region):
            self.region_index[obj.id] = obj
        elif isinstance(obj, Spectrum):
            self.spectra_index[obj.id] = obj

    def add_spectrum(self, spectrum: Spectrum):
        """
        Add a spectrum and automatically register all its regions and peaks.
        """
        self.register(spectrum)
        for region in spectrum.regions:
            self.register(region)
            for peak in region.peaks:
                self.register(peak)

    def get_spectrum(self, id: str):
        """
        Retrieve a spectrum by its UUID.

        Parameters
        ----------
        id : str
            Spectrum UUID.

        Returns
        -------
        Spectrum
            The spectrum corresponding to the given ID.
        """
        return self.spectra_index[id]

    def get_peak(self, id: str):
        """
        Retrieve a peak by its UUID.
        """
        return self.peaks_index[id]

    def get_region(self, id: str):
        """
        Retrieve a region by its UUID.
        """
        return self.region_index[id]
