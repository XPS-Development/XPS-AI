from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from .funcs import pvoigt, static_shirley_background, linear_background


class PeakParameter:
    """
    Single peak parameter with bounds, variation flag, and optional dependency expression.

    Parameters
    ----------
    name : str
        Parameter name (e.g., "cen", "amp", "sig").
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
        E.g., '2 * p0123' reference to the same parameter (cen/amp/...) in peak with id 0123.

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

    A `Peak` encapsulates four parameters (amplitude, center, width, Lorentzian fraction)
    that define a pseudo-Voigt profile. Parameters are stored as `PeakParameter` objects,
    but can be accessed and modified directly via attributes (`amp`, `cen`, `sig`, `frac`).
    The class also provides a convenient `fwhm` property synchronized with `sig`.

    Attributes
    ----------
    id : str
        Unique identifier of the peak (UUID4 hex string).
    amp_par : PeakParameter
        Amplitude parameter of the peak. Also accessible via `peak.amp`.
    cen_par : PeakParameter
        Center position parameter of the peak. Also accessible via `peak.cen`.
    sig_par : PeakParameter
        Gaussian/Lorentzian width parameter of the peak (σ). Related to FWHM as `fwhm = 2 * sig`.
        Also accessible via `peak.sig`.
    frac_par : PeakParameter
        Fraction of Lorentzian component in the pseudo-Voigt (0 = pure Gaussian, 1 = pure Lorentzian).
        Also accessible via `peak.frac`.
    fwhm : float
        Full width at half maximum of the peak. Setting this property updates `sig`.
    region_id : str or None, optional
        Identifier of the parent region this peak belongs to. Default is None.
    """

    def __init__(
        self,
        id_: str | None = None,
        amp: float = 1,
        cen: float = 0,
        sig: float = 1,
        frac: float = 1,
        region_id: str | None = None,
    ) -> None:

        self.id: str = f"p{uuid4().hex}" if id_ is None else id_
        self.region_id: str | None = region_id

        self.amp_par: PeakParameter = PeakParameter("amp", value=amp, min_val=0, max_val=np.inf)
        self.cen_par: PeakParameter = PeakParameter("cen", value=cen, min_val=-np.inf, max_val=np.inf)
        self.sig_par: PeakParameter = PeakParameter("sig", value=sig, min_val=0, max_val=np.inf)
        self.frac_par: PeakParameter = PeakParameter("frac", value=frac, min_val=0, max_val=1)

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

    A `Region` represents a subsection of spectral data defined by its
    X/Y arrays. It may include background information, normalization
    coefficients, and a set of associated peaks. Regions can be linked
    to a parent :class:`Spectrum` and managed collectively in a
    :class:`SpectrumCollection`.

    Parameters
    ----------
    x : Optional[NDArray], default=None
        X-axis values (e.g., binding energy).
    y : Optional[NDArray], default=None
        Y-axis values (intensity).
    norm_coefs : tuple[float, float], default=(0, 1)
        Coefficients used for normalization of intensity values.
    i_1 : Optional[float], default=None
        Background intensity at the start of the region.
    i_2 : Optional[float], default=None
        Background intensity at the end of the region.
    spectrum_id : Optional[str], default=None
        UUID of the parent spectrum this region belongs to.
    background_type : str, default="shirley"
        Static background model to apply. Supported values:
        ``"shirley"`` or ``"linear"``.

    Attributes
    ----------
    id : str | None
        Unique identifier of the region (auto-generated, format ``r<uuid>``).
    peaks : list[str]
        List of IDs of peaks associated with this region.
    background : NDArray
        Computed background for the region. The algorithm depends on
        ``background_type`` and uses ``x``, ``y``, ``i_1``, and ``i_2``.
    """

    id: str = field(default_factory=lambda: f"r{uuid4().hex}")
    x: Optional[NDArray] = None
    y: Optional[NDArray] = None
    norm_coefs: Tuple[float, float] = (0, 1)

    i_1: Optional[float] = None
    i_2: Optional[float] = None

    spectrum_id: Optional[str] = None

    background_type: str = "shirley"

    peaks: List[str] = field(default_factory=list)  # peak IDs

    @property
    def background(self) -> NDArray:
        """
        Compute the static background for the region.

        Returns
        -------
        NDArray
            Background intensity values matching the region's X-axis.
        """
        if self.background_type == "shirley":
            return static_shirley_background(self.x, self.y, self.i_1, self.i_2)
        elif self.background_type == "linear":
            return linear_background(self.x, self.i_1, self.i_2)
        else:
            raise ValueError(f"Unknown static background type: {self.background_type}")

    def add_peak(self, peak_id: str) -> None:
        """
        Attach a peak to the region.

        Parameters
        ----------
        peak_id : str
            ID of the peak to add.
        """
        self.peaks.append(peak_id)

    def remove_peak(self, peak_id: str) -> None:
        """
        Remove a peak from the region.

        Parameters
        ----------
        peak_id : str
            ID of the peak to remove.
        """
        self.peaks.remove(peak_id)

    def update_range(
        self,
        x: NDArray,
        y: NDArray,
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
        i_1 : Optional[float], default=None
            Background intensity at the start of the region.
        i_2 : Optional[float], default=None
            Background intensity at the end of the region.
        """

        self.x = x
        self.y = y

        if i_1 is not None:
            self.i_1 = i_1
        if i_2 is not None:
            self.i_2 = i_2

    def __repr__(self) -> str:
        s_id = self.spectrum_id[:8] if self.spectrum_id is not None else "N/A"
        return (
            f"<Region id={self.id[:8]} spectrum_id={s_id} "
            f"points={self.x.size if self.x is not None else 0} "
            f"peaks={len(self.peaks)} background_type={self.background_type}>"
        )


@dataclass
class Spectrum:
    """
    Spectrum data container with optional processed attributes.

    A :class:`Spectrum` stores raw spectral data (`x`, `y`) and may include
    processed results such as normalization coefficients, smoothing, and
    identified regions. Supports charge correction and region management.

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
        Unique identifier for the spectrum (format ``s<uuid>``).
    regions : list[str]
        List of region IDs associated with this spectrum.
    charge_correction : float
        Applied shift to the energy axis.
    norm_coefs : Tuple[float, float]
        Minimum and maximum values used for normalization.
    y_smoothed : Optional[NDArray]
        Smoothed intensity values (e.g., via Savitzky-Golay filter).
    """

    x: NDArray
    y: NDArray
    name: Optional[str] = None
    file: Optional[str] = None
    group: Optional[str] = None

    id: str = field(default_factory=lambda: f"s{uuid4().hex}")
    regions: List[str] = field(default_factory=list)  # region IDs
    charge_correction: float = 0.0

    # Optional fields for processed data
    y_smoothed: Optional[NDArray] = None

    @property
    def norm_coefs(self) -> Tuple[float, float]:
        return self.y.min(), self.y.max()

    def add_region(self, region_id) -> None:
        """Attach region to spectrum."""
        self.regions.append(region_id)

    def remove_region(self, region_id: str) -> None:
        """
        Remove a region from the spectrum.

        Parameters
        ----------
        region_id : str
            ID of the region to remove.
        """
        self.regions.remove(region_id)

    def create_region(self, start_idx: int, end_idx: int, background_type: str = "shirley") -> Region:
        """
        Create a region from a subset of the spectrum data and attach it.

        Parameters
        ----------
        start_idx : int
            Start index of the region in the spectrum arrays.
        end_idx : int
            End index of the region in the spectrum arrays.
        background_type : str, default="shirley"
            Type of background for the region.

        Returns
        -------
        Region
            The newly created region attached to this spectrum.
        """
        if self.y_smoothed is None:
            i_1 = self.y[start_idx]
            i_2 = self.y[end_idx]
        else:
            i_1 = self.y_smoothed[start_idx]
            i_2 = self.y_smoothed[end_idx]

        region = Region(
            spectrum_id=self.id,
            x=self.x[start_idx:end_idx],
            y=self.y[start_idx:end_idx],
            norm_coefs=self.norm_coefs,
            i_1=i_1,
            i_2=i_2,
            background_type=background_type,
        )
        self.add_region(region)
        return region

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
            f"Smoothed: {'Yes' if self.y_smoothed is not None else 'No'}",
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

    A :class:`SpectrumCollection` maintains lookup tables (indexes) that map
    object IDs to their corresponding objects. This enables retrieval of
    any registered :class:`Spectrum`, :class:`Region`, or :class:`Peak`
    by its UUID.

    Notes
    -----
    - All objects (`Spectrum`, `Region`, `Peak`) must have an `id` attribute
      that is unique within the collection.
    - Adding a spectrum via :meth:`register` automatically registers its regions
      and peaks.
    - Removing objects updates parent-child relationships to keep the collection
      consistent.

    Attributes
    ----------
    peaks_index : dict[str, Peak]
        Mapping of peak UUIDs to peak objects.
    regions_index : dict[str, Region]
        Mapping of region UUIDs to region objects.
    spectra_index : dict[str, Spectrum]
        Mapping of spectrum UUIDs to spectrum objects.
    """

    def __init__(self):
        self.peaks_index: Dict[str, Peak] = {}
        self.regions_index: Dict[str, Region] = {}
        self.spectra_index: Dict[str, Spectrum] = {}

    def add_link(self, parent: Spectrum | Region | str, child: Region | Peak) -> None:
        """
        Establish a parent-child relationship between objects.

        Parameters
        ----------
        parent : Spectrum | Region | str
            Parent object or its UUID.
        child : Region | Peak
            Child object to link to the parent.
        """
        if isinstance(parent, str):
            parent_id = parent
        else:
            parent_id = parent.id

        if parent_id.startswith("s"):  # parents id must be in collection
            parent = self.get_spectrum(parent_id)
        elif parent_id.startswith("r"):
            parent = self.get_region(parent_id)

        if isinstance(parent, Spectrum) and isinstance(child, Region):
            child.spectrum_id = parent_id
            parent.add_region(child.id)  # attach child to parent
            self.register(child)  # attach child to collection
        elif isinstance(parent, Region) and isinstance(child, Peak):
            child.region_id = parent_id
            parent.add_peak(child.id)
            self.register(child)
        else:
            raise TypeError(
                "Given parent and child objects must be compatible"
                f"but got: parent={parent}, child={child}"
            )

    def register(self, obj: Spectrum | Region | Peak) -> None:
        """
        Register an object in the collection.

        Parameters
        ----------
        obj : Spectrum or Region or Peak
            Object to register.
        """
        if isinstance(obj, Peak):
            self._register_peak(obj)
        elif isinstance(obj, Region):
            self._register_region(obj)
        elif isinstance(obj, Spectrum):
            self._register_spectrum(obj)

    def _register_peak(self, peak: Peak) -> None:
        """
        Add a peak to the collection.
        """
        self.peaks_index[peak.id] = peak

    def _register_region(self, region: Region) -> None:
        """
        Add a region and automatically register all its peaks.
        """
        self.regions_index[region.id] = region

    def _register_spectrum(self, spectrum: Spectrum) -> None:
        """
        Add a spectrum and automatically register all its regions and peaks.
        """
        self.spectra_index[spectrum.id] = spectrum

    def remove(self, obj: Spectrum | Region | Peak | str) -> None:
        """
        Remove an object from the collection.

        Parameters
        ----------
        obj : Spectrum or Region or Peak or str
            Object instance or its UUID.
        """
        if isinstance(obj, (Spectrum, Region, Peak)):
            obj = obj.id

        if obj.startswith("s"):
            self._remove_spectrum(obj)
        elif obj.startswith("r"):
            self._remove_region(obj)
        elif obj.startswith("p"):
            self._remove_peak(obj)

    def _remove_peak(self, peak_id: str) -> None:
        peak = self.get_peak(peak_id)
        parent_region_id = peak.region_id
        peak.region_id = None

        if parent_region_id is not None:
            self.get_region(parent_region_id).remove_peak(peak_id)

        self.peaks_index.pop(peak_id)

    def _remove_region(self, region_id: str) -> None:
        region = self.get_region(region_id)
        parent_spectrum_id = region.spectrum_id
        region.spectrum_id = None

        if parent_spectrum_id is not None:
            self.get_spectrum(parent_spectrum_id).remove_region(region_id)

        for peak_id in region.peaks:
            self._remove_peak(peak_id)

        self.regions_index.pop(region_id)

    def _remove_spectrum(self, spectrum_id: str) -> None:
        spectrum = self.get_spectrum(spectrum_id)

        for region_id in spectrum.regions:
            self._remove_region(region_id)

        self.spectra_index.pop(spectrum_id)

    def get_spectrum(self, id: str) -> Spectrum:
        """
        Retrieve a spectrum by its UUID.
        """
        if id not in self.spectra_index:
            raise KeyError(f"Spectrum with ID {id} not found in collection.")

        return self.spectra_index[id]

    def get_region(self, id: str) -> Region:
        """
        Retrieve a region by its UUID.
        """
        if id not in self.regions_index:
            raise KeyError(f"Region with ID {id} not found in collection.")

        return self.regions_index[id]

    def get_peak(self, id: str) -> Peak:
        """
        Retrieve a peak by its UUID.
        """
        if id not in self.peaks_index:
            raise KeyError(f"Peak with ID {id} not found in collection.")

        return self.peaks_index[id]
