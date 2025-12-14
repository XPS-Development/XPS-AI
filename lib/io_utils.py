import datetime
import json
import gzip
from pathlib import Path

import numpy as np

from lib.spectra import SpectrumCollection, Spectrum, Region, Peak, PeakParameter
from lib.parsers import VAMAS, SPECS

from io import TextIOWrapper
from typing import Tuple, List, Optional
from numpy.typing import NDArray


DEFAULT_PEAK_PARAMS = ("amp", "cen", "sig", "frac")
ADDITIONAL_PEAK_PARAMS = ("position", "area", "fwhm", "gl")


class RawReader:
    """
    Reader and importer for raw spectral data files.

    `RawReader` provides a unified interface for reading spectra from several
    common XPS data formats (VAMAS `.vms`, CasaXPS text `.txt`, generic
    two-column text/CSV). Parsed spectra are automatically added to a
    `SpectrumCollection`.

    Parameters
    ----------
    collection : SpectrumCollection
        Target collection to which all loaded `Spectrum` objects will be added.
    x_type : {"BE", "KE"}, default="BE"
        Specifies which X-axis representation to load:
        - ``"BE"`` — binding energy
        - ``"KE"`` — kinetic energy
    y_type : {"CPS", "COUNTS"}, default="CPS"
        Specifies the Y-axis interpretation:
        - ``"Counts"`` — raw counts
        - ``"CPS"`` — normalized to counts per second

    Notes
    -----
    - The reader does *not* modify spectra after loading (no smoothing,
      background, or calibration).
    - All numeric arrays are converted to `np.float32`.
    - Unsupported formats (e.g. new SPECS XML) raise `NotImplementedError`.

    Examples
    --------
    >>> col = SpectrumCollection()
    >>> reader = RawReader(col, x_type="BE", y_type="CPS")
    >>> reader.read_files([Path("sample1.vms"), Path("spectrum.txt")])
    """

    def __init__(self, collection: SpectrumCollection, x_type: str = "BE", y_type: str = "CPS"):
        self.collection = collection
        self.x_type = x_type.lower()  # "be" or "ke" (binding energy or kinetic energy)
        self.y_type = y_type.lower()  # "counts" or "cps" (counts per second)

    def add_to_collection(self, spectrum: Spectrum) -> None:
        """
        Register a spectrum in the target collection.

        Parameters
        ----------
        spectrum : Spectrum
            Spectrum instance to register.
        """
        self.collection.register(spectrum)

    def read_vms(self, path: Path) -> None:
        """
        Read VAMAS (.vms) file and extract spectra.

        Parameters
        ----------
        path : Path
            Path to a `.vms` VAMAS file.

        Notes
        -----
        - For each block in the file, one `Spectrum` is created.
        - X-axis selection depends on `x_type` (BE/KE).
        - If `y_type="cps"`, the intensity is normalized by
          ``collection_time * num_scans``.
        - Spectrum metadata:
            * name  — taken from VAMAS block name
            * file  — file name
            * group — block sample name
        """
        obj = VAMAS(path)
        file_name = path.name
        for block in obj.blocks:
            group = block.sample  # group_name
            name = block.name  # spectrum_name

            if self.x_type == "ke":
                x = np.array(block.kinetic_axis, dtype=np.float32)
            elif self.x_type == "be":
                x = np.array(block.binding_axis, dtype=np.float32)
            y = np.array(block.data[0], dtype=np.float32)
            if self.y_type == "cps":
                y /= block.signal_collection_time * block.num_scans  # turn counts to counts-per-second

            spectrum = Spectrum(x, y, name=name, file=str(path.resolve()), group=group)
            self.add_to_collection(spectrum)

    def read_specs(self, path: Path):
        """
        Read SPECS XML file (not implemented).

        Parameters
        ----------
        path : Path
            Path to a `.xml` file.

        Raises
        ------
        NotImplementedError
            Always raised, functionality pending future support.
        """
        # obj = SPECS(path)
        # file_name = path.name
        # for g in obj.groups:
        #     group = g.name
        #     for r in g.regions:
        #         name = r.name
        #         x = r.binding_axis
        #         y = r.counts
        #         self.add_spectrum(x, y, name=name, file=file_name, group=group)
        raise NotImplementedError("New specs file format not supported yet.")

    def read_casa_text(self, file: TextIOWrapper) -> Tuple[NDArray, NDArray]:
        """
        Read CasaXPS-style ASCII file after the first line (spectrum name).

        Parameters
        ----------
        file : TextIOWrapper
            Open text file object positioned *after* the name line.

        Returns
        -------
        x : ndarray of float32
            X-axis values.
        y : ndarray of float32
            Y-axis values.

        Notes
        -----
        - CasaXPS typically stores 4 columns (plusone extra empty column when splitting with tabs);
        only relevant columns are read:
          * kinetic energy     → col 0
          * counts             → col 1
          * binding energy     → col 3
          * cps                → col 4

        - Column choice depends on `x_type` and `y_type`.
        """
        xcol = 0 if self.x_type == "ke" else 3
        ycol = 1 if self.y_type == "counts" else 4
        usecols = (xcol, ycol)
        data = np.loadtxt(file, delimiter="\t", skiprows=3, usecols=usecols, dtype=np.float32)
        x, y = data[:, 0], data[:, 1]
        return x, y

    def read_csvlike_text(self, file: TextIOWrapper) -> Tuple[NDArray, NDArray]:
        """
        Read a plain two-column text/CSV file with X and Y values.

        Parameters
        ----------
        file : TextIOWrapper
            Open text file object.

        Returns
        -------
        x : ndarray of float32
        y : ndarray of float32

        Notes
        -----
        Assumes two whitespace-separated numeric columns. No header is expected.
        """
        data = np.loadtxt(file, dtype=np.float32)
        x, y = data[:, 0], data[:, 1]
        return x, y

    def read_text(self, path: Path) -> None:
        """
        Read text-based spectral data (either CasaXPS or generic two-column).

        Parameters
        ----------
        path : Path
            Path to `.txt`, `.dat`, or `.csv` file.

        Logic
        -----
        - If the first line contains a *single* token -> treat as CasaXPS file.
        - If it contains *two* numbers -> treat as generic X/Y data.
        - Otherwise -> raise ``ValueError``.

        Creates a `Spectrum` with:
        - name — file name (generic) or the first line (CasaXPS)
        - file — file name
        """
        with path.open("r") as f:
            first_line = f.readline().split()
            # consider as a file from CasaXPS
            if len(first_line) == 1:
                name = first_line[0]  # spectrum name should be the first line
                x, y = self.read_casa_text(f)
            # consider as a file with two columns x and y
            elif len(first_line) == 2:
                name = path.stem  # spectrum name should be the file name
                x, y = self.read_csvlike_text(f)
            else:
                raise ValueError(f"Unknown file format: {path.name}")

            spectrum = Spectrum(x, y, name=name, file=str(path.resolve()))
            self.add_to_collection(spectrum)

    def read_files(self, files: List[Path]) -> None:
        """
        Read multiple files and automatically detect format.

        Parameters
        ----------
        files : list of Path
            Paths to input files.

        Raises
        ------
        ValueError
            If a file does not exist or its extension is unknown.

        Supported extensions
        --------------------
        - `.txt`, `.dat`, `.csv` - text-based
        - `.vms`                 - VAMAS
        - `.xml`                 - (unsupported SPECS)
        """
        for file in files:
            if not file.exists():
                raise ValueError(f"File {file} does not exist.")
            suffix = file.suffix.lower()
            if suffix == ".txt" or suffix == ".dat" or suffix == ".csv":
                self.read_text(file)
            elif suffix == ".vms":
                self.read_vms(file)
            elif suffix == ".xml":
                self.read_specs(file)
            else:
                raise ValueError(f"Unknown file extension: {file.name}")


#################################################################################
# Sample JSON
#################################################################################
# {
#     "spectra":
#     {
#         <spectrum_id>:
#         {
#             "energy": {"start": <start>, "step": <step>, "num_points": <end>},
#             "intensity": <y>,
#             "charge_correction": <charge_correction>,
#             "name": <name>,
#             "file": <file>,
#             "group": <group>,
#             "regions": {
#                 <region_id>:
#                 {
#                     "start_idx": <start>,
#                     "end_idx": <end>,
#                     "background_type": <background_type>,
#                     "peaks": [<peak_id>, ...]
#                 }
#             },
#             "peaks": {
#                 <peak_id>:
#                 {
#                     "amp": {"value": <value>, "min": <min>, "max": <max>, "vary": <vary>, "expr": <expr>},
#                     "cen": ...,
#                     "sig": ...,
#                     "frac": ....,
#                     "position": <cen_value>,
#                     "area": <amp_value>,
#                     "fwhm": <sig_value * 2>,
#                     "gl": <frac_value * 100>
#                 }
#             }
#         }
#     }
# }
################################################################################


class SpectrumCollectionIO:
    """
    Input/output helper for serializing and deserializing SpectrumCollection objects.

    This class provides utilities for converting a :class:`SpectrumCollection`
    into a JSON-serializable dictionary and restoring it back. Both plain JSON
    and gzip-compressed JSON formats are supported.

    The serialization format is designed to preserve:
    - spectral energy axis (via start/step/number of points),
    - intensity data,
    - charge correction,
    - regions and their background types,
    - peak parameters and constraints.

    Notes
    -----
    - Internal helper methods (prefixed with `_`) are not part of the public API.
    - The `load` method is tolerant to partially incompatible files and may
      return ``None`` if deserialization fails.
    """

    def __init__(self, default_folder: Optional[Path | str] = None, *args, **kwargs):
        """
        Initialize the IO helper.

        Parameters
        ----------
        default_folder : Path or str, optional
            Folder where serialized files will be written by default.
            If None, the current working directory is used.
        *args, **kwargs
            Reserved for future extensions.
        """
        if default_folder is None:
            default_folder = Path.cwd()
        if not isinstance(default_folder, Path):
            default_folder = Path(default_folder)
        self.default_folder = default_folder

    def _param_to_dict(self, param: PeakParameter) -> dict:
        """
        Convert a PeakParameter into a JSON-serializable dictionary.

        Parameters
        ----------
        param : PeakParameter
            Peak parameter to serialize.

        Returns
        -------
        dict
            Dictionary containing parameter attributes:
            ``value``, ``min``, ``max``, ``vary`` and ``expr``.
        """
        return {
            "value": param.value,
            "min": param.min,
            "max": param.max,
            "vary": param.vary,
            "expr": param.expr,
        }

    def _param_from_dict(self, param: PeakParameter, param_dict: dict):
        """
        Restore a PeakParameter from a serialized dictionary.

        Parameters
        ----------
        param : PeakParameter
            Target parameter object to modify in-place.
        param_dict : dict
            Dictionary produced by :meth:`_param_to_dict`.

        Notes
        -----
        All attributes of the parameter are overwritten.
        """
        param.value = param_dict["value"]
        param.min = param_dict["min"]
        param.max = param_dict["max"]
        param.vary = param_dict["vary"]
        param.expr = param_dict["expr"]

    def _serialize_collection(self, collection: SpectrumCollection) -> dict:
        """
        Serialize a SpectrumCollection into a JSON-compatible dictionary.

        Parameters
        ----------
        collection : SpectrumCollection
            Collection containing spectra, regions, and peaks.

        Returns
        -------
        dict
            Dictionary representation of the collection suitable for
            JSON serialization.

        Notes
        -----
        - The energy axis is stored using ``start``, ``step`` and
          ``num_points`` instead of the full array.
        - Charge correction is temporarily removed before serialization
          and stored explicitly.
        - Peak parameters are stored in two equivalent representations:
          full parameter dictionaries and derived physical quantities
          (area, position, FWHM, GL fraction).
        """
        result = {"spectra": {}}

        for spectrum_id, spectrum in collection.spectra_index.items():

            # ENERGY
            charge_correction = spectrum.charge_correction
            spectrum.remove_charge_correction()
            x = spectrum.x
            energy_start = float(x[0])
            energy_step = float(x[1] - x[0]) if len(x) > 1 else 0.0
            energy_points = len(x)

            # REGIONS AND PEAKS
            regions_dict = {}
            peaks_dict = {}

            for region_id in spectrum.regions:
                region = collection.get(region_id)

                regions_dict[region_id] = {
                    "start_idx": region.start_idx,
                    "end_idx": region.end_idx,
                    "background_type": region.background_type,
                    "peaks": region.peaks[:],
                }

                for peak_id in region.peaks:
                    peak = collection.get(peak_id)

                    peaks_dict[peak_id] = {
                        "amp": self._param_to_dict(peak.amp_par),
                        "cen": self._param_to_dict(peak.cen_par),
                        "sig": self._param_to_dict(peak.sig_par),
                        "frac": self._param_to_dict(peak.frac_par),
                        "position": peak.cen_par.value,
                        "area": peak.amp_par.value,
                        "fwhm": peak.sig_par.value * 2,
                        "gl": peak.frac_par.value * 100,
                    }

            # MAKE FINAL DICT
            result["spectra"][spectrum_id] = {
                "energy": {"start": energy_start, "step": energy_step, "num_points": energy_points},
                "intensity": spectrum.y.tolist(),
                "charge_correction": charge_correction,
                "name": spectrum.name,
                "file": spectrum.file,
                "group": spectrum.group,
                "regions": regions_dict,
                "peaks": peaks_dict,
            }

        return result

    def _deserialize_peak(self, peak_id: str, peak_data: dict):
        """
        Create a Peak instance from serialized peak data.

        Parameters
        ----------
        peak_id : str
            Identifier of the peak.
        peak_data : dict
            Dictionary describing the peak parameters.

        Returns
        -------
        Peak
            Restored peak object.

        Raises
        ------
        KeyError
            If the dictionary does not contain a recognizable
            set of peak parameters.

        Notes
        -----
        Two formats are supported:
        1. Full PeakParameter dictionaries (``amp``, ``cen``, ``sig``, ``frac``).
        2. Derived physical quantities (``area``, ``position``, ``fwhm``, ``gl``).
        """
        # amp, cen, ... are stored in json
        if all((param in peak_data for param in DEFAULT_PEAK_PARAMS)):
            peak = Peak(id_=peak_id)
            for param in DEFAULT_PEAK_PARAMS:
                param_data = peak_data[param]
                # get amp_par, ...: PeakParameter object from empty peak to
                # fill it with parsed values
                param_obj: PeakParameter = getattr(peak, param + "_par")
                self._param_from_dict(param_obj, param_data)
            return peak
        # area, position, .. are stored in json
        elif all((param in peak_data for param in ADDITIONAL_PEAK_PARAMS)):
            peak = Peak(
                id_=peak_id,
                amp=peak_data["area"],
                cen=peak_data["position"],
                sig=peak_data["fwhm"] / 2,
                frac=peak_data["gl"] / 100,
            )
            return peak
        else:
            raise KeyError(f"Peak data has no keys to parse: {peak_data}")

    def _deserialize_collection(self, data: dict) -> SpectrumCollection:
        """
        Restore a SpectrumCollection from a serialized dictionary.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`_serialize_collection`
            or compatible external software.

        Returns
        -------
        SpectrumCollection
            Fully reconstructed collection with spectra, regions, and peaks.

        Raises
        ------
        KeyError
            If required keys (energy, intensity, ...) are missing.

        Notes
        -----
        - Supports both ``"energy"`` and legacy ``"BE"`` keys.
        - Supports both ``"intensity"`` and legacy ``"raw_intensity"`` keys.
        - If no regions are stored but peaks exist, a default region
          spanning the full spectrum is created.
        """
        collection = SpectrumCollection()

        spectra = data.get("spectra", {})

        for spectrum_id, spectrum_data in spectra.items():

            # RESTORE X
            energy = spectrum_data.get("energy", {}) | spectrum_data.get("BE", {})
            if not energy:
                raise KeyError(f"Spectrum {spectrum_id} has no 'energy' or 'BE' keys to parse")
            start = energy["start"]
            step = energy["step"]
            num = energy["num_points"]
            end = start + step * (num - 1)
            x = np.linspace(start, end, num, dtype=np.float32)

            # RESTORE Y
            y_data = spectrum_data.get("intensity", []) + spectrum_data.get("raw_intensity", [])
            if not y_data:
                raise KeyError(f"Spectrum {spectrum_id} has no 'intensity' or 'raw_intensity' keys to parse")
            y = np.array(y_data, dtype=np.float32)

            # SPECTRUM
            spectrum = Spectrum(
                x=x,
                y=y,
                name=spectrum_data.get("name"),
                file=spectrum_data.get("file"),
                group=spectrum_data.get("group"),
                id=spectrum_id,
            )
            charge_correction = spectrum_data.get("charge_correction", 0)
            spectrum.set_charge_correction(charge_correction)

            collection.register(spectrum)

            # REGIONS
            spectrum_regions = spectrum_data.get("regions", {})
            spectrum_peaks = spectrum_data.get("peaks", {})
            if spectrum_regions:
                for region_id, region_data in spectrum_regions.items():
                    region = spectrum.create_region(
                        start_idx=region_data["start_idx"],
                        end_idx=region_data["end_idx"],
                        background_type=region_data["background_type"],
                        region_id=region_id,
                    )
                    collection.add_link(spectrum, region)
                    # RELATED PEAKS
                    for peak_id in region_data["peaks"]:
                        peak_data = spectrum_peaks.get(peak_id, {})
                        peak = self._deserialize_peak(peak_id, peak_data)
                        collection.add_link(region, peak)
            # NO REGIONS BUT PEAKS ARE STORED
            elif spectrum_peaks:
                # assign region as whole spectrum
                region = spectrum.create_region(0, len(spectrum.x))
                collection.add_link(spectrum, region)
                # add all peaks to the region
                for peak_id, peak_data in spectrum_peaks.items():
                    peak = self._deserialize_peak(peak_id, peak_data)
                    collection.add_link(region, peak)

        return collection

    def dump(self, collection: SpectrumCollection) -> Path:
        """
        Serialize a SpectrumCollection to a JSON file.

        Parameters
        ----------
        collection : SpectrumCollection
            Collection to serialize.

        Returns
        -------
        Path
            Path to the created JSON file.

        Notes
        -----
        The filename is generated automatically using the current timestamp.
        """
        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = self.default_folder / f"saved_{current_date}.json"
        result = self._serialize_collection(collection)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        return path

    def dump_compressed(self, collection: SpectrumCollection) -> Path:
        """
        Serialize a SpectrumCollection to a gzip-compressed JSON file.

        Parameters
        ----------
        collection : SpectrumCollection
            Collection to serialize.

        Returns
        -------
        Path
            Path to the created ``.json.gz`` file.
        """
        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = self.default_folder / f"saved_{current_date}.json.gz"
        result = self._serialize_collection(collection)

        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(result, f)

        return path

    def load(self, path: Path) -> SpectrumCollection | None:
        """
        Load a SpectrumCollection from a JSON or compressed JSON file.

        Parameters
        ----------
        path : Path
            Path to a ``.json`` or ``.json.gz`` file.

        Returns
        -------
        SpectrumCollection or None
            Restored collection if deserialization succeeds,
            otherwise ``None``.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        KeyError
            If required parsing keys are missing.
        """
        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return self._deserialize_collection(data)
        elif path.suffix == ".json.gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
                return self._deserialize_collection(data)
        else:
            raise ValueError("Unsupported file format.")
