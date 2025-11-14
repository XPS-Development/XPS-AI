from pathlib import Path

import numpy as np

from lib.spectra import SpectrumCollection, Spectrum
from lib.parsers import VAMAS, SPECS

from io import TextIOWrapper
from typing import Tuple, List
from numpy.typing import NDArray


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

            spectrum = Spectrum(x, y, name=name, file=file_name, group=group)
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

            spectrum = Spectrum(x, y, name=name, file=path.name)
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
