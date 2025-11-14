from pathlib import Path

import numpy as np

from lib.spectra import SpectrumCollection, Spectrum
from lib.parsers import VAMAS, SPECS

from io import TextIOWrapper
from typing import Tuple, List
from numpy.typing import NDArray


class RawReader:
    def __init__(self, collection: SpectrumCollection, x_type: str = "BE", y_type: str = "CPS"):
        self.collection = collection
        self.x_type = x_type.lower()  # "be" or "ke" (binding energy or kinetic energy)
        self.y_type = y_type.lower()  # "counts" or "cps" (counts per second)

    def add_to_collection(self, spectrum: Spectrum):
        self.collection.register(spectrum)

    def read_vms(self, path: Path):
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
        xcol = 0 if self.x_type == "ke" else 3
        ycol = 1 if self.y_type == "counts" else 4
        usecols = (xcol, ycol)
        data = np.loadtxt(file, delimiter="\t", skiprows=3, usecols=usecols, dtype=np.float32)
        x, y = data[:, 0], data[:, 1]
        return x, y

    def read_csvlike_text(self, file: TextIOWrapper) -> Tuple[NDArray, NDArray]:
        data = np.loadtxt(file, dtype=np.float32)
        x, y = data[:, 0], data[:, 1]
        return x, y

    def read_text(self, path: Path):
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

    def read_files(self, files: List[Path]):
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
