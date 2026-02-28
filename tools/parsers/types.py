"""
Shared types for spectrum parsers.
"""

from dataclasses import dataclass

import numpy as np

from core.metadata import SpectrumMetadata

from numpy.typing import NDArray


@dataclass
class ParsedSpectrum:
    """
    Result of parsing a spectrum from a file.

    Parameters
    ----------
    x : NDArray[np.floating]
        X-axis values (e.g. binding energy).
    y : NDArray[np.floating]
        Y-axis intensity values.
    metadata : SpectrumMetadata
    """

    x: NDArray[np.floating]
    y: NDArray[np.floating]
    metadata: SpectrumMetadata

    def __post_init__(self) -> None:
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("x and y must be 1D arrays")
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")

        if self.x[0] > self.x[-1]:
            self.x = np.flip(self.x)
            self.y = np.flip(self.y)
