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
