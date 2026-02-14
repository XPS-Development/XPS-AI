"""
Strongly-typed metadata structures for core objects.

Defines immutable dataclasses for spectra, regions, and peaks.
Metadata is stored separately from core objects and keyed by object IDs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpectrumMetadata:
    """
    Metadata for a spectrum.

    Human-readable display information associated with a spectrum.

    Parameters
    ----------
    name : str
        Display name of the spectrum.
    group : str
        Group or category label.
    file : str
        Source file path or identifier.
    """

    name: str
    group: str
    file: str


@dataclass(frozen=True)
class RegionMetadata:
    """
    Metadata for a region.

    Empty placeholder for future extension.
    Regions currently have no metadata fields.
    """

    pass


@dataclass(frozen=True)
class PeakMetadata:
    """
    Metadata for a peak.

    Parameters
    ----------
    element_type : str
        Chemical element or peak type label.
    """

    element_type: str
