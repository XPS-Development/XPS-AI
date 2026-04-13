"""
NN adapter and pre/postprocessing pipelines.

Pipeline input is SpectrumLike (core.types); for the segmenter, y is expected
normalized. Output types (e.g. RegionBounds) are converted to CreateRegion /
CreatePeak in the app layer.
"""

from .pipeline import InferencePipeline, SegmenterPipeline

__all__ = [
    "InferencePipeline",
    "SegmenterPipeline",
]
