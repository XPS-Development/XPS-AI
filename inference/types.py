"""
NN pipeline input/output types and contracts.

Pipeline input is SpectrumDTO (x, y) from core.dto; for the segmenter,
y is expected to be already normalized (e.g. from DataQueryService or SpectrumDTO
with normalize=True). Output types are domain-friendly for conversion to
CreateRegion / CreatePeak in the app layer.
"""

from dataclasses import dataclass
from typing import Protocol

from numpy.typing import NDArray

from core.dto import SpectrumDTO


@dataclass(frozen=True)
class RegionDetectionResult:
    """
    Result for one region: index bounds in x-space.

    Maps to CreateRegion(spectrum_id, start, stop).
    """

    start: int
    stop: int


@dataclass(frozen=True)
class PeakDetectionResult:
    """
    Result for one peak: model name and parameters.

    Maps to CreatePeak(region_id, model_name, parameters).
    """

    model_name: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class BackgroundDetectionResult:
    """
    Result for one background: model name and parameters.

    Maps to CreateBackground(region_id, model_name, parameters).
    """

    model_name: str
    parameters: dict[str, float]


# Model input/output are model-specific (e.g. dict of arrays for ONNX).
# Type aliases for clarity; adapters and pre/post processors use these.
ModelInputT = dict[str, NDArray]
ModelOutputT = dict[str, NDArray]


class PreprocessorProtocol(Protocol):
    """Callable protocol for preprocessors: SpectrumDTO -> model input."""

    def __call__(self, data: SpectrumDTO) -> ModelInputT:
        """Produce model input from spectrum-like data."""
        ...


class AdapterProtocol(Protocol):
    """Protocol for adapters: run inference on model input."""

    def run(self, model_input: ModelInputT) -> ModelOutputT:
        """Run inference; return raw model output (e.g. tensors / dict)."""
        ...


class PostprocessorProtocol(Protocol):
    """Protocol for postprocessors: raw output + kwargs -> pipeline result."""

    def __call__(self, model_output: ModelOutputT, **kwargs) -> object:
        """Convert model output to result type using kwargs."""
        ...
