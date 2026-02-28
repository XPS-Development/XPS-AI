"""
NN pipeline input/output types and contracts.

Pipeline input is SpectrumLike (x, y) from core.types; for the segmenter,
y is expected to be already normalized (e.g. from DataQueryService or SpectrumDTO
with normalize=True). Output types are domain-friendly for conversion to
CreateRegion / CreatePeak in the app layer.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from numpy.typing import NDArray

from core.types import SpectrumLike


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
    Result for one peak: index bounds in x-space.

    Maps to CreatePeak(region_id, model_name, parameters).
    """

    model_name: str
    parameters: dict[str, float]


# Model input/output are model-specific (e.g. dict of arrays for ONNX).
# Type aliases for clarity; adapters and pre/post processors use these.
ModelInputT = dict[str, NDArray]
ModelOutputT = dict[str, NDArray]


class PreprocessorProtocol(Protocol):
    """Protocol for preprocessors: SpectrumLike -> model input."""

    def __call__(self, data: SpectrumLike) -> ModelInputT:
        """Produce model input from spectrum-like data."""
        ...


class AdapterProtocol(Protocol):
    """Protocol for adapters: run inference on model input."""

    def run(self, model_input: ModelInputT) -> ModelOutputT:
        """Run inference; return raw model output (e.g. tensors / dict)."""
        ...


class PostprocessorProtocol(Protocol):
    """Protocol for postprocessors: raw output + kwargs -> pipeline result."""

    def __call__(self, model_output: ModelOutputT, **kwargs: Any) -> Any:
        """Convert model output to result type using kwargs."""
        ...
