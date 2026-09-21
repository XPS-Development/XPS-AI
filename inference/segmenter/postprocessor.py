"""
Postprocessor layer: raw model output -> domain-friendly result.

Postprocessors convert adapter output (e.g. masks) into typed results such as
RegionBounds for use by the app layer (CreateRegion, CreatePeak).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.math_models import ModelRegistry
from core.numerics import recalculate_idx

from ..types import (
    BackgroundDetectionResult,
    ModelOutputT,
    PeakDetectionResult,
    RegionDetectionResult,
)
from .adapter import ONNXSegmenterAdapter


@dataclass(frozen=True)
class SegmenterResult:
    """Result for the segmenter: RegionDetectionResult and tuple of PeakDetectionResult."""

    region: RegionDetectionResult
    peaks: tuple[PeakDetectionResult, ...]
    background: BackgroundDetectionResult | None


class SegmenterPostprocessor:
    """
    Postprocessor for segmenter: smooth/restrict masks, find borders, map to original x.

    Converts raw peak_mask and max_mask into a list of RegionBounds (start, stop,
    peak_positions in x-space). Requires original x and interpolated x_int for index mapping.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        smooth: bool = True,
        window_length: int = 10,
        min_border_distance: int = 5,
        peak_model_name: str = "pseudo-voigt",
        background_model_name: str = "shirley",
    ) -> None:
        """Configure mask thresholding, smoothing, and default models.

        Parameters
        ----------
        threshold : float, optional
            Probability threshold for mask binarization (default 0.5).
        smooth : bool, optional
            Whether to smooth the peak mask before thresholding (default True).
        window_length : int, optional
            Window length for smoothing (default 10).
        min_border_distance : int, optional
            Minimum distance between borders to keep them separate (default 5).
        peak_model_name : str, optional
            Registered peak model for ``guess_initial``.
        background_model_name : str, optional
            Registered background model for ``guess_initial``.
        """
        self._threshold = threshold
        self._smooth = smooth
        self._window_length = window_length
        self._min_border_distance = min_border_distance
        self._peak_model_name = peak_model_name
        self._background_model_name = background_model_name

    def __call__(
        self,
        model_output: ModelOutputT,
        *,
        x: NDArray,
        x_int: NDArray,
        y: NDArray,
    ) -> list[SegmenterResult]:
        """
        Convert segmenter model output to list of RegionBounds.

        Parameters
        ----------
        model_output : ModelOutputT
            Dict with PEAK_MASK_KEY and MAX_MASK_KEY (1d arrays).
        x : NDArray
            Original spectrum x (for index mapping and peak positions in x-space).
        x_int : NDArray
            Interpolated x used during inference (same length as masks).

        Returns
        -------
        list[SegmenterResult]
            Each region prediction with its peak/background guesses.
        """
        region_raw = model_output.get(ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[0])
        max_raw = model_output.get(ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[1])
        if region_raw is None or max_raw is None:
            raise KeyError(
                f"Model output must contain {ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[0]!r} and {ONNXSegmenterAdapter.CHANNEL_MASK_KEYS[1]!r}"
            )
        region_mask, max_mask = self._restrict_mask(region_raw, max_raw)
        return self._get_parameters_from_masks(x, x_int, y, region_mask, max_mask)

    def _smooth_mask(self, mask: NDArray) -> NDArray:
        """Smooth mask using moving average."""
        kernel = np.ones(self._window_length) / self._window_length
        return np.convolve(mask, kernel, mode="same")

    def _restrict_mask(
        self, region_raw_mask: NDArray, max_raw_mask: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Binarize masks with optional smoothing on peak mask."""
        if self._smooth:
            region_mask = (self._smooth_mask(region_raw_mask) > self._threshold).astype(np.float64)
        else:
            region_mask = (region_raw_mask > self._threshold).astype(np.float64)
        max_mask = (max_raw_mask > self._threshold).astype(np.float64)
        return region_mask, max_mask

    def _find_borders(self, mask: NDArray) -> NDArray:
        """Return indices of mask borders (transitions 0->1 or 1->0)."""
        padded = np.pad(mask, (1, 1), mode="constant", constant_values=(0, 0))
        diff = np.diff(padded, append=0)
        return np.argwhere(np.abs(diff)).reshape(-1)

    def _prepare_max_mask(self, max_mask: NDArray) -> NDArray:
        """Return indices of medians (center) of each run in max_mask."""
        borders = self._find_borders(max_mask)
        medians = [(t + f) // 2 for f, t in zip(borders[0::2], borders[1::2], strict=False)]
        return np.array(medians)

    def _guess_peaks(
        self, x: NDArray, y: NDArray, max_idxs: NDArray
    ) -> tuple[PeakDetectionResult, ...]:
        """Guess peak parameters via the configured peak model."""
        peak_model = ModelRegistry.get(self._peak_model_name)
        return tuple(
            PeakDetectionResult(
                model_name=self._peak_model_name,
                parameters=peak_model.guess_initial(x, y, peak_index=int(idx)),
            )
            for idx in max_idxs
        )

    def _get_parameters_from_masks(
        self,
        x: NDArray,
        x_int: NDArray,
        y: NDArray,
        region_mask: NDArray,
        max_mask: NDArray,
    ) -> list[SegmenterResult]:
        """Build RegionBounds from borders and max positions, mapped to original indices."""
        region_borders = self._find_borders(region_mask)
        max_idxs = self._prepare_max_mask(max_mask)
        region_borders = np.array([recalculate_idx(int(i), x_int, x) for i in region_borders])
        max_idxs = np.array([recalculate_idx(int(i), x_int, x) for i in max_idxs])

        connected_region_borders: list[int] = []
        for b in region_borders:
            b_int = int(b)
            if len(connected_region_borders) == 0:
                connected_region_borders.append(b_int)
            elif b_int - connected_region_borders[-1] < self._min_border_distance:
                connected_region_borders.pop()
            else:
                connected_region_borders.append(b_int)

        bg_model = ModelRegistry.get(self._background_model_name)
        result: list[SegmenterResult] = []
        borders = np.array(connected_region_borders)
        for i in range(0, len(borders) - 1, 2):
            f, t = borders[i], borders[i + 1]
            local_max_idxs = max_idxs[(max_idxs > f) & (max_idxs < t)]
            if local_max_idxs.size != 0:
                reg = RegionDetectionResult(start=int(f), stop=int(t))
                peaks = self._guess_peaks(x, y, local_max_idxs)
                background = BackgroundDetectionResult(
                    model_name=self._background_model_name,
                    parameters=bg_model.guess_initial(x, y, start=int(f), stop=int(t)),
                )
                result.append(SegmenterResult(region=reg, peaks=peaks, background=background))
        return result
