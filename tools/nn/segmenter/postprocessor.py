"""
Postprocessor layer: raw model output -> domain-friendly result.

Postprocessors convert adapter output (e.g. masks) into typed results such as
RegionBounds for use by the app layer (CreateRegion, CreatePeak).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..types import (
    BackgroundDetectionResult,
    ModelOutputT,
    PeakDetectionResult,
    RegionDetectionResult,
)
from .adapter import ONNXSegmenterAdapter


@dataclass(frozen=True)
class SegmenterResult:
    """
    Result for the segmenter: RegionDetectionResult and tuple of PeakDetectionResult.
    """

    region: RegionDetectionResult
    peaks: tuple[PeakDetectionResult, ...]
    background: BackgroundDetectionResult | None


class SegmenterPostprocessor:
    """
    Postprocessor for segmenter: smooth/restrict masks, find borders, map to original x.

    Converts raw peak_mask and max_mask into a list of RegionBounds (start, stop,
    peak_positions in x-space). Requires original x and interpolated x_int for index mapping.
    """

    DEFAULT_PEAK_MODEL = "pseudo-voigt"
    DEFAULT_BACKGROUND_MODEL = "shirley"
    DEFAULT_FRACTION = 0.0

    def __init__(
        self,
        threshold: float = 0.5,
        smooth: bool = True,
        window_length: int = 10,
        min_border_distance: int = 5,
    ) -> None:
        """
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
        """
        self._threshold = threshold
        self._smooth = smooth
        self._window_length = window_length
        self._min_border_distance = min_border_distance

    def __call__(
        self,
        model_output: ModelOutputT,
        *,
        x: NDArray,
        x_int: NDArray,
        y: NDArray,
    ) -> SegmenterResult:
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
        list of RegionBounds
            Each with start, stop (original indices) and peak_positions (x values).
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

    def _restrict_mask(self, region_raw_mask: NDArray, max_raw_mask: NDArray) -> tuple[NDArray, NDArray]:
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
        medians = [(t + f) // 2 for f, t in zip(borders[0::2], borders[1::2])]
        return np.array(medians)

    def _recalculate_idx(self, idx: int, array_1: NDArray, array_2: NDArray) -> int:
        """Map index from interpolated grid (array_1) to original grid (array_2)."""
        if idx >= len(array_1):
            return len(array_2)
        val = array_1[idx]
        return int(np.abs(array_2 - val).argmin())

    def _get_sig(self, x: NDArray, y: NDArray, max_idx: int) -> float:
        """Sigma of the peak at max_idx."""
        half_max = (y[max_idx] - y.min()) / 2 + y.min()
        l_hm_idx = np.where(y[:max_idx] <= half_max)[0][-1]
        r_hm_idx = np.where(y[max_idx:] <= half_max)[0][0] + max_idx
        return (x[r_hm_idx] - x[l_hm_idx]) / 2

    def _get_amp(self, y: NDArray, max_idx: int, sig: float, frac: float) -> float:
        """Amplitude of the peak at max_idx."""
        shape_mult = frac / np.pi + (1 - frac) * np.sqrt(np.log(2) / np.pi)
        amp = (y[max_idx] - y.min()) * sig / shape_mult
        return amp

    def _get_peak_parameters(
        self, x: NDArray, y: NDArray, max_idxs: NDArray, frac: float = DEFAULT_FRACTION
    ) -> list[PeakDetectionResult]:
        """Make peak parameters from x and y values."""
        res = []
        for max_idx in max_idxs:
            sig = self._get_sig(x, y, max_idx)
            amp = self._get_amp(y, max_idx, sig, frac)
            res.append(
                PeakDetectionResult(
                    model_name=self.DEFAULT_PEAK_MODEL,
                    parameters={
                        "amp": float(amp),
                        "cen": float(x[max_idx]),
                        "sig": float(sig),
                        "frac": float(frac),
                    },
                )
            )
        return res

    def _get_background_parameters(self, y: NDArray, start: int, stop: int) -> BackgroundDetectionResult:
        """Get background parameters from x and y values."""
        return BackgroundDetectionResult(
            model_name=self.DEFAULT_BACKGROUND_MODEL,
            parameters={"i1": float(y[start]), "i2": float(y[stop])},
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
        region_borders = np.array([self._recalculate_idx(int(i), x_int, x) for i in region_borders])
        max_idxs = np.array([self._recalculate_idx(int(i), x_int, x) for i in max_idxs])

        connected_region_borders: list[int] = []
        for b in region_borders:
            b_int = int(b)
            if len(connected_region_borders) == 0:
                connected_region_borders.append(b_int)
            elif b_int - connected_region_borders[-1] < self._min_border_distance:
                connected_region_borders.pop()
            else:
                connected_region_borders.append(b_int)

        result: list[SegmenterResult] = []
        borders = np.array(connected_region_borders)
        for i in range(0, len(borders) - 1, 2):
            f, t = borders[i], borders[i + 1]
            local_max_idxs = max_idxs[(max_idxs > f) & (max_idxs < t)]
            if local_max_idxs.size != 0:
                reg = RegionDetectionResult(start=int(f), stop=int(t))
                peaks = self._get_peak_parameters(x, y, local_max_idxs)
                background = self._get_background_parameters(y, f, t)
                result.append(SegmenterResult(region=reg, peaks=tuple(peaks), background=background))
        return result
