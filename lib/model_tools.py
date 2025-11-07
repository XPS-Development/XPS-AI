import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Sequence
import onnxruntime as ort
from .tools import interpolate, norm_with_coefs
import numpy as np
from numpy.typing import NDArray
from lib.spectra import SpectrumCollection, Spectrum, Region, Peak
from lib.optimization import OptimizationManager


import onnxruntime as ort
from typing import Sequence, Tuple, Any


class BaseModelProcessor:
    """
    Abstract base class for processing spectral data using a trained ONNX model.

    This class defines a generic workflow for applying an ONNX model to a sequence of
    `Spectrum` objects, organizing the model outputs into hierarchical relationships
    (Spectrum → Region → Peak), and storing those relationships in a `SpectrumCollection`.

    Subclasses should implement the `proceed()` method to define the model-specific
    inference and postprocessing logic.

    Attributes
    ----------
    ort_session : ort.InferenceSession
        ONNX Runtime session used to perform inference with the provided model.
    collection : SpectrumCollection
        Data structure for managing links between spectra, regions, and peaks.

    Parameters
    ----------
    model_path : str
        Path to the ONNX model file used for inference.
    collection : SpectrumCollection
        The data collection object where processed results will be stored.
    *args, **kwargs :
        Additional arguments reserved for subclasses (ignored by the base class).
    """

    def __init__(self, model_path: str, collection: SpectrumCollection, *args, **kwargs) -> None:
        self.ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.collection = collection

    def __call__(self, spectra: Sequence[Spectrum]) -> None:
        """
        Process a batch of spectra using the model and update the collection with results.

        For each spectrum, this method calls the subclass-defined `proceed()` method,
        which should yield tuples of `(region, peaks)`. The method then establishes
        hierarchical links between each spectrum, its regions, and the peaks in the regions.

        Parameters
        ----------
        spectra : Sequence[Spectrum]
            A list or other iterable of `Spectrum` objects to process.

        Raises
        ------
        NotImplementedError
            If `proceed()` is not implemented in a subclass.

        Examples
        --------
        >>> processor = MyModelProcessor("model.onnx", collection)
        >>> processor(spectra)
        """
        for spectrum in spectra:
            for region, peaks in self.proceed(spectrum):
                self.collection.add_link(spectrum, region)
                for peak in peaks:
                    self.collection.add_link(region, peak)

    def proceed(self, spectrum: Spectrum) -> Sequence[Tuple[Region, Sequence[Peak]]]:
        """
        Perform model inference and extract structured results for a single spectrum.

        This method must be implemented in subclasses. It should use the ONNX model
        session (`self.ort_session`) to perform inference on the given spectrum,
        then parse the model outputs into regions and peaks.

        Parameters
        ----------
        spectrum : Spectrum
            A single spectrum instance to process.

        Returns
        -------
        Sequence[Tuple[Region, Sequence[Peak]]]
            A sequence of tuples, where each tuple contains:
              - A `Region` object corresponding to a detected region.
              - A sequence of `Peak` objects belonging to that region.

        Example
        -------
        >>> [
        ...     (region1, [peak1, peak2]),
        ...     (region2, [peak3, peak4, peak5])
        ... ]
        """
        raise NotImplementedError("Subclasses must implement this method.")


class SegmenterModelProcessor(BaseModelProcessor):
    """Default data processing pipelines for segmenter model with ONNX runtime."""

    def __init__(
        self,
        model_path,
        collection,
        pred_threshold,
        optimization_manger: Optional[OptimizationManager] = None,
        add_mask_smoothing: bool = True,
        default_background: str = "shirley",
        *args,
        **kwargs
    ):
        super().__init__(model_path, collection, *args, **kwargs)
        self.pred_threshold = pred_threshold
        self.optimization_manger = optimization_manger
        self.add_mask_smoothing = add_mask_smoothing
        self.default_background = default_background

    def __call__(self, spectra):
        super().__call__(spectra)

        # additional optimization after model processing
        if self.optimization_manger is None:
            return

        regions_to_opt = []
        for s in spectra:
            regions_to_opt.extend(s.regions)

        self.optimization_manger.proceed_regions_opt(regions_to_opt)

    def proceed(self, spectrum: Spectrum) -> Sequence[Tuple[Region, Sequence[Peak]]]:
        x, y = spectrum.x, spectrum.y
        y_norm = norm_with_coefs(y, spectrum.norm_coefs)
        x_int, y_int = interpolate(x, y_norm)

        peak_mask, max_mask = self.predict(y_int, self.pred_threshold)
        return self.parse_results(x, x_int, peak_mask, max_mask, spectrum)

    # TODO: check init values for peaks
    def parse_results(
        self,
        x,
        x_int,
        peak_mask,
        max_mask,
        spectrum: Spectrum,
    ) -> Sequence[Tuple[Region, Sequence[Peak]]]:
        results = []
        for f, t, max_positions in self.parse_masks_to_regions(x, x_int, peak_mask, max_mask, spectrum):
            region = spectrum.create_region(f, t, background_type=self.default_background)
            peaks = [Peak(cen=position) for position in max_positions]
            results.append((region, peaks))
        return results

    def prepare_input(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Prepare input tensor for the model by normalizing and stacking the input data."""
        x_log = np.log(10 * x + 1)
        x_log = (x_log - x_log.min()) / (x_log.max() - x_log.min())
        x_inp = np.stack((x, x_log), axis=0, dtype=np.float32)
        return x_inp[np.newaxis, :, :]

    def _predict(self, x):
        """Add predicted masks to spectra."""
        inp = {"l_x_": self.prepare_input(x)}
        out = self.ort_session.run(None, inp)[0]

        peak_mask = out[0, 0, :]
        max_mask = out[0, 1, :]

        return peak_mask, max_mask

    def smooth_mask(self, mask, window_length=10):
        """Smooth mask using moving average."""
        return np.convolve(mask, np.ones(window_length) / window_length, mode="same")

    def restrict_mask(self, peak_raw_mask, max_raw_mask):
        """Restrict masks to the peaks with the highest probability."""

        if self.add_mask_smoothing:
            peak_mask = self.smooth_mask(peak_raw_mask) > self.pred_threshold
        else:
            peak_mask = peak_raw_mask > self.pred_threshold

        max_mask = max_raw_mask > self.pred_threshold
        return peak_mask, max_mask

    def predict(self, data):
        """Add predicted masks to spectra."""
        raw_peak, raw_max = self._predict(data)
        return self.restrict_mask(raw_peak, raw_max)

    def find_borders(self, mask):
        """Return idxs of borders in mask"""
        mask = np.pad(mask, (1, 1), "constant", constant_values=(0, 0))
        mask = np.diff(mask, append=0)
        return np.argwhere(np.abs(mask)).reshape(-1)

    def prepare_max_mask(self, max_mask):
        """Find idxs of medians in max_mask."""
        medians = []
        borders = self.find_borders(max_mask)
        for f, t in zip(borders[0::2], borders[1::2]):
            medians.append((t + f) // 2)
        return np.array(medians)

    def recalculate_idx(self, idx, array_1, array_2):
        if idx >= len(array_1):
            return len(array_2)
        val = array_1[idx]
        return (np.abs(array_2 - val)).argmin()

    def parse_masks_to_regions(self, x, x_int, peak_mask, max_mask):
        # find region borders in peak_mask
        peak_borders = self.find_borders(peak_mask)
        # find max idxs in max_mask
        max_idxs = self.prepare_max_mask(max_mask)
        # recaculate idxs to non interpolated data
        peak_borders = np.array([self.recalculate_idx(idx, x_int, x) for idx in peak_borders])
        max_idxs = np.array([self.recalculate_idx(idx, x_int, x) for idx in max_idxs])

        # delete close borders
        connected_peak_borders = []
        for b in peak_borders:
            if len(connected_peak_borders) == 0:
                connected_peak_borders.append(b)
            elif b - connected_peak_borders[-1] < 5:
                connected_peak_borders.pop()
            else:
                connected_peak_borders.append(b)
        connected_peak_borders = np.array(connected_peak_borders)

        # split borders to regions
        for f, t in zip(connected_peak_borders[0::2], connected_peak_borders[1::2]):
            # choose max_idxs in region
            local_max_idxs = max_idxs[(max_idxs > f) & (max_idxs < t)]

            if local_max_idxs.size != 0:
                max_positions = x[local_max_idxs]
                yield f, t, max_positions
            else:
                continue


class SegmenterModelProcessor:
    """Default data processing pipelines for segmenter model with ONNX runtime."""

    def __init__(self, model):
        self.ort_session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

    def prepare_input(self, x):
        """Prepare input tensor for the model by normalizing and stacking the input data."""
        x_log = np.log(10 * x + 1)
        x_log = (x_log - x_log.min()) / (x_log.max() - x_log.min())
        x_inp = np.stack((x, x_log), axis=0, dtype=np.float32)
        return x_inp[np.newaxis, :, :]

    def _predict(self, x):
        """Add predicted masks to spectra."""
        inp = {"l_x_": self.prepare_input(x)}
        out = self.ort_session.run(None, inp)[0]

        peak_mask = out[0, 0, :]
        max_mask = out[0, 1, :]

        return peak_mask, max_mask

    def smooth_mask(self, mask, window_length=10):
        """Smooth mask using moving average."""
        return np.convolve(mask, np.ones(window_length) / window_length, mode="same")

    def restrict_mask(self, peak_raw_mask, max_raw_mask, threshold=0.5, smooth=True):
        """Restrict masks to the peaks with the highest probability."""
        if smooth:
            peak_mask = self.smooth_mask(peak_raw_mask) > threshold
        else:
            peak_mask = peak_raw_mask > threshold
        max_mask = max_raw_mask > threshold
        return peak_mask, max_mask

    def predict(self, data, pred_threshold=0.5):
        """Add predicted masks to spectra."""
        raw_peak, raw_max = self._predict(data)
        return self.restrict_mask(raw_peak, raw_max, threshold=pred_threshold)

    def find_borders(self, mask):
        """Return idxs of borders in mask"""
        mask = np.pad(mask, (1, 1), "constant", constant_values=(0, 0))
        mask = np.diff(mask, append=0)
        return np.argwhere(np.abs(mask)).reshape(-1)

    def prepare_max_mask(self, max_mask):
        """Find idxs of medians in max_mask."""
        medians = []
        borders = self.find_borders(max_mask)
        for f, t in zip(borders[0::2], borders[1::2]):
            medians.append((t + f) // 2)
        return np.array(medians)

    def recalculate_idx(self, idx, array_1, array_2):
        if idx >= len(array_1):
            return len(array_2)
        val = array_1[idx]
        return (np.abs(array_2 - val)).argmin()

    def parse_masks_to_regions(self, x, x_int, peak_mask, max_mask):
        # find region borders in peak_mask
        peak_borders = self.find_borders(peak_mask)
        # find max idxs in max_mask
        max_idxs = self.prepare_max_mask(max_mask)
        # recaculate idxs to non interpolated data
        peak_borders = np.array([self.recalculate_idx(idx, x_int, x) for idx in peak_borders])
        max_idxs = np.array([self.recalculate_idx(idx, x_int, x) for idx in max_idxs])

        # delete close borders
        connected_peak_borders = []
        for b in peak_borders:
            if len(connected_peak_borders) == 0:
                connected_peak_borders.append(b)
            elif b - connected_peak_borders[-1] < 5:
                connected_peak_borders.pop()
            else:
                connected_peak_borders.append(b)
        connected_peak_borders = np.array(connected_peak_borders)

        # split borders to regions
        for f, t in zip(connected_peak_borders[0::2], connected_peak_borders[1::2]):
            # choose max_idxs in region
            local_max_idxs = max_idxs[(max_idxs > f) & (max_idxs < t)]

            if local_max_idxs.size != 0:
                max_positions = x[local_max_idxs]
                yield f, t, max_positions
            else:
                continue

    def run(self, spectrum, pred_threshold=0.5):
        x, y = spectrum.x, spectrum.y_norm
        x_int, y_int = interpolate(x, y)

        peak_mask, max_mask = self.predict(y_int, pred_threshold)
        yield from self.parse_masks_to_regions(x, x_int, peak_mask, max_mask)
