from typing import Optional, Iterable, Tuple, Sequence
from numpy.typing import NDArray


import numpy as np
import onnxruntime as ort
import numpy as np

from lib.tools import interpolate, norm_with_coefs
from lib.spectra import SpectrumCollection, Spectrum, Region, Peak
from lib.optimization import OptimizationManager


class BaseModelProcessor:
    """
    Abstract base class for processing spectral data using a trained ONNX model.

    This class defines a generic workflow for applying an ONNX model to a sequence of
    `Spectrum` objects, organizing the model outputs into hierarchical relationships
    (Spectrum → Region → Peak), and storing those relationships in a `SpectrumCollection`.

    Subclasses should implement the `run()` method to define the model-specific
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
    """

    def __init__(self, model_path: str, collection: SpectrumCollection, *args, **kwargs) -> None:
        self.ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.collection = collection

    def __call__(self, spectra: Sequence[Spectrum]) -> None:
        """
        Process a batch of spectra using the model and update the collection with results.

        For each spectrum, this method calls the subclass-defined `run()` method,
        which should yield tuples of `(region, peaks)`. The method then establishes
        hierarchical links between each spectrum, its regions, and the peaks in the regions.

        Parameters
        ----------
        spectra : Sequence[Spectrum]
            A list or other iterable of `Spectrum` objects to process.

        Raises
        ------
        NotImplementedError
            If `run()` is not implemented in a subclass.

        Examples
        --------
        >>> processor = MyModelProcessor("model.onnx", collection)
        >>> processor(spectra)
        """
        for spectrum in spectra:
            for region, peaks in self.run(spectrum):
                self.collection.add_link(spectrum, region)
                for peak in peaks:
                    self.collection.add_link(region, peak)

    def run(self, spectrum: Spectrum) -> Sequence[Tuple[Region, Sequence[Peak]]]:
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
    """
    Default data processing pipeline for a segmentation-based spectral model using ONNX Runtime.

    This processor loads a trained ONNX segmentation model and applies it to
    spectra to identify regions and peaks. It can optionally smooth model
    predictions, threshold masks, and run a post-processing optimization step.

    Attributes
    ----------
    pred_threshold : float
        Probability threshold for deciding whether a pixel belongs to a region or peak.
    optimization_manger : Optional[OptimizationManager]
        Optional optimization manager to refine region results after inference.
    add_mask_smoothing : bool
        If True, smooths model output masks before thresholding.
    default_background : str
        Default background type used when creating new regions (e.g., "shirley").

    Parameters
    ----------
    model_path : str
        Path to the ONNX model file used for segmentation.
    collection : SpectrumCollection
        Collection for storing spectrum-region-peak relationships.
    pred_threshold : float
        Threshold for binary mask classification.
    optimization_manger : Optional[OptimizationManager], default=None
        Optional optimization manager applied after segmentation.
    add_mask_smoothing : bool, default=True
        Whether to smooth model masks before applying threshold.
    default_background : str, default="shirley"
        Default background method for generated regions.
    """

    def __init__(
        self,
        model_path: str,
        collection: SpectrumCollection,
        pred_threshold: float,
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

    def __call__(self, spectra: Sequence[Spectrum]) -> None:
        """
        Run segmentation on all spectra and apply optional optimization.

        Parameters
        ----------
        spectra : Sequence[Spectrum]
            Collection of `Spectrum` instances to process.

        Notes
        -----
        After calling `super().__call__`, this method optionally invokes
        the optimization manager to refine all regions.
        """
        super().__call__(spectra)

        # additional optimization after model processing
        if self.optimization_manger is None:
            return

        regions_to_opt = [region for s in spectra for region in s.regions]
        self.optimization_manger.proceed_regions_opt(regions_to_opt)

    def run(self, spectrum: Spectrum) -> Sequence[Tuple[Region, Sequence[Peak]]]:
        """
        Process a single spectrum through the model and return detected regions and peaks.

        Parameters
        ----------
        spectrum : Spectrum
            Spectrum to process.

        Returns
        -------
        output : Sequence[Tuple[Region, Sequence[Peak]]]
            List of `(region, peaks)` tuples derived from segmentation masks.
        """
        x, y = spectrum.x, spectrum.y
        y_norm = norm_with_coefs(y, spectrum.norm_coefs)
        x_int, y_int = interpolate(x, y_norm)

        peak_mask, max_mask = self.predict(y_int)
        return self.parse_results(x, x_int, peak_mask, max_mask, spectrum)

    def parse_results(
        self,
        x: NDArray[np.float32],
        x_int: NDArray[np.float32],
        peak_mask: NDArray[np.float32],
        max_mask: NDArray[np.float32],
        spectrum: Spectrum,
    ) -> Sequence[Tuple[Region, Sequence[Peak]]]:
        """
        Convert binary masks into Region and Peak objects.

        Parameters
        ----------
        x, x_int : NDArray[np.float32]
            Original and interpolated x-axis arrays.
        peak_mask, max_mask : NDArray[np.float32]
            Binary masks indicating peak and local maximum positions.
        spectrum : Spectrum
            Spectrum being processed.

        Returns
        -------
        output : Sequence[Tuple[Region, Sequence[Peak]]]
            Extracted regions and their corresponding peaks.
        """
        results = []
        for f, t, max_positions in self.parse_masks_to_regions(x, x_int, peak_mask, max_mask):
            region = spectrum.create_region(f, t, background_type=self.default_background)
            peaks = [Peak(cen=float(position)) for position in max_positions]
            results.append((region, peaks))
        return results

    def prepare_input(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Prepare normalized model input tensor from raw intensity values.

        Applies a logarithmic transform and normalization, then stacks
        the original and log-transformed data as input channels.

        Parameters
        ----------
        x : ndarray
            Raw input data.

        Returns
        -------
        prepared_input : ndarray
            Model-ready tensor of shape (1, 2, 256).
        """
        x_log = np.log(10 * x + 1)
        x_log = (x_log - x_log.min()) / (x_log.max() - x_log.min())
        x_inp = np.stack((x, x_log), axis=0).astype(np.float32)
        return x_inp[np.newaxis, :, :]

    def _predict(self, x: NDArray[np.float32]) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Run the model on prepared input and return raw peak and max masks.

        Parameters
        ----------
        x : ndarray
            Input spectrum data.

        Returns
        -------
        peak_mask, max_mask : Tuple[ndarray, ndarray]
            Raw (peak_mask, max_mask) arrays.
        """
        inp = {"l_x_": self.prepare_input(x)}
        out = self.ort_session.run(None, inp)[0]
        return out[0, 0, :], out[0, 1, :]

    def smooth_mask(self, mask: NDArray[np.float32], window_length: int = 10) -> NDArray[np.float32]:
        """
        Smooth binary mask using a moving average.

        Parameters
        ----------
        mask : ndarray
            Input mask array.
        window_length : int, default=10
            Window size for smoothing.

        Returns
        -------
        smoothed mask : ndarray
        """
        return np.convolve(mask, np.ones(window_length) / window_length, mode="same")

    def restrict_mask(
        self, peak_raw_mask: NDArray[np.float32], max_raw_mask: NDArray[np.float32]
    ) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """
        Apply smoothing and thresholding to raw model outputs.

        Parameters
        ----------
        peak_raw_mask, max_raw_mask : ndarray
            Raw model outputs for peak and max channels.

        Returns
        -------
        peak_mask, max_mask : Tuple[ndarray, ndarray]
            Thresholded boolean masks.
        """
        if self.add_mask_smoothing:
            peak_mask = self.smooth_mask(peak_raw_mask) > self.pred_threshold
        else:
            peak_mask = peak_raw_mask > self.pred_threshold

        max_mask = max_raw_mask > self.pred_threshold
        return peak_mask, max_mask

    def predict(self, data: NDArray[np.float32]) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """
        Perform full prediction pipeline including thresholding.

        Parameters
        ----------
        data : ndarray
            Input intensity data.

        Returns
        -------
        peak_mask, max_mask : Tuple[ndarray, ndarray]
            Binary (peak_mask, max_mask).
        """
        raw_peak, raw_max = self._predict(data)
        return self.restrict_mask(raw_peak, raw_max)

    def find_borders(self, mask: NDArray[np.bool_]) -> NDArray[np.int_]:
        """
        Identify start and end indices of True segments in a binary mask.

        Parameters
        ----------
        mask : ndarray
            Input mask array.

        Returns
        -------
        indices : ndarray
            Indices marking region borders.
        """
        mask = np.pad(mask, (1, 1), "constant", constant_values=(0, 0))
        diff = np.diff(mask.astype(np.int32))
        return np.argwhere(np.abs(diff)).reshape(-1)

    def prepare_max_mask(self, max_mask: NDArray[np.bool_]) -> NDArray[np.int_]:
        """
        Compute median indices of contiguous True segments in max mask.

        Parameters
        ----------
        max_mask : ndarray
            Boolean mask of detected maxima.

        Returns
        -------
        indices : ndarray
        """
        medians = []
        borders = self.find_borders(max_mask)
        for f, t in zip(borders[0::2], borders[1::2]):
            medians.append((t + f) // 2)
        return np.array(medians, dtype=int)

    def recalculate_idx(self, idx: int, array_1: NDArray[np.float32], array_2: NDArray[np.float32]) -> int:
        """
        Recalculate index from one axis to another based on closest x-position.

        Parameters
        ----------
        idx : int
            Index in array_1.
        array_1, array_2 : NDArray[np.float32]
            Source and target x-axis arrays.

        Returns
        -------
        idx : int
            Closest matching index in array_2.
        """
        if idx >= len(array_1):
            return len(array_2) - 1
        val = array_1[idx]
        return int(np.abs(array_2 - val).argmin())

    def parse_masks_to_regions(
        self,
        x: NDArray[np.float32],
        x_int: NDArray[np.float32],
        peak_mask: NDArray[np.bool_],
        max_mask: NDArray[np.bool_],
    ) -> Iterable[Tuple[int, int, NDArray[np.float32]]]:
        """
        Parse binary masks into start/end index pairs and peak positions.

        Parameters
        ----------
        x, x_int : ndarray
            Original and interpolated x-axis arrays.
        peak_mask, max_mask : ndarray
            Boolean masks from model output.

        Yields
        ------
        f_idx, t_idx, max_positions : Tuple[int, int, ndarray]
            (from_index, to_index, max_positions) for each detected region.
        """
        peak_borders = self.find_borders(peak_mask)
        max_idxs = self.prepare_max_mask(max_mask)

        # Recalculate to original x indices
        peak_borders = np.array([self.recalculate_idx(idx, x_int, x) for idx in peak_borders])
        max_idxs = np.array([self.recalculate_idx(idx, x_int, x) for idx in max_idxs])

        # Merge close borders
        connected_peak_borders = []
        for b in peak_borders:
            if not connected_peak_borders or b - connected_peak_borders[-1] >= 5:
                connected_peak_borders.append(b)
            else:
                connected_peak_borders.pop()
        connected_peak_borders = np.array(connected_peak_borders)

        # Split into regions
        for f, t in zip(connected_peak_borders[0::2], connected_peak_borders[1::2]):
            local_max_idxs = max_idxs[(max_idxs > f) & (max_idxs < t)]
            if local_max_idxs.size > 0:
                yield f, t, x[local_max_idxs]
