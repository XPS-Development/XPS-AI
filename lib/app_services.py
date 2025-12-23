import numpy as np

from .parametrics import NormalizationContext
from .domain import SpectrumCollection, Spectrum, Region, Peak, Background, Component
from .domain_services import CollectionQueryService, DataQueryService

from typing import Dict, Tuple
from numpy.typing import NDArray


class PlotDataService:
    """
    Application-level service for numerical evaluation of spectra, regions,
    and model components (peaks/backgrounds).
    """

    def __init__(self, collection: SpectrumCollection, *, normalized: bool = False):
        self._collection = collection
        self.normalized = normalized

        self._query = CollectionQueryService(collection)
        self._data = DataQueryService(collection)

    def use_normalized(self, flag: bool) -> None:
        self.normalized = flag

    def _get_region_xy(self, region_id: str) -> tuple[NDArray, NDArray]:
        return self._data.get_region_data(region_id, normalized=self.normalized)

    # def _get_norm_ctx_for_region(self, region_id: str) -> NormalizationContext | None:
    #     if not self.normalized:
    #         return None
    #     region = self._query.get(region_id)
    #     spectrum = self._query.get(region.parent_id)
    #     return spectrum.norm_ctx

    def _get_parameters(
        self, component: Component, norm_ctx: NormalizationContext | None
    ) -> dict[str, float]:
        params = component.parameters
        if norm_ctx is not None:
            params = component.model.normalize(params, norm_ctx)
        return {name: p.value for name, p in params.items()}

    def _component_y(self, component: Component) -> NDArray:
        x, y = self._get_region_xy(component.parent_id)

        if not self.normalized:
            norm_ctx = None
        else:
            norm_ctx = self._data.get_norm_ctx(region_id=component.parent_id)

        params = self._get_parameters(component, norm_ctx)

        return component.model.evaluate(x, y, **params)

    def region_xy(self, region_id: str) -> tuple[NDArray, NDArray]:
        return self._get_region_xy(region_id)

    def background_y(self, region_id: str) -> NDArray:
        bg = self._query.get_background(region_id)
        return self._component_y(bg)

    def peak_y(self, peak_id: str) -> NDArray:
        peak = self._query.get(peak_id)
        return self._component_y(peak)

    def peaks_y(self, region_id: str) -> dict[str, NDArray]:
        return {peak.id_: self._component_y(peak) for peak in self._query.get_peaks(region_id)}

    def peaks_sum_y(self, region_id: str) -> NDArray:
        x, _ = self._get_region_xy(region_id)
        y = np.zeros_like(x)

        for peak in self._query.get_peaks(region_id):
            y += self._component_y(peak)

        return y

    def model_y(self, region_id: str, *, include_background: bool = True) -> NDArray:
        y = self.peaks_sum_y(region_id)

        if include_background:
            y += self.background_y(region_id)

        return y

    def region_plot_bundle(self, region_id: str) -> dict:
        x, y = self.region_xy(region_id)

        peaks = self.peaks_y(region_id)
        background = self.background_y(region_id)
        model = self.model_y(region_id)

        return {
            "x": x,
            "y": y,
            "background": background,
            "peaks": peaks,  # dict[peak_id -> y]
            "model": model,
        }

    def spectrum_plot_bundles(self, spectrum_id: str) -> dict[str, dict]:
        """
        Return plot bundles for all regions in a spectrum.

        Returns
        -------
        dict[str, dict]
            region_id -> region_plot_bundle
        """
        x, y = self._data.get_spectrum_data(spectrum_id, normalized=self.normalized)
        result = dict(x=x, y=y)
        regions = self._query.get_regions(spectrum_id)
        result.update({region.id_: self.region_plot_bundle(region.id_) for region in regions})
        return result
