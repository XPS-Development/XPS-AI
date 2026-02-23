"""
Matplotlib-based viewer for spectra and region models.

Uses EvaluationService and protocol-typed objects (SpectrumLike, RegionLike,
ComponentLike) from core.types for structural typing without coupling to DTOs.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from core.types import ComponentLike, RegionLike, SpectrumLike

from .evaluation import EvaluationService


class ViewerDataProvider(Protocol):
    """
    Protocol for supplying spectrum/region/component data to the viewer.

    Implementations (e.g. DTOService plus query) provide protocol-typed
    objects so the viewer stays decoupled from concrete DTOs and domain services.
    """

    def get_spectrum(self, spectrum_id: str, *, normalized: bool = False) -> SpectrumLike:
        """Return a spectrum-like projection with .x and .y arrays."""
        ...

    def get_region(self, region_id: str, *, normalized: bool = False) -> RegionLike:
        """Return a region-like projection with .x and .y arrays."""
        ...

    def get_spectrum_repr(
        self, spectrum_id: str, *, normalized: bool = False
    ) -> tuple[SpectrumLike, tuple[tuple[RegionLike, tuple[ComponentLike, ...]], ...]]:
        """Return spectrum-like and its region-like and component-like objects for evaluation."""
        ...

    def get_region_repr(
        self, region_id: str, *, normalized: bool = False
    ) -> tuple[RegionLike, tuple[ComponentLike, ...]]:
        """Return region-like and its component-like objects for evaluation."""
        ...


class MatplotlibViewer:
    """
    High-level matplotlib viewer for debugging and exploration.

    Uses EvaluationService for numerical model evaluation and protocol-typed
    data (SpectrumLike, RegionLike, ComponentLike) for structural typing.

    plot_raw_*:
        Draw only raw numerical data (spectrum or region).

    plot_spectrum / plot_region:
        Draw full semantic visualization:
        - spectrum
        - regions
        - background
        - peaks (background + peak)
    """

    def __init__(
        self,
        evaluation_svc: EvaluationService,
        data_provider: ViewerDataProvider,
        *,
        normalized: bool = False,
    ) -> None:
        """
        Initialize the viewer.

        Parameters
        ----------
        evaluation_svc : EvaluationService
            Stateless service for evaluating region/spectrum bundles.
        data_provider : ViewerDataProvider
            Supplies spectrum-like, region-like, and component-like objects by id.
        normalized : bool, optional
            Whether to request normalized data and parameters from the provider.
        """
        self._eval = evaluation_svc
        self._provider = data_provider
        self._normalized = normalized

    # -------------------------------------------------
    # helpers
    # -------------------------------------------------

    def _get_ax(self, ax: Optional[Axes]) -> Axes:
        if ax is not None:
            return ax
        _, ax = plt.subplots()
        return ax

    # -------------------------------------------------
    # raw plotting
    # -------------------------------------------------

    def plot_raw_spectrum(
        self,
        spectrum_id: str,
        *,
        ax: Optional[Axes] = None,
        label: Optional[str] = None,
        **kwargs: object,
    ) -> Axes:
        """
        Plot raw x, y data for a spectrum. No models, regions, or components.
        """
        ax = self._get_ax(ax)
        spectrum = self._provider.get_spectrum(spectrum_id, normalized=self._normalized)
        ax.plot(spectrum.x, spectrum.y, label=label or spectrum_id, **kwargs)
        return ax

    def plot_raw_region(
        self,
        region_id: str,
        *,
        ax: Optional[Axes] = None,
        label: Optional[str] = None,
    ) -> Axes:
        """Plot raw x, y data for a region. No models or components."""
        ax = self._get_ax(ax)
        region, _ = self._provider.get_region_repr(region_id, normalized=self._normalized)
        ax.plot(region.x, region.y, label=label or region_id, color="k")
        return ax

    def plot_spectrum(
        self,
        spectrum_id: str,
        *,
        ax: Optional[Axes] = None,
        show_raw: bool = True,
        region_span_alpha: float = 0.15,
        plot_models: bool = True,
    ) -> Axes:
        """
        Draw full spectrum visualization: raw spectrum, region spans, and
        per-region components (background, peaks, model).

        Parameters
        ----------
        spectrum_id : str
            The identifier for the spectrum to plot.
        ax : Optional[Axes], optional
            The matplotlib Axes to plot on. If None, a new Axes will be created.
        show_raw : bool, optional
            Whether to display the raw spectrum data.
        region_span_alpha : float, optional
            Opacity of the region highlight spans.
        plot_models : bool, optional
            Whether to plot the model components for each region.

        Returns
        -------
        Axes
            The matplotlib Axes with the plotted spectrum.
        """
        ax = self._get_ax(ax)

        spectrum, regions = self._provider.get_spectrum_repr(spectrum_id, normalized=self._normalized)

        if show_raw:
            ax.plot(spectrum.x, spectrum.y, color="black")

        for region, _ in regions:
            ax.axvspan(region.x[0], region.x[-1], alpha=region_span_alpha)
            self.plot_region(region.id_, ax=ax, show_raw=False, plot_model=plot_models)

        return ax

    def plot_region(
        self,
        region_id: str,
        *,
        ax: Optional[Axes] = None,
        show_raw: bool = True,
        plot_model: bool = True,
    ) -> Axes:
        """
        Draw region visualization: raw y, background, peaks (background + peak),
        and full model.
        """
        ax = self._get_ax(ax)

        region, components = self._provider.get_region_repr(region_id, normalized=self._normalized)
        bundle = self._eval.region_bundle(region, components, include_background=True)

        x = bundle.x
        y = bundle.y
        bg_y = bundle.background.y if bundle.background else np.zeros_like(x)

        if show_raw:
            ax.plot(x, y, color="black", linewidth=1)

        ax.plot(x, bg_y, linestyle="--", label="background", color="k", alpha=0.8)

        for peak_res in bundle.peaks:
            ax.plot(
                x,
                bg_y + peak_res.y,
                label=f"peak {peak_res.id_}",
                alpha=1,
            )

        if plot_model:
            ax.plot(
                x,
                bundle.model,
                color="red",
                linewidth=1.5,
                label="model",
            )

        return ax
