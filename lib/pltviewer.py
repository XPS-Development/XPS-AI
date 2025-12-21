import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional

from .app_services import PlotDataService
from .domain import Spectrum, Region


class MatplotlibViewer:
    """
    High-level matplotlib viewer for debugging and exploration.

    plot_raw:
        Draw only raw numerical data (spectrum or region).

    plot:
        Draw full semantic visualization:
        - spectrum
        - regions
        - background
        - peaks (background + peak)
    """

    def __init__(self, plot_data: PlotDataService):
        self.pltdata = plot_data
        self._query = plot_data._query
        self._data = plot_data._data

    # -------------------------------------------------
    # helpers
    # -------------------------------------------------

    def _get_ax(self, ax: Optional[Axes]) -> Axes:
        if ax is not None:
            return ax
        _, ax = plt.subplots()
        return ax

    def _is_spectrum(self, obj_id: str) -> bool:
        return isinstance(self._query.get(obj_id), Spectrum)

    def _is_region(self, obj_id: str) -> bool:
        return isinstance(self._query.get(obj_id), Region)

    # -------------------------------------------------
    # raw plotting
    # -------------------------------------------------

    def plot_raw_spectrum(
        self,
        obj_id: str,
        *,
        ax: Optional[Axes] = None,
        label: Optional[str] = None,
        **kwargs,
    ) -> Axes:
        """
        Plot raw x, y data for spectrum or region.
        No models, no regions, no components.
        """
        ax = self._get_ax(ax)

        x, y = self._data.get_spectrum_data(
            obj_id,
            normalized=self.pltdata.normalized,
        )

        ax.plot(x, y, label=label or obj_id, **kwargs)
        return ax

    def plot_raw_region(
        self,
        obj_id: str,
        *,
        ax: Optional[Axes] = None,
        label: Optional[str] = None,
    ):
        x, y = self.pltdata.region_xy(obj_id)
        ax.plot(x, y, label=label or obj_id, color="k")
        return ax

    def plot_spectrum(
        self,
        obj_id: str,
        *,
        ax: Optional[Axes] = None,
        show_raw: bool = True,
        region_alpha: float = 0.15,
    ) -> Axes:
        """
        Plot full semantic representation.

        Spectrum:
            - raw spectrum
            - regions (spans)
            - all region components

        Region:
            - raw region data
            - background
            - peaks (background + peak)
            - full model
        """
        ax = self._get_ax(ax)

        if self._is_spectrum(obj_id):
            self._plot_spectrum(obj_id, ax, show_raw, region_alpha)

        elif self._is_region(obj_id):
            self._plot_region(obj_id, ax, show_raw)

        else:
            raise ValueError(f"Object {obj_id} is not Spectrum or Region")

        return ax

    def plot_spectrum(
        self,
        spectrum_id: str,
        *,
        ax: Optional[Axes] = None,
        show_raw: bool = True,
        region_alpha: float = 0.15,
        plot_models: bool = True,
    ):
        ax = self._get_ax(ax)

        # spectrum data
        if show_raw:
            self.plot_raw_spectrum(spectrum_id, ax=ax, color="black")

        # regions
        regions = self._query.get_regions(spectrum_id)
        for region in regions:
            x, _ = self.pltdata.region_xy(region.id_)
            ax.axvspan(x[0], x[-1], alpha=region_alpha)

            # plot region components on top
            self.plot_region(region.id_, ax=ax, show_raw=False, plot_model=plot_models)

        return ax

    def plot_region(self, region_id: str, *, ax: Axes, show_raw: bool, plot_model: bool = True):
        ax = self._get_ax(ax)

        bundle = self.pltdata.region_plot_bundle(region_id)

        x = bundle["x"]
        y = bundle["y"]
        bg = bundle["background"]

        if show_raw:
            ax.plot(x, y, color="black", linewidth=1)

        # background
        ax.plot(x, bg, linestyle="--", label="background", color="k", alpha=0.8)

        # peaks (peak + background)
        for peak_id, peak_y in bundle["peaks"].items():
            ax.plot(
                x,
                bg + peak_y,
                label=f"peak {peak_id}",
                alpha=1,
            )

        # full model
        if plot_model:
            ax.plot(
                x,
                bundle["model"],
                color="red",
                linewidth=1.5,
                label="model",
            )

        return ax
