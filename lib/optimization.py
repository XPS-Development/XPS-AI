import re
from itertools import chain
from typing import Sequence, Tuple, List, Optional, Callable

import numpy as np
from numpy.typing import NDArray
from lmfit import Parameter, Parameters, minimize
from lmfit.minimizer import MinimizerResult

from .tools import batch, norm_with_coefs, denorm_with_coefs
from .funcs import ndpvoigt
from .spectra import SpectrumCollection, Region, Peak, PeakParameter


class Optimizer:
    """Optimizer for multi-spectrum peak fitting.

    Parameters
    ----------
    x : NDArray | Sequence[NDArray]
        X-data. Either a single 1D array or a sequence of 1D arrays.
    y : NDArray | Sequence[NDArray]
        Y-data, aligned with x.
    peaks_parameters : Parameters
        lmfit Parameters object, number of parameters must be divisible by 4.
    combinations : tuple[tuple[int, ...]], optional
        Defines which peaks belong to each spectrum.
        If None, parameters are split uniformly.
    model : callable, default=ndpvoigt
        Model function: (params, x, combinations) -> list[NDArray]
    """

    def __init__(
        self,
        x: NDArray | Sequence[NDArray],
        y: NDArray | Sequence[NDArray],
        peaks_parameters: Parameters,
        combinations: Optional[Tuple[Tuple[str, ...], ...]] = None,
        model: Callable = None,
    ) -> None:

        self._validate_peaks(peaks_parameters)
        self.x, self.y = self._process_xy(x, y)
        self.combinations = self._process_combinations(combinations, len(self.x), len(peaks_parameters) // 4)

        self.init_params = peaks_parameters
        self.model = model if model is not None else ndpvoigt

    def _validate_peaks(self, params: Parameters) -> None:
        if len(params) % 4 != 0:
            raise ValueError(f"Expected number of parameters divisible by 4, got {len(params)}")

    def _process_xy(
        self,
        x: NDArray | Sequence[NDArray],
        y: NDArray | Sequence[NDArray],
    ) -> Tuple[Tuple[NDArray, ...], Tuple[NDArray, ...]]:

        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if len(x) != len(y):
                raise ValueError("x and y must have the same length.")
            return (x,), (y,)
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        return tuple(x), tuple(y)

    def _process_combinations(
        self,
        combinations: Optional[Tuple[Tuple[int, ...], ...]],
        n_spectra: int,
        n_peaks: int,
    ) -> Tuple[Tuple[int, ...], ...]:

        if combinations is None:  # split peaks uniformly
            peaks_per_spectrum = n_peaks // n_spectra
            return tuple(tuple(i) for i in batch(range(n_peaks), peaks_per_spectrum))
        if len(combinations) != n_spectra:
            raise ValueError("Number of combinations must match number of spectra.")
        if len(tuple(chain(*combinations))) != n_peaks:
            raise ValueError("Total peaks in combinations must match number of peaks.")
        return combinations

    # def _check_normalization(self, y: Sequence[NDArray], is_norm: bool) -> None:
    #     if is_norm and any(np.max(arr) > 1 for arr in y):
    #         raise ValueError("is_norm=True, but y contains values greater than 1.")

    def ndresid(
        self,
        params: Parameters,
        x: Sequence[NDArray],
        y: Sequence[NDArray],
        combinations: Tuple[Tuple[int, ...], ...],
    ) -> NDArray:
        """Calculate residuals for multiple spectra simultaneously."""
        y_true = np.concatenate(y).astype(float)
        y_model = np.concatenate(self.model(params, x, combinations)).astype(float)
        return y_model - y_true

    def fit(self, return_result: bool = False, **kwargs) -> Parameters | MinimizerResult:
        """Run optimization.

        Parameters
        ----------
        return_result : bool, default=False
            If True, returns full lmfit.MinimizerResult, else only params.
        kwargs : dict
            Extra arguments passed to lmfit.minimize.

        Returns
        -------
        Parameters or MinimizerResult
        """
        res = minimize(
            self.ndresid,
            self.init_params,
            args=(self.x, self.y, self.combinations),
            **kwargs,
        )
        return res if return_result else res.params

    def __repr__(self) -> str:
        return (
            f"Optimizer(n_spectra={len(self.x)}, "
            f"n_peaks={len(self.init_params)//4}, "
            f"norm={self.is_norm}, "
            f"model={self.model.__name__ if self.model else None})"
        )


class OptimizationManager:
    """
    Manager for preparing and performing optimization of peaks and regions.

    The `OptimizationManager` serves as a bridge between the spectral data
    (peaks and regions stored in a :class:`SpectrumCollection`) and the
    optimization routines. It is responsible for:

    - Converting `PeakParameter` objects into lmfit `Parameter` objects.
    - Handling normalization/denormalization of peak parameters.
    - Preparing optimization data for regions or groups of regions.
    - Updating peak values after optimization.
    """

    def __init__(self) -> None:
        self.collection: Optional[SpectrumCollection] = None
        self.def_id_pattern = r"p[\w]{1,32}"

    def set_collection(self, collection: SpectrumCollection) -> None:
        """
        Attach a spectrum collection to the manager.

        Parameters
        ----------
        collection : SpectrumCollection
            The spectrum collection containing spectra, regions, and peaks
            to be used for optimization.
        """
        self.collection = collection

    def proceed_query(
        self, region_ids: Optional[Sequence[str]] = None, peak_ids: Optional[Sequence[str]] = None
    ):
        """
        Define the scope of optimization query.

        Parameters
        ----------
        region_ids : sequence of str, optional
            List of region UUIDs to include in optimization.
        peak_ids : sequence of str, optional
            List of peak UUIDs to include in optimization.

        Notes
        -----
        Only one of `region_ids` or `peak_ids` should be provided.
        """
        if region_ids is not None and peak_ids is None:
            self.proceed_regions_opt(region_ids)
            # elif peak_ids is not None and peak_ids is None:
            #     self.procceed_peaks_opt(peak_ids)
        else:
            raise ValueError("Only one of region_ids or peak_ids should be provided.")

    def parse_expr(self, expr: str | None, param_name: str) -> str | None:
        """
        Replace parameter references in an expression with normalized names.

        Parameters
        ----------
        expr : str
            Original expression referencing peak IDs.
        param_name : str
            Name of the parameter (e.g., "amp", "cen", "sig", "frac").

        Returns
        -------
        str
            Updated expression with expanded parameter references.
        """
        if expr is None:
            return

        def replacer(match):
            return f"p{match.group(0)[1:]}_{param_name}"

        return re.sub(self.def_id_pattern, replacer, expr)

    def peakparam_to_param(
        self,
        peak_id: str,
        param: PeakParameter,
        norm_coefs: Optional[Tuple[float, float]] = None,
        force_fix: bool = False,
    ) -> Parameter:
        """
        Convert a PeakParameter into an lmfit Parameter.

        Parameters
        ----------
        peak_id : str
            Unique identifier of the peak.
        param : PeakParameter
            The parameter to convert.
        norm_coefs : tuple of float, optional
            Normalization coefficients (min, max). Only used for amplitude.
        force_fix : bool, default=False
            If True, override the `vary` flag and fix the parameter.

        Returns
        -------
        Parameter
            The corresponding lmfit parameter object.
        """
        norm_coefs = norm_coefs or (0, 1)
        name = f"{peak_id}_{param.name}"
        expr = self.parse_expr(param.expr, param.name)

        return Parameter(
            name,
            value=norm_with_coefs(param.value, norm_coefs),
            vary=param.vary if not force_fix else False,
            min=param.min,
            max=param.max,
            expr=expr,
        )

    @staticmethod
    def get_peak_params(peak: Peak) -> Tuple[PeakParameter, ...]:
        """
        Retrieve the four parameters of a peak.

        Parameters
        ----------
        peak : Peak
            Peak object.

        Returns
        -------
        tuple of PeakParameter
            (amp_par, cen_par, sig_par, frac_par)
        """
        return peak.amp_par, peak.cen_par, peak.sig_par, peak.frac_par

    def update_peak_param_values(self, parameters: Parameters, from_norm: bool = True) -> None:
        """
        Update peak parameter values from fitted lmfit Parameters.

        Parameters
        ----------
        parameters : Parameters
            Optimized lmfit parameters.
        from_norm : bool, default=True
            Whether to denormalize amplitude values using region coefficients.
        """
        for param_opt_name in parameters:
            idx, param = param_opt_name.split("_")
            peak = self.collection.get_peak(idx)
            opt_value = parameters[param_opt_name].value

            if param == "amp" and from_norm:
                region = self.collection.get_region(peak.region_id)
                norm_coefs = region.norm_coefs
                opt_value = denorm_with_coefs(opt_value, norm_coefs)

            peak.set(param, value=opt_value)

    def peak_to_params(self, peak: Peak, norm_coefs=None, force_fix: bool = False) -> List[Parameter]:
        """
        Convert a peak into a list of lmfit parameters.

        Parameters
        ----------
        peak : Peak
            Peak to convert.
        norm_coefs : tuple of float, optional
            Normalization coefficients for amplitude.
        force_fix : bool, default=False
            If True, fix all parameters (disable variation).

        Returns
        -------
        list of Parameter
            List of lmfit parameters for this peak.
        """
        amp, cen, sig, frac = self.get_peak_params(peak)
        params = []
        params.append(self.peakparam_to_param(peak.id, amp, norm_coefs, force_fix=force_fix))
        for par in (cen, sig, frac):
            params.append(self.peakparam_to_param(peak.id, par, force_fix=force_fix))

        return params

    def peaks_to_params(
        self, peaks: Sequence[Peak], norm_coefs=None, force_fix: bool = False
    ) -> List[Parameter]:
        """
        Convert multiple peaks into a flat list of lmfit parameters.

        Parameters
        ----------
        peaks : sequence of Peak
            List of peaks to convert.
        norm_coefs : tuple of float, optional
            Normalization coefficients for amplitude.
        force_fix : bool, default=False
            If True, fix all parameters (disable variation).

        Returns
        -------
        list of Parameter
            Combined list of lmfit parameters from all peaks.
        """
        params = []
        for peak in peaks:
            params.extend(self.peak_to_params(peak, norm_coefs, force_fix=force_fix))
        return params

    def prepare_region(
        self, region: Region, normalize: bool = True
    ) -> Tuple[NDArray, NDArray, Tuple[str, ...], List[Parameter]]:
        """
        Prepare data and parameters for optimizing a single region.

        Parameters
        ----------
        region : Region
            Region to optimize.
        normalize : bool, default=True
            Whether to use normalized intensity values.

        Returns
        -------
        tuple
            (x, y, reg_combinations, params)
            - x : ndarray
                X-axis values.
            - y : ndarray
                Y-axis values (normalized or raw).
            - reg_combinations : tuple of str
                Tuple of peak IDs in this region.
            - params : list of Parameter
                Parameters corresponding to peaks.
        """
        params = []
        reg_combinations = self.get_combinations(region.peaks)
        x = region.x
        peaks = region.peaks
        if normalize:
            norm_coefs = region.norm_coefs
            y = norm_with_coefs(region.y, norm_coefs)
        else:
            y = region.y
            norm_coefs = None

        params.extend(self.peaks_to_params(peaks, norm_coefs))
        return x, y, reg_combinations, params

    def resolve_dependencies(self, params: List[Parameter]) -> None:
        """
        Remove invalid parameter expressions that reference missing peaks.

        Parameters
        ----------
        params : list of Parameter
            List of lmfit parameters to check and update.
        """
        curr_ids = set()
        curr_ids.update(p.name for p in params)

        for p in params:
            if p.expr is None:
                continue
            peak_dep = re.match(self.def_id_pattern, p.expr)
            if peak_dep and peak_dep.group(0) not in curr_ids:
                p.expr = None

    def get_regions_opt(self, region_ids: Sequence[str], normalize: bool = True, default_model=ndpvoigt):
        """
        Build an Optimizer for the given region IDs.

        Parameters
        ----------
        region_ids : sequence of str
            UUIDs of regions to optimize.
        normalize : bool, default=True
            Whether to use normalized intensities.
        default_model : callable, default=ndpvoigt
            Model function used for optimization.

        Returns
        -------
        Optimizer
            Configured optimizer instance.
        """
        regions = [self.collection.get_region(region_id) for region_id in region_ids]
        x_data = []
        y_data = []
        combinations = []
        all_params = []
        for region in regions:
            x, y, reg_combinations, params = self.prepare_region(region, normalize=normalize)
            x_data.append(x)
            y_data.append(y)
            combinations.append(reg_combinations)
            all_params.extend(params)

        self.resolve_dependencies(all_params)
        params_to_opt = Parameters()
        params_to_opt.add_many(*all_params)
        opt = Optimizer(x_data, y_data, all_params, combinations, model=default_model)

        return opt

    # def get_peaks_opt(self, peak_ids: Sequence[str], normalize: bool = True, default_model=ndpvoigt):
    #     """
    #     Build an Optimizer for the given peak IDs.

    #     Parameters
    #     ----------
    #     peak_ids : sequence of str
    #         UUIDs of peaks to optimize.
    #     normalize : bool, default=True
    #         Whether to use normalized intensities.
    #     default_model : callable, default=ndpvoigt
    #         Model function used for optimization.

    #     Returns
    #     -------
    #     Optimizer
    #         Configured optimizer instance.
    #     """
    #     peaks_to_opt = [self.collection.get_peak(peak_id) for peak_id in peak_ids]
    #     regions = [self.collection.get_region(peak.region_id) for peak in peaks_to_opt]

    def proceed_regions_opt(
        self, region_ids: Sequence[str], normalize: bool = True, default_model=ndpvoigt
    ) -> None:
        """
        Run optimization for the given regions and update peak values.

        Parameters
        ----------
        region_ids : sequence of str
            UUIDs of regions to optimize.
        normalize : bool, default=True
            Whether to normalize amplitudes during optimization.
        default_model : callable, default=ndpvoigt
            Model function used for optimization.
        """
        opt = self.get_regions_opt(region_ids, normalize=normalize, default_model=default_model)
        opt_params = opt.fit(return_result=False)
        self.update_peak_param_values(opt_params, from_norm=normalize)

    # def procceed_peaks_opt(self, peak_ids: Sequence[str], normalize: bool = True, default_model=ndpvoigt) -> None:
    #     """
    #     Run optimization for the given peaks and update peak values.

    #     Parameters
    #     ----------
    #     peak_ids : sequence of str
    #         UUIDs of peaks to optimize.
    #     normalize : bool, default=True
    #         Whether to normalize amplitudes during optimization.
    #     default_model : callable, default=ndpvoigt
    #         Model function used for optimization.
    #     """
    #     opt = self.get_peaks_opt(peak_ids, normalize=normalize, default_model=default_model)
    #     opt_params = opt.fit(return_result=False)
    #     self.update_peak_param_values(opt_params, from_norm=normalize
