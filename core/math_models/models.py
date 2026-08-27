"""Concrete peak and background models registered with :class:`ModelRegistry`."""

from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from .base_models import BaseBackgroundModel, BasePeakModel, ParameterSpec
from .guess_helpers import amp_from_height, edge_intensities, half_max_sigma
from .model_funcs import linear_background, pvoigt, static_shirley_background


def _bg_slice_kwargs(
    kwargs: dict[str, float | int | str],
) -> tuple[int | float, int | float, Literal["value", "index"], int]:
    """Extract background slice arguments from ``guess_initial`` kwargs."""
    if "start" not in kwargs or "stop" not in kwargs:
        raise TypeError("background guess_initial requires 'start' and 'stop'")
    start = cast(int | float, kwargs["start"])
    stop = cast(int | float, kwargs["stop"])
    mode_raw = kwargs.get("mode", "index")
    if mode_raw not in ("value", "index"):
        raise ValueError(f"mode must be 'value' or 'index', got {mode_raw!r}")
    mode = cast(Literal["value", "index"], mode_raw)
    avg_on = int(kwargs.get("avg_on", 3))
    return start, stop, mode, avg_on


class PseudoVoigtPeakModel(BasePeakModel):
    """Pseudo-Voigt peak: linear mix of Gaussian and Lorentzian profiles."""

    name = "pseudo-voigt"
    parameter_schema = (
        ParameterSpec(name="amp", default=1, lower=0),
        ParameterSpec(name="cen", default=0),
        ParameterSpec(name="sig", default=1, lower=0),
        ParameterSpec(name="frac", default=1, lower=0, upper=1),
    )
    normalization_target_parameters = ("amp",)
    use_scale = True
    use_offset = False

    @staticmethod
    def evaluate(
        x: NDArray,
        y: NDArray | None,
        **kwargs: float,
    ) -> NDArray:
        """Return the pseudo-Voigt intensity at ``x``."""
        # Accept arbitrary keyword args for protocol compatibility.
        amp = kwargs["amp"]
        cen = kwargs["cen"]
        sig = kwargs["sig"]
        frac = kwargs["frac"]
        return pvoigt(x, amp, cen, sig, frac)

    @staticmethod
    def guess_initial(x: NDArray, y: NDArray, **kwargs: float | int | str) -> dict[str, float]:
        """Guess amp/cen/sig/frac at ``peak_index`` (optional ``frac``, default 0.5)."""
        if "peak_index" not in kwargs:
            raise TypeError("pseudo-voigt guess_initial requires 'peak_index'")
        peak_index = int(kwargs["peak_index"])
        frac = float(kwargs.get("frac", 0.5))
        sig = half_max_sigma(x, y, peak_index)
        amp = amp_from_height(y, peak_index, sig, frac)
        return {"amp": amp, "cen": float(x[peak_index]), "sig": sig, "frac": frac}


class ConstantBackgroundModel(BaseBackgroundModel):
    """Constant (flat) background."""

    name = "constant"
    parameter_schema = (ParameterSpec("const", 0.0, vary=False),)
    normalization_target_parameters = ("const",)
    use_scale = False
    use_offset = True

    @staticmethod
    def evaluate(x: NDArray, y: NDArray | None, **kwargs: float) -> NDArray:
        """Return a constant array filled with ``const``."""
        const = kwargs["const"]
        return np.full_like(x, fill_value=const)

    @staticmethod
    def guess_initial(x: NDArray, y: NDArray, **kwargs: float | int | str) -> dict[str, float]:
        """Guess ``const`` as the minimum of edge intensities at start/stop."""
        start, stop, mode, avg_on = _bg_slice_kwargs(kwargs)
        edges = edge_intensities(x, y, start, stop, mode=mode, avg_on=avg_on)
        return {"const": min(edges["i1"], edges["i2"])}


class LinearBackgroundModel(BaseBackgroundModel):
    """Linear background between endpoint intensities ``i1`` and ``i2``."""

    name = "linear"
    parameter_schema = (
        ParameterSpec("i1", 0.0, vary=False),
        ParameterSpec("i2", 0.0),
    )
    normalization_target_parameters = ("i1", "i2")
    use_scale = True
    use_offset = True

    @staticmethod
    def evaluate(x: NDArray, y: NDArray | None, **kwargs: float) -> NDArray:
        """Return a linear background spanning ``i1`` to ``i2``."""
        i1 = kwargs["i1"]
        i2 = kwargs["i2"]
        return linear_background(x, i1=i1, i2=i2)

    @staticmethod
    def guess_initial(x: NDArray, y: NDArray, **kwargs: float | int | str) -> dict[str, float]:
        """Guess ``i1`` / ``i2`` from edge intensities at start/stop."""
        start, stop, mode, avg_on = _bg_slice_kwargs(kwargs)
        return edge_intensities(x, y, start, stop, mode=mode, avg_on=avg_on)


class ShirleyBackgroundModel(BaseBackgroundModel):
    """Iterative Shirley background using endpoint intensities ``i1`` and ``i2``."""

    name = "shirley"
    parameter_schema = (
        ParameterSpec("i1", 0.0, vary=False),
        ParameterSpec("i2", 0.0, vary=False),
    )
    normalization_target_parameters = ("i1", "i2")
    use_scale = True
    use_offset = True

    @staticmethod
    def evaluate(x: NDArray, y: NDArray | None, **kwargs: float) -> NDArray:
        """Return an iterative Shirley background on ``x`` and ``y``."""
        i1 = kwargs["i1"]
        i2 = kwargs["i2"]
        if y is None:
            raise ValueError("Shirley background requires y reference signal")
        return static_shirley_background(x, y, i1=i1, i2=i2)

    @staticmethod
    def guess_initial(x: NDArray, y: NDArray, **kwargs: float | int | str) -> dict[str, float]:
        """Guess ``i1`` / ``i2`` from edge intensities at start/stop."""
        start, stop, mode, avg_on = _bg_slice_kwargs(kwargs)
        return edge_intensities(x, y, start, stop, mode=mode, avg_on=avg_on)
