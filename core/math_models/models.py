"""Concrete peak and background models registered with :class:`ModelRegistry`."""

import math
from collections.abc import Mapping
from typing import ClassVar, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .base_models import BaseBackgroundModel, BasePeakModel, ParameterSpec, ParametricModel
from .guess_helpers import amp_from_height, edge_intensities, half_max_sigma
from .model_funcs import (
    asym_pvoigt,
    linear_background,
    pvoigt,
    static_shirley_background,
    tail_pvoigt,
)
from .soft_ranges import (
    clip_soft_range,
    intensity_soft_range,
    prefer_finite_hard_bounds,
    soft_window_around,
)


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

    name: ClassVar[str] = "pseudo-voigt"
    parameter_schema: ClassVar[tuple[ParameterSpec, ...]] = (
        ParameterSpec(name="amp", default=1, lower=0),
        ParameterSpec(name="cen", default=0),
        ParameterSpec(name="sig", default=1, lower=0),
        ParameterSpec(name="frac", default=1, lower=0, upper=1),
    )
    normalization_target_parameters: ClassVar[tuple[str, ...]] = ("amp",)
    use_scale: ClassVar[bool] = True
    use_offset: ClassVar[bool] = False

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

    @staticmethod
    def soft_parameter_range(
        name: str,
        value: float,
        lower: float,
        upper: float,
        *,
        x_min: float | None = None,
        x_max: float | None = None,
        y_max: float | None = None,
    ) -> tuple[float, float]:
        """Soft slider ranges for amp/cen/sig/frac."""
        hard = prefer_finite_hard_bounds(lower, upper)
        if hard is not None:
            return hard
        key = name.lower()
        v = float(value) if math.isfinite(value) else 0.0
        if key == "frac":
            return (0.0, 1.0)
        if key == "cen":
            if x_min is not None and x_max is not None and x_max != x_min:
                lo, hi = (
                    (float(x_min), float(x_max)) if x_min < x_max else (float(x_max), float(x_min))
                )
                return (lo, hi)
            span = max(abs(v) * 0.1, 5.0)
            return soft_window_around(v, span=span, lower=lower, upper=upper)
        if key == "sig":
            return clip_soft_range(0.1, 30.0, lower, upper)
        if key == "amp":
            return intensity_soft_range(v, lower, upper, y_max=y_max, non_negative=True)
        return ParametricModel.soft_parameter_range(
            name, value, lower, upper, x_min=x_min, x_max=x_max, y_max=y_max
        )


class AsymPseudoVoigtPeakModel(BasePeakModel):
    """Pseudo-Voigt skewed by a log warp of the energy axis."""

    name: ClassVar[str] = "asym-pseudo-voigt"
    parameter_schema: ClassVar[tuple[ParameterSpec, ...]] = (
        ParameterSpec(name="amp", default=1, lower=0),
        ParameterSpec(name="cen", default=0),
        ParameterSpec(name="sig", default=1, lower=0),
        ParameterSpec(name="frac", default=1, lower=0, upper=1),
        ParameterSpec(name="asym", default=0, lower=-0.9, upper=0.9),
    )
    normalization_target_parameters: ClassVar[tuple[str, ...]] = ("amp",)
    use_scale: ClassVar[bool] = True
    use_offset: ClassVar[bool] = False

    @staticmethod
    def evaluate(
        x: NDArray,
        y: NDArray | None,
        **kwargs: float,
    ) -> NDArray:
        """Return the axis-warped pseudo-Voigt intensity at ``x``."""
        return asym_pvoigt(
            x,
            kwargs["amp"],
            kwargs["cen"],
            kwargs["sig"],
            kwargs["frac"],
            kwargs["asym"],
        )

    @staticmethod
    def guess_initial(x: NDArray, y: NDArray, **kwargs: float | int | str) -> dict[str, float]:
        """Guess a symmetric pseudo-Voigt and set ``asym`` to 0."""
        guessed = PseudoVoigtPeakModel.guess_initial(x, y, **kwargs)
        guessed["asym"] = 0.0
        return guessed

    @staticmethod
    def soft_parameter_range(
        name: str,
        value: float,
        lower: float,
        upper: float,
        *,
        x_min: float | None = None,
        x_max: float | None = None,
        y_max: float | None = None,
    ) -> tuple[float, float]:
        """Soft slider ranges; ``asym`` uses its hard bounds."""
        return PseudoVoigtPeakModel.soft_parameter_range(
            name, value, lower, upper, x_min=x_min, x_max=x_max, y_max=y_max
        )


class TailPseudoVoigtPeakModel(BasePeakModel):
    """Pseudo-Voigt with an exponential tail toward higher x."""

    name: ClassVar[str] = "tail-pseudo-voigt"
    parameter_schema: ClassVar[tuple[ParameterSpec, ...]] = (
        ParameterSpec(name="amp", default=1, lower=0),
        ParameterSpec(name="cen", default=0),
        ParameterSpec(name="sig", default=1, lower=0),
        ParameterSpec(name="frac", default=1, lower=0, upper=1),
        ParameterSpec(name="tscale", default=0, lower=0, upper=1),
        ParameterSpec(name="tlen", default=1, lower=0),
    )
    normalization_target_parameters: ClassVar[tuple[str, ...]] = ("amp",)
    use_scale: ClassVar[bool] = True
    use_offset: ClassVar[bool] = False

    def area(self, parameters: Mapping[str, float]) -> float | None:
        """Return None; the tail adds area beyond ``amp``."""
        del parameters
        return None

    @staticmethod
    def evaluate(
        x: NDArray,
        y: NDArray | None,
        **kwargs: float,
    ) -> NDArray:
        """Return the tailed pseudo-Voigt intensity at ``x``."""
        return tail_pvoigt(
            x,
            kwargs["amp"],
            kwargs["cen"],
            kwargs["sig"],
            kwargs["frac"],
            kwargs["tscale"],
            kwargs["tlen"],
        )

    @staticmethod
    def guess_initial(x: NDArray, y: NDArray, **kwargs: float | int | str) -> dict[str, float]:
        """Guess a symmetric pseudo-Voigt with the tail turned off."""
        guessed = PseudoVoigtPeakModel.guess_initial(x, y, **kwargs)
        guessed["tscale"] = 0.0
        guessed["tlen"] = guessed["sig"]
        return guessed

    @staticmethod
    def soft_parameter_range(
        name: str,
        value: float,
        lower: float,
        upper: float,
        *,
        x_min: float | None = None,
        x_max: float | None = None,
        y_max: float | None = None,
    ) -> tuple[float, float]:
        """Soft slider ranges; ``tlen`` follows the ``sig`` window."""
        key = name.lower()
        if key == "tlen":
            return PseudoVoigtPeakModel.soft_parameter_range(
                "sig", value, lower, upper, x_min=x_min, x_max=x_max, y_max=y_max
            )
        return PseudoVoigtPeakModel.soft_parameter_range(
            name, value, lower, upper, x_min=x_min, x_max=x_max, y_max=y_max
        )


class ConstantBackgroundModel(BaseBackgroundModel):
    """Constant (flat) background."""

    name: ClassVar[str] = "constant"
    parameter_schema: ClassVar[tuple[ParameterSpec, ...]] = (
        ParameterSpec("const", 0.0, vary=False),
    )
    normalization_target_parameters: ClassVar[tuple[str, ...]] = ("const",)
    use_scale: ClassVar[bool] = False
    use_offset: ClassVar[bool] = True

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

    @staticmethod
    def soft_parameter_range(
        name: str,
        value: float,
        lower: float,
        upper: float,
        *,
        x_min: float | None = None,
        x_max: float | None = None,
        y_max: float | None = None,
    ) -> tuple[float, float]:
        """Soft slider range for ``const``."""
        return intensity_soft_range(value, lower, upper, y_max=y_max)


class LinearBackgroundModel(BaseBackgroundModel):
    """Linear background between endpoint intensities ``i1`` and ``i2``."""

    name: ClassVar[str] = "linear"
    parameter_schema: ClassVar[tuple[ParameterSpec, ...]] = (
        ParameterSpec("i1", 0.0, vary=False),
        ParameterSpec("i2", 0.0),
    )
    normalization_target_parameters: ClassVar[tuple[str, ...]] = ("i1", "i2")
    use_scale: ClassVar[bool] = True
    use_offset: ClassVar[bool] = True

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

    @staticmethod
    def soft_parameter_range(
        name: str,
        value: float,
        lower: float,
        upper: float,
        *,
        x_min: float | None = None,
        x_max: float | None = None,
        y_max: float | None = None,
    ) -> tuple[float, float]:
        """Soft slider range for ``i1`` / ``i2``."""
        return intensity_soft_range(value, lower, upper, y_max=y_max)


class ShirleyBackgroundModel(BaseBackgroundModel):
    """Iterative Shirley background using endpoint intensities ``i1`` and ``i2``."""

    name: ClassVar[str] = "shirley"
    parameter_schema: ClassVar[tuple[ParameterSpec, ...]] = (
        ParameterSpec("i1", 0.0, vary=False),
        ParameterSpec("i2", 0.0, vary=False),
    )
    normalization_target_parameters: ClassVar[tuple[str, ...]] = ("i1", "i2")
    use_scale: ClassVar[bool] = True
    use_offset: ClassVar[bool] = True

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

    @staticmethod
    def soft_parameter_range(
        name: str,
        value: float,
        lower: float,
        upper: float,
        *,
        x_min: float | None = None,
        x_max: float | None = None,
        y_max: float | None = None,
    ) -> tuple[float, float]:
        """Soft slider range for Shirley ``i1`` / ``i2``."""
        return intensity_soft_range(value, lower, upper, y_max=y_max)
