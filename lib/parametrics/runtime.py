from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class RuntimeParameter:
    """
    Mutable value object representing a single adjustable model parameter
    at runtime.

    ``RuntimeParameter`` stores the current numerical value of a parameter
    together with its optimization metadata: bounds, variation flag, and
    an optional dependency expression.

    This class is intentionally *model-agnostic* and *context-agnostic*.
    It does not know:
    - to which model it belongs;
    - how it should be normalized or denormalized;
    - how it is mapped to an optimizer backend (e.g. lmfit).

    These responsibilities are handled by higher-level components
    (models, normalization policies, optimization pipeline).

    Parameters
    ----------
    name : str
        Parameter name as defined by the owning model schema
        (e.g. ``"amp"``, ``"cen"``, ``"sig"``).
        Must be unique within a single model component.
    value : float
        Initial parameter value. The value is clipped to the interval
        ``[lower, upper]`` during initialization.
    lower : float, default=-np.inf
        Lower bound for the parameter value.
    upper : float, default=np.inf
        Upper bound for the parameter value.
    vary : bool, default=True
        Flag indicating whether this parameter is allowed to vary during
        optimization.
    expr : str or None, default=None
        Optional dependency expression referencing other parameters
        (typically resolved later by the optimization layer).

    Notes
    -----
    - ``RuntimeParameter`` is **mutable**: its value and bounds may change
      during interactive editing or optimization.
    - Use :meth:`set` to update attributes safely and preserve invariants.
    - Use :meth:`copy_with` to create modified copies without mutating
      the original object (required for normalization pipelines).
    """

    name: str
    value: float
    lower: float = -np.inf
    upper: float = np.inf
    vary: bool = True
    expr: Optional[str] = None

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            self.lower = -np.inf
            self.upper = np.inf
        self.value = self._clip(self.value)

    def _clip(self, value: float) -> float:
        """Clip value to the inclusive interval [lower, upper]."""
        return min(max(value, self.lower), self.upper)

    def _set_bound(self, lower: Optional[float] = None, upper: Optional[float] = None) -> None:
        if lower is not None and upper is not None:
            if lower > upper:
                raise ValueError
            self.lower = lower
            self.upper = upper
        elif lower is not None:
            if lower > self.upper:
                self.lower = self.upper
            else:
                self.lower = lower
        elif upper is not None:
            if upper < self.lower:
                self.upper = self.lower
            else:
                self.upper = upper

    def set(
        self,
        *,
        value: Optional[float] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        vary: Optional[bool] = None,
        expr: Optional[str] = None,
    ) -> None:
        """
        Mutate parameter attributes while preserving invariants.

        All arguments are optional. Bounds are applied first, and the
        current value is re-clipped if necessary.

        Parameters
        ----------
        value : float, optional
            New parameter value.
        lower : float, optional
            New lower bound.
        upper : float, optional
            New upper bound.
        vary : bool, optional
            New optimization variation flag.
        expr : str, optional
            New dependency expression.
        """
        self._set_bound(lower=lower, upper=upper)

        if value is not None:
            self.value = self._clip(float(value))
        else:
            # Re-clip existing value if bounds changed
            self.value = self._clip(self.value)

        if vary is not None:
            self.vary = bool(vary)
        if expr is not None:
            self.expr = expr

    def copy_with(
        self,
        dst: Optional["RuntimeParameter"] = None,
        *,
        value: Optional[float] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        vary: Optional[bool] = None,
        expr: Optional[str] = None,
    ) -> "RuntimeParameter":
        """
        Create a modified copy of this parameter, or write the modified
        values into an existing :class:`RuntimeParameter` instance.

        The method applies a *copy-on-write* strategy: the source parameter
        is never mutated. Instead, a new parameter is created, or an
        explicitly provided destination object is updated.

        This is primarily intended for:
        - normalization and denormalization pipelines;
        - preparing parameter sets for optimization;
        - safely transforming parameters without affecting domain state.

        Parameters
        ----------
        dst : RuntimeParameter, optional
            Destination parameter to update in-place. If ``None``,
            a new :class:`RuntimeParameter` instance is created and returned.
        value : float, optional
            New parameter value. If ``None``, the original value is preserved.
        lower : float, optional
            New lower bound. If ``None``, the original bound is preserved.
        upper : float, optional
            New upper bound. If ``None``, the original bound is preserved.
        vary : bool, optional
            Whether the parameter is allowed to vary during optimization.
            If ``None``, the original flag is preserved.
        expr : str, optional
            lmfit expression linking this parameter to others.
            If ``None``, the original expression is preserved.

        Returns
        -------
        RuntimeParameter
            The resulting parameter instance. This is either:
            - a newly created parameter, if ``dst`` is ``None``;
            - or the updated destination parameter.
        """
        param_cfg = dict(
            value=self.value if value is None else value,
            lower=self.lower if lower is None else lower,
            upper=self.upper if upper is None else upper,
            vary=self.vary if vary is None else vary,
            expr=self.expr if expr is None else expr,
        )
        if dst is None:
            param_cfg["name"] = self.name
            return RuntimeParameter(**param_cfg)
        else:
            dst.set(**param_cfg)

    def __repr__(self) -> str:
        return (
            f"<RuntimeParameter {self.name}: "
            f"value={self.value}, "
            f"lower={self.lower}, upper={self.upper}, "
            f"vary={self.vary}, expr={self.expr}>"
        )
