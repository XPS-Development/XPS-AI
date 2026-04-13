from uuid import uuid4
from dataclasses import dataclass, field, asdict

import numpy as np

from .math_models import NormalizationContext, BasePeakModel, BaseBackgroundModel, ParametricModelLike

from typing import Protocol, Dict, Any, Optional, runtime_checkable
from numpy.typing import NDArray

_EXPR_MISSING: object = object()


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
        expr: Optional[str] = _EXPR_MISSING,
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

        if expr is not _EXPR_MISSING:
            self.expr = expr

    def clone(self, **overrides: str | float | bool) -> "RuntimeParameter":
        """
        Create a modified copy of the parameter.

        Returns a new ``RuntimeParameter`` instance with the same attributes
        as the current one, optionally overridden by provided keyword arguments.

        Parameters
        ----------
        **overrides
            Parameter attributes to override in the cloned instance.
            Typical keys include ``value``, ``lower``, ``upper``, ``vary``,
            and ``expr``.

        Returns
        -------
        RuntimeParameter
            A new parameter instance with updated attributes.
        """
        params = asdict(self)
        params.update(**overrides)
        return RuntimeParameter(**params)

    def copy_to(self, dst: "RuntimeParameter") -> None:
        """
        Copy parameter configuration to another parameter instance.

        Updates the destination parameter in-place by copying the current
        parameter's numerical value, bounds, variation flag, and expression.

        Parameters
        ----------
        dst : RuntimeParameter
            Destination parameter instance to be updated.
        """
        params = asdict(self)
        params.pop("name")
        dst.set(**params)

    def __repr__(self) -> str:
        return (
            f"<RuntimeParameter {self.name}: "
            f"value={self.value}, "
            f"lower={self.lower}, upper={self.upper}, "
            f"vary={self.vary}, expr={self.expr}>"
        )


@runtime_checkable
class CoreObject(Protocol):
    id_: str
    parent_id: Optional[str]


class Component:
    """
    Base class for parametric model components (e.g. peaks, backgrounds).

    A ``Component`` represents a *runtime instance* of a parametric model
    bound to a specific parent core object (typically a Region). It combines:

    - a concrete :class:`ParametricModelLike` (defining mathematical behavior);
    - a set of runtime-adjustable parameters (:class:`RuntimeParameter`);
    - a stable unique identifier;
    - a parent-child relationship to the core model.

    ``Component`` itself is agnostic to the semantics of the model it represents.
    It does not know whether it is a peak, a background, or another parametric
    object. Specializations (e.g. :class:`Peak`, :class:`Background`) define
    core-specific constraints.

    The Component:
    - is always associated with exactly one Region;
    - evaluates to a 1D array over the region's x-grid;

    Responsibilities
    ----------------
    - Instantiate runtime parameters from the model's parameter schema.
    - Validate provided parameter values against the schema.
    - Provide a uniform interface for parameter access and mutation.
    - Act as a bridge between core objects and optimization models.

    Explicitly out of scope
    -----------------------
    - Parameter normalization / denormalization.
    - Optimization logic or dependency resolution.
    - Knowledge of spectra, regions, or collections beyond ``parent_id``.

    Parameters
    ----------
    model : ParametricModelLike
        Parametric model providing a parameter schema and an evaluate function.
    parent_id : str
        Identifier of the parent core object (e.g. Region ID).
        Used for structural integrity and traceability.
    component_id : str, optional
        Explicit identifier of the component. If not provided, a new ID
        is generated using ``component_prefix``.
    component_prefix : str, default="c"
        Prefix used for auto-generated component IDs.
        Concrete subclasses typically override this
        (e.g. ``"p"`` for peaks, ``"b"`` for backgrounds).
    **param_values : float
        Optional initial values for model parameters.
        Keys must match parameter names defined in the model's schema.
        Missing parameters are initialized from schema defaults.

    Raises
    ------
    ValueError
        If unknown parameter names are provided.

    Attributes
    ----------
    id_ : str
        Unique identifier of the component.
    model : ParametricModel
        Parametric model instance defining parameter schema and evaluation logic.
    parent_id : str
        Identifier of the parent core object.
    parameters : dict[str, RuntimeParameter]
        Mapping of parameter names to runtime parameter objects.
    """

    def __init__(
        self,
        *,
        model: ParametricModelLike,
        parent_id: str,
        component_id: Optional[str] = None,
        component_prefix: str = "c",
        **param_values: float,
    ) -> None:

        self.id_: str = component_id or f"{component_prefix}{uuid4().hex}"
        self.model = model
        self.parent_id = parent_id

        # Initialize parameters
        self.parameters: Dict[str, RuntimeParameter] = {}
        schema = {p.name: p for p in model.parameter_schema}

        unknown = set(param_values) - set(schema)
        if unknown:
            raise ValueError(f"Unknown parameters for model '{model.name}': {unknown}")

        for name, spec in schema.items():
            value = param_values.get(name, spec.default)
            self.parameters[name] = RuntimeParameter(
                name=name,
                value=value,
                lower=spec.lower,
                upper=spec.upper,
                vary=spec.vary,
                expr=spec.expr,
            )

    def set_param(self, name: str, **kwargs: Any) -> None:
        """
        Update attributes of a runtime parameter.

        Parameters
        ----------
        name : str
            Name of the parameter to update.
        **kwargs
            Attributes to update (e.g. value, lower, upper, vary, expr).

        Raises
        ------
        KeyError
            If the parameter does not exist in this component.
        """
        if name not in self.parameters:
            raise KeyError(f"Parameter '{name}' not found in component '{self.id_}'")
        self.parameters[name].set(**kwargs)

    def get_param(self, name: str) -> RuntimeParameter:
        """
        Retrieve a runtime parameter by name.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        RuntimeParameter
            Parameter instance.

        Raises
        ------
        KeyError
            If the parameter does not exist.
        """
        if name not in self.parameters:
            raise KeyError(f"Parameter '{name}' not found in component '{self.id_}'")
        return self.parameters[name]

    def __repr__(self) -> str:
        params = ", ".join(f"{n}={p.value:.4g}" for n, p in self.parameters.items())
        pid = self.parent_id[:8] if self.parent_id else "None"
        return f"<{type(self).__name__} parent_id={pid} id={self.id_[:8]} model={self.model.name} {params}>"


class Peak(Component):
    """
    Peak component bound to a spectral region.

    ``Peak`` represents a concrete peak instance with a specific peak model
    (e.g. pseudo-Voigt, Gaussian, custom user-defined model) and a set of
    runtime parameters.

    The peak:
    - is always associated with exactly one Region;
    - evaluates to a 1D array over the region's x-grid;
    - does not handle normalization internally.

    The mathematical behavior is fully delegated to the underlying
    :class:`BasePeakModel`.

    Parameters
    ----------
    model : BasePeakModel
        Peak model defining the peak shape and parameter schema.
    region_id : str
        Identifier of the parent region.
    component_id : str, optional
        Explicit peak ID. If omitted, a new ID is generated.
    **param_values : float
        Initial parameter values overriding model defaults.
    """

    def __init__(
        self,
        *,
        model: BasePeakModel,
        region_id: str,
        component_id: Optional[str] = None,
        **param_values: float,
    ):
        super().__init__(
            model=model,
            parent_id=region_id,
            component_id=component_id,
            component_prefix="p",
            **param_values,
        )


class Background(Component):
    """
    Background component bound to a spectral region.

    ``Background`` represents a parametric background model associated with
    a region.

    Typical use cases include:
    - static backgrounds (linear, Shirley);
    - active backgrounds with optimizable parameters.

    The mathematical behavior is delegated to the underlying
    :class:`BaseBackgroundModel`.

    Parameters
    ----------
    model : BaseBackgroundModel
        Background model defining evaluation logic and parameters..
    region_id : str
        Identifier of the parent region.
    component_id : str, optional
        Explicit background ID. If omitted, a new ID is generated.
    **param_values : float
        Initial parameter values overriding model defaults.
    """

    def __init__(
        self,
        *,
        model: BaseBackgroundModel,
        region_id: str,
        component_id: Optional[str] = None,
        **param_values: float,
    ):
        super().__init__(
            model=model,
            parent_id=region_id,
            component_id=component_id,
            component_prefix="b",
            **param_values,
        )


@dataclass
class Region:
    """
    Region core object.

    A ``Region`` represents a contiguous sub-range of a spectrum defined
    purely by index slicing. It is a *structural* and *semantic* core entity
    that groups peaks and a background, but does not store numerical data
    such as x/y arrays or normalization context.

    The region:
    - belongs to exactly one Spectrum;
    - is defined by an index slice into the parent spectrum;

    ``Region`` intentionally avoids any knowledge of optimization,
    parameter normalization, or fitting libraries.

    Design principles
    -----------------
    - **Index-based**: the region is defined by indices, not by x-values.
    - **Data-light**: x and y are retrieved externally from SpectrumCollection.
    - **One background per region**: background is mandatory and unique.
    - **Immutable geometry**: region boundaries are not expected to change
      during optimization (recreate instead).

    Parameters
    ----------
    slice_ : slice
        Index slice defining the region range within the parent spectrum.
        Must have ``step`` equal to ``None`` or ``1``.
    parent_id : str
        Identifier of the parent Spectrum.
    id_ : str, optional
        Explicit region identifier. If not provided, a new one is generated.

    Attributes
    ----------
    id_ : str
        Unique identifier of the region.
    parent_id : str
        Identifier of the parent Spectrum.
    slice_ : slice
        Slice selecting the region range in the spectrum arrays.
    """

    slice_: slice
    parent_id: str
    id_: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.slice_, slice):
            raise TypeError("slice must be a slice object")
        if self.slice_.step not in (None, 1):
            raise ValueError("slice step must be None or 1")
        if self.id_ is None:
            self.id_ = f"r{uuid4().hex}"


@dataclass
class Spectrum:
    """
    Spectrum core object.

    Represents a single measured spectrum and owns the raw spectral data
    (`x`, `y`) together with a normalization context derived from it.

    A `Spectrum` is the top-level core entity in the spectral hierarchy:
    regions and peaks are always created *from* a spectrum and are
    indirectly associated with it via `spectrum_id` / `parent_id`.

    Design principles
    -----------------
    - A Spectrum is *data-owning*: it stores raw, immutable measurement data.
    - It does NOT store regions or peaks directly.
      All structural relationships are managed externally
      (e.g. by `SpectrumCollection`).
    - Geometry (indexing, slicing) is defined here;
      model fitting and evaluation are handled by other layers.
    - Normalization rules are centralized in the spectrum and exposed
      via a `NormalizationContext`.

    Lifecycle
    ---------
    1. Created from raw x/y arrays (e.g. loaded from file).
    2. Normalization context is computed once in `__post_init__`.
    3. Spectrum itself remains unchanged during fitting and optimization.

    Parameters
    ----------
    x : NDArray
        One-dimensional array of x-axis values (e.g. energy, wavelength).
    y : NDArray
        One-dimensional array of measured intensities.
        Must have the same length as `x`.

    id_ : str, optional
        Unique spectrum identifier.
        If not provided, a new ID is generated automatically.
        Conventionally prefixed with ``"s"``.

    Attributes
    ----------
    id_ : str
        Unique identifier of the spectrum.
    parent_id : None
        Always ``None`` for Spectrum.
        Present for consistency with other core objects.
    norm_ctx : NormalizationContext
        Normalization context derived from the spectrum's y-data.
        Used by regions, peaks, and optimization routines.

    Notes
    -----
    - The spectrum assumes x and y are not mutated after creation.
    - No fitting or model-specific logic should be added here.
    """

    x: NDArray
    y: NDArray

    id_: Optional[str] = None
    parent_id = None

    norm_ctx: NormalizationContext = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        self.norm_ctx = NormalizationContext.from_array(self.y)
        if self.id_ is None:
            self.id_ = f"s{uuid4().hex}"

    def _validate(self) -> None:
        if not isinstance(self.x, np.ndarray) or not isinstance(self.y, np.ndarray):
            raise TypeError("x and y must be numpy arrays")
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("x and y must be 1D arrays")
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")
