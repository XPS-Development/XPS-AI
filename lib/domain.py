from uuid import uuid4
from dataclasses import dataclass, field

import numpy as np

from .parametrics import (
    NormalizationContext,
    BasePeakModel,
    BaseBackgroundModel,
    ParametricModelLike,
    RuntimeParameter,
)

from typing import Protocol, Dict, Any, Optional, TypeVar
from numpy.typing import NDArray


class DomainObject(Protocol):
    id_: str
    parent_id: Optional[str]


class Component:
    """
    Base class for parametric model components (e.g. peaks, backgrounds).

    A ``Component`` represents a *runtime instance* of a parametric model
    bound to a specific parent domain object (typically a Region). It combines:

    - a concrete :class:`ParametricModelLike` (defining mathematical behavior);
    - a set of runtime-adjustable parameters (:class:`RuntimeParameter`);
    - a stable unique identifier;
    - a parent-child relationship to the domain model.

    ``Component`` itself is agnostic to the semantics of the model it represents.
    It does not know whether it is a peak, a background, or another parametric
    object. Specializations (e.g. :class:`Peak`, :class:`Background`) define
    evaluation signatures and domain-specific constraints.

    Responsibilities
    ----------------
    - Instantiate runtime parameters from the model's parameter schema.
    - Validate provided parameter values against the schema.
    - Provide a uniform interface for parameter access and mutation.
    - Act as a bridge between domain objects and optimization models.

    Explicitly out of scope
    -----------------------
    - Parameter normalization / denormalization.
    - Optimization logic or dependency resolution.
    - Knowledge of spectra, regions, or collections beyond ``parent_id``.

    Parameters
    ----------
    model : ParametricModelLike
        Parametric model providing a parameter schema and an evaluate function.
        The model provides a parameter schema and an evaluation function.
    parent_id : str
        Identifier of the parent domain object (e.g. Region ID).
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
        Identifier of the parent domain object.
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

    def evaluate(self, *args, **kwargs) -> NDArray:
        """
        Evaluate the component.

        This method must be implemented by subclasses, as different component
        types have different evaluation signatures.

        Raises
        ------
        NotImplementedError
            If called on the base class.
        """
        raise NotImplementedError

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

    def evaluate(self, x: NDArray) -> NDArray:
        """
        Evaluate the peak model on the given x-grid.

        Parameters
        ----------
        x : NDArray
            X-axis values for evaluation.

        Returns
        -------
        NDArray
            Peak contribution evaluated at ``x``.
        """
        kwargs = {name: param.value for name, param in self.parameters.items()}
        return self.model.evaluate(x, y=None, **kwargs)


class Background(Component):
    """
    Background component bound to a spectral region.

    ``Background`` represents a parametric background model associated with
    a region. Unlike peaks, background models may depend not only on the
    x-axis but also on the observed intensity y.

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

    def evaluate(self, x: NDArray, y: NDArray | None = None) -> NDArray:
        """
        Evaluate the background model. Some background models may depend on y.

        Parameters
        ----------
        x : NDArray
            X-axis values.
        y : NDArray, optional
            Y-axis values, used by background models that depend on intensity.

        Returns
        -------
        NDArray
            Evaluated background.
        """
        kwargs = {name: param.value for name, param in self.parameters.items()}
        return self.model.evaluate(x, y, **kwargs)


@dataclass
class Region:
    """
    Region domain object.

    A ``Region`` represents a contiguous sub-range of a spectrum defined
    purely by index slicing. It is a *structural* and *semantic* domain entity
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
    Spectrum domain object.

    Represents a single measured spectrum and owns the raw spectral data
    (`x`, `y`) together with a normalization context derived from it.

    A `Spectrum` is the top-level domain entity in the spectral hierarchy:
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
        Present for consistency with other domain objects.
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


T = TypeVar("T")


class SpectrumCollection:
    """
    Collection and lifecycle manager for spectral domain objects.

    `SpectrumCollection` is a lightweight registry that stores and indexes
    all domain objects participating in the spectral model:
    - Spectrum
    - Region
    - Peak
    - Background

    The collection is responsible for:
    - Object registration and lookup by ID
    - Maintaining parent–child relationships via `parent_id`
    - Cascading removal of hierarchical objects

    It is intentionally *not* responsible for:
    - Validation of scientific correctness
    - Business logic (copy/paste, fitting, optimization)
    - Object creation policies

    Design principles
    -----------------
    - Single source of truth: all objects are stored in a single index.
    - Relationships are implicit and resolved via `parent_id`.
    - No reliance on ID prefixes or explicit type dispatch.
    - Minimal assumptions about object internals.

    Object hierarchy assumptions
    -----------------------------
    The collection assumes the following containment hierarchy:

        Spectrum
          └── Region
                └── Peak / Background / other components

    These rules are not enforced structurally, but are relied upon
    for correct cascading deletion.

    Notes
    -----
    - All objects must expose:
        - `id_ : str`
        - `parent_id : Optional[str]`
    - IDs must be globally unique within the collection.
    - Objects are assumed to be immutable with respect to identity
      (`id_` and `parent_id` should not change after registration).

    Attributes
    ----------
    objects_index : dict[str, Spectrum | Region | Peak]
        Global mapping from object ID to domain object instance.
    """

    def __init__(self):
        self.objects_index: Dict[str, DomainObject] = {}

    def add(self, obj: DomainObject) -> None:
        """
        Register an object in the collection.

        This method adds the object to the internal index.
        It does NOT automatically add parents or children.

        Parameters
        ----------
        obj : Spectrum or Region or Peak
            Domain object to register.

        Raises
        ------
        ValueError
            If the object does not have a valid `id_`.
        KeyError
            If an object with the same ID already exists.
        """
        if obj.id_ is None:
            raise ValueError("Object must have an id_")

        if obj.id_ in self.objects_index:
            raise KeyError(f"Object with id '{obj.id_}' already exists")

        self.objects_index[obj.id_] = obj

    def remove(self, obj: DomainObject | str) -> None:
        """
        Remove an object from the collection.

        Removal is recursive:
        - If a Spectrum is removed, all its Regions and Peaks are removed.
        - If a Region is removed, all its Peaks (and Backgrounds) are removed.
        - If a Peak is removed, only the peak itself is removed.

        Parameters
        ----------
        obj : Spectrum or Region or Peak or Background or str
            Object instance or its ID.

        Raises
        ------
        KeyError
            If the object ID is not present in the collection.
        """
        if isinstance(obj, str):
            obj_id = obj
        else:
            obj_id = obj.id_

        obj = self.objects_index.pop(obj_id)

        if isinstance(obj, (Spectrum, Region)):
            children = list(self.get_children(obj_id))
            for ch in children:
                self.remove(ch)

    def get(self, obj_id: str) -> DomainObject:
        """
        Retrieve an object by its ID.

        Parameters
        ----------
        obj_id : str
            Identifier of the requested object.

        Returns
        -------
        Spectrum or Region or Peak or Background
            The corresponding domain object.

        Raises
        ------
        KeyError
            If no object with this ID exists in the collection.
        """
        return self.objects_index[obj_id]

    def get_typed(self, obj_id: str, tp: type[T]) -> T:
        obj = self.get(obj_id)
        if not isinstance(obj, tp):
            raise TypeError(f"{obj_id} is not {tp.__name__}")
        return obj

    def get_children(self, obj_id: str) -> tuple[DomainObject, ...]:
        """
        Retrieve direct children of a given object.

        Children are defined as objects whose `parent_id`
        matches the given `obj_id`.

        Parameters
        ----------
        obj_id : str
            Parent object ID.

        Returns
        -------
        tuple of Region or Peak or Background
            Direct children of the given object.
            The order is not guaranteed.
        """
        return tuple(ch for ch in self.objects_index.values() if ch.parent_id == obj_id)
