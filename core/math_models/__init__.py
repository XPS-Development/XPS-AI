"""Parametric peak and background models with a name-based registry."""

from typing import ClassVar

from .base_models import (
    BaseBackgroundModel,
    BasePeakModel,
    ParametricModelLike,
)
from .base_models import (
    ParameterSpec as ParameterSpec,
)
from .models import (
    ConstantBackgroundModel,
    LinearBackgroundModel,
    PseudoVoigtPeakModel,
    ShirleyBackgroundModel,
)
from .normalization import NormalizationContext as NormalizationContext

# NOTE: all core objects have the same ParametricModel instance


class ModelRegistry:
    """Name-to-instance registry for parametric peak and background models."""

    _registry: ClassVar[dict[str, ParametricModelLike]] = {}

    @classmethod
    def register(cls, model_cls: type[ParametricModelLike]) -> None:
        """Register a model class by instantiating it under ``model_cls.name``."""
        cls._registry[model_cls.name] = model_cls()

    @classmethod
    def get(cls, name: str) -> ParametricModelLike:
        """Return the registered model instance for ``name``."""
        return cls._registry[name]

    @classmethod
    def get_peak_model_names(cls) -> list[str]:
        """Return names of all registered peak models."""
        return [n for n, m in cls._registry.items() if isinstance(m, BasePeakModel)]

    @classmethod
    def get_background_model_names(cls) -> list[str]:
        """Return names of all registered background models."""
        return [n for n, m in cls._registry.items() if isinstance(m, BaseBackgroundModel)]


ModelRegistry.register(PseudoVoigtPeakModel)
ModelRegistry.register(ConstantBackgroundModel)
ModelRegistry.register(LinearBackgroundModel)
ModelRegistry.register(ShirleyBackgroundModel)
