from .normalization import NormalizationContext
from .base_models import ParameterSpec, ParametricModelLike, BasePeakModel, BaseBackgroundModel

from typing import Dict


# NOTE: all core objects have the same ParametricModel instance
class ModelRegistry:
    _registry: Dict[str, type[ParametricModelLike]] = {}

    @classmethod
    def register(cls, model_cls: type[ParametricModelLike]):
        cls._registry[model_cls.name] = model_cls()

    @classmethod
    def get(cls, name: str) -> ParametricModelLike:
        return cls._registry[name]

    @classmethod
    def get_peak_model_names(cls) -> list[str]:
        """Return names of all registered peak models."""
        return [n for n, m in cls._registry.items() if isinstance(m, BasePeakModel)]

    @classmethod
    def get_background_model_names(cls) -> list[str]:
        """Return names of all registered background models."""
        return [n for n, m in cls._registry.items() if isinstance(m, BaseBackgroundModel)]


from .models import (
    PseudoVoigtPeakModel,
    ConstantBackgroundModel,
    LinearBackgroundModel,
    ShirleyBackgroundModel,
)


ModelRegistry.register(PseudoVoigtPeakModel)
ModelRegistry.register(ConstantBackgroundModel)
ModelRegistry.register(LinearBackgroundModel)
ModelRegistry.register(ShirleyBackgroundModel)
