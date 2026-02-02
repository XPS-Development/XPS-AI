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
