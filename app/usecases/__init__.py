"""Application use-cases that translate user intents into Change objects."""

from .analysis import AnalysisUseCases
from .editing import EditingUseCases

__all__ = ["AnalysisUseCases", "EditingUseCases"]
