"""Lmfit-based spectral fitting utilities."""

from core.fitting.expressions import (
    ComponentReferenceMatch,
    ExpressionComponentRef,
    ExpressionIssue,
    ParsedParameterExpression,
    match_component_reference,
    parse_parameter_expression,
    resolve_component_reference,
)
from core.fitting.optimization import (
    LmfitOptimizer,
    OptimizationContext,
    OptimizationExpressionPlan,
    OptimizationPlanner,
    OptimizedComponent,
    build_contexts,
    optimize,
)

__all__ = [
    "ComponentReferenceMatch",
    "ExpressionComponentRef",
    "ExpressionIssue",
    "LmfitOptimizer",
    "OptimizationContext",
    "OptimizationExpressionPlan",
    "OptimizationPlanner",
    "OptimizedComponent",
    "ParsedParameterExpression",
    "build_contexts",
    "match_component_reference",
    "optimize",
    "parse_parameter_expression",
    "resolve_component_reference",
]
