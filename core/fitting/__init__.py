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
from core.fitting.fit_scope import (
    ExpressionProblem,
    collect_expression_problems,
    expand_fit_region_ids,
    format_expression_problems,
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
    "ExpressionProblem",
    "LmfitOptimizer",
    "OptimizationContext",
    "OptimizationExpressionPlan",
    "OptimizationPlanner",
    "OptimizedComponent",
    "ParsedParameterExpression",
    "build_contexts",
    "collect_expression_problems",
    "expand_fit_region_ids",
    "format_expression_problems",
    "match_component_reference",
    "optimize",
    "parse_parameter_expression",
    "resolve_component_reference",
]
