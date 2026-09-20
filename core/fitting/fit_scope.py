"""
Fit-scope expansion from parameter expression dependencies.

Given a selected set of regions and the document's components, expand to the
undirected closure of regions linked by constraint expressions. Pure: no
query/UI; callers supply component DTOs (``parent_id`` = region id).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.fitting.expressions import ExpressionIssue, parse_parameter_expression

if TYPE_CHECKING:
    from collections.abc import Sequence

    from core.dto import ComponentDTO


@dataclass(frozen=True)
class ExpressionProblem:
    """
    One invalid parameter expression in the fit scope.

    Attributes
    ----------
    component_id : str
        Component that owns the constrained parameter.
    parameter_name : str
        Parameter whose ``expr`` failed to resolve.
    expression : str
        Raw expression text.
    issues : tuple of ExpressionIssue
        Per-token problems from :func:`parse_parameter_expression`.
    """

    component_id: str
    parameter_name: str
    expression: str
    issues: tuple[ExpressionIssue, ...]


def collect_expression_problems(
    components: Sequence[ComponentDTO],
    *,
    region_ids: Sequence[str] | None = None,
) -> tuple[ExpressionProblem, ...]:
    """
    Find parameter expressions that do not fully resolve in the given scope.

    Parameters
    ----------
    components : Sequence[ComponentDTO]
        Document components used both as the known-id set and as owners to
        check (typically the whole collection).
    region_ids : Sequence[str] or None, optional
        If given, only expressions on components whose ``parent_id`` is in this
        set are reported. Known-id resolution still uses all ``components``.

    Returns
    -------
    tuple of ExpressionProblem
        One entry per constrained parameter with at least one issue.
    """
    known_ids = frozenset(cmp.id_ for cmp in components)
    parameter_names_by_component = {cmp.id_: set(cmp.parameters.keys()) for cmp in components}
    region_filter = set(region_ids) if region_ids is not None else None

    problems: list[ExpressionProblem] = []
    for cmp in components:
        if region_filter is not None and cmp.parent_id not in region_filter:
            continue
        for pname, param in cmp.parameters.items():
            if not param.expr:
                continue
            parsed = parse_parameter_expression(
                param.expr,
                owner_component_id=cmp.id_,
                parameter_name=pname,
                known_component_ids=known_ids,
                parameter_names_by_component=parameter_names_by_component,
            )
            if parsed.issues:
                problems.append(
                    ExpressionProblem(
                        component_id=cmp.id_,
                        parameter_name=pname,
                        expression=param.expr,
                        issues=parsed.issues,
                    )
                )
    return tuple(problems)


def format_expression_problems(problems: Sequence[ExpressionProblem]) -> str:
    """
    Format expression problems for a dialog or exception message.

    Parameters
    ----------
    problems : Sequence[ExpressionProblem]
        Problems to format.

    Returns
    -------
    str
        Multi-line human-readable summary.
    """
    lines: list[str] = ["Cannot optimize: some parameter expressions are invalid."]
    for problem in problems:
        short_id = problem.component_id[:8]
        issue_text = "; ".join(issue.message for issue in problem.issues)
        lines.append(
            f"• {short_id}….{problem.parameter_name} = {problem.expression!r}: {issue_text}"
        )
    return "\n".join(lines)


def expand_fit_region_ids(
    selected_region_ids: Sequence[str],
    components: Sequence[ComponentDTO],
) -> tuple[str, ...]:
    """
    Expand selected regions to the undirected expression-dependency closure.

    Builds a component graph from all non-empty parameter ``expr`` fields using
    :func:`parse_parameter_expression` with document-wide known ids, then
    includes every region that owns a component in the same connected component
    as any component in the selection.

    Parameters
    ----------
    selected_region_ids : Sequence[str]
        Regions the caller intends to optimize (order preserved in the result).
    components : Sequence[ComponentDTO]
        All components in scope (typically the whole document). ``parent_id``
        must be the owning region id.

    Returns
    -------
    tuple[str, ...]
        Closed region ids: selected ids first (stable), then any extras sorted.
    """
    if not selected_region_ids:
        return ()

    selected = set(selected_region_ids)
    known_ids = frozenset(cmp.id_ for cmp in components)
    parameter_names_by_component = {cmp.id_: set(cmp.parameters.keys()) for cmp in components}
    region_by_component = {
        cmp.id_: cmp.parent_id for cmp in components if cmp.parent_id is not None
    }

    graph: dict[str, set[str]] = {cid: set() for cid in known_ids}
    for cmp in components:
        for pname, param in cmp.parameters.items():
            if not param.expr:
                continue
            parsed = parse_parameter_expression(
                param.expr,
                owner_component_id=cmp.id_,
                parameter_name=pname,
                known_component_ids=known_ids,
                parameter_names_by_component=parameter_names_by_component,
            )
            for ref in parsed.references:
                if ref.component_id == cmp.id_ or ref.component_id not in graph:
                    continue
                graph[cmp.id_].add(ref.component_id)
                graph[ref.component_id].add(cmp.id_)

    seed = [cid for cid, region_id in region_by_component.items() if region_id in selected]
    visited: set[str] = set()
    stack = list(seed)
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(graph[current] - visited)

    closed_regions = {region_by_component[cid] for cid in visited if cid in region_by_component}
    closed_regions |= selected

    result: list[str] = []
    seen: set[str] = set()
    for region_id in selected_region_ids:
        if region_id not in seen:
            result.append(region_id)
            seen.add(region_id)
    for region_id in sorted(closed_regions - seen):
        result.append(region_id)
    return tuple(result)
