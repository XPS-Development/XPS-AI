"""
Parse and resolve lmfit-style parameter constraint expressions.

Provides a pure API that maps user-facing component tokens onto full component
ids and lmfit parameter names, returning per-token issues instead of silently
dropping the whole expression. Used by optimization planning and (later) by
editor validation / fit-scope closure without depending on the optimizer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Set

# Numbers first so ``1.5e3`` is not split into ``1`` / ``5e3``.
_TOKEN_RE = re.compile(
    r"(?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)|"
    r"(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
)

# Identifiers that lmfit/asteval treat as math builtins or constants.
# Unresolved tokens in this set are left untouched; they are not component refs.
_LMFIT_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "pi",
        "e",
        "inf",
        "nan",
        "True",
        "False",
        "abs",
        "sqrt",
        "exp",
        "log",
        "log10",
        "log2",
        "ln",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "min",
        "max",
        "floor",
        "ceil",
        "round",
        "fabs",
        "pow",
        "power",
        "hypot",
        "erf",
        "erfc",
        "sign",
        "where",
    }
)

ComponentRefStatus = Literal["exact", "prefix", "unknown", "ambiguous"]
ExpressionIssueCode = Literal[
    "unknown_component",
    "ambiguous_component",
    "missing_parameter",
]


@dataclass(frozen=True)
class ComponentReferenceMatch:
    """
    Result of matching one identifier against known component ids.

    Attributes
    ----------
    token : str
        Original identifier from the expression.
    status : {"exact", "prefix", "unknown", "ambiguous"}
        How the token related to ``component_ids``.
    component_id : str or None
        Resolved full id when ``status`` is ``exact`` or ``prefix``.
    candidates : tuple of str
        Matching ids when ``status`` is ``ambiguous`` (may be empty otherwise).
    """

    token: str
    status: ComponentRefStatus
    component_id: str | None = None
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExpressionComponentRef:
    """
    One successfully resolved component reference inside an expression.

    Attributes
    ----------
    token : str
        Original text span that resolved to a component.
    start, end : int
        Half-open character range into the source expression.
    component_id : str
        Full component id after prefix/exact resolution.
    parameter_name : str
        Parameter on the target component (same name as the constrained
        parameter of the owner — dotted ``id.param`` syntax comes later).
    """

    token: str
    start: int
    end: int
    component_id: str
    parameter_name: str


@dataclass(frozen=True)
class ExpressionIssue:
    """
    One unresolved or invalid token in an expression.

    Attributes
    ----------
    token : str
        Offending identifier (or empty when not applicable).
    start, end : int
        Half-open character range into the source expression.
    code : ExpressionIssueCode
        Machine-readable reason.
    message : str
        Human-readable explanation suitable for tooltips.
    """

    token: str
    start: int
    end: int
    code: ExpressionIssueCode
    message: str


@dataclass(frozen=True)
class ParsedParameterExpression:
    """
    Pure parse of one parameter constraint expression.

    Attributes
    ----------
    source : str
        Original expression text.
    references : tuple of ExpressionComponentRef
        Successfully resolved component references (order of appearance).
    issues : tuple of ExpressionIssue
        Per-token problems; empty when the expression is fully valid.
    lmfit_expr : str or None
        Expression with component tokens rewritten to ``{id}_{param}`` for
        lmfit, or ``None`` when any issue prevents a safe translation.
    """

    source: str
    references: tuple[ExpressionComponentRef, ...]
    issues: tuple[ExpressionIssue, ...]
    lmfit_expr: str | None


def match_component_reference(
    token: str,
    component_ids: Iterable[str],
) -> ComponentReferenceMatch:
    """
    Classify how ``token`` matches the known component id set.

    Exact id match wins. Otherwise a unique ``id.startswith(token)`` match is a
    prefix hit; multiple prefix hits are ambiguous; none is unknown.

    Parameters
    ----------
    token : str
        Identifier from an expression (short prefix or full id).
    component_ids : Iterable[str]
        Known component ids for the current resolution scope.

    Returns
    -------
    ComponentReferenceMatch
        Status plus resolved id or ambiguous candidates.
    """
    ids = tuple(component_ids)
    if token in ids:
        return ComponentReferenceMatch(token=token, status="exact", component_id=token)
    matches = tuple(cid for cid in ids if cid.startswith(token))
    if len(matches) == 1:
        return ComponentReferenceMatch(
            token=token,
            status="prefix",
            component_id=matches[0],
            candidates=matches,
        )
    if len(matches) > 1:
        return ComponentReferenceMatch(
            token=token,
            status="ambiguous",
            candidates=matches,
        )
    return ComponentReferenceMatch(token=token, status="unknown")


def resolve_component_reference(token: str, component_ids: Iterable[str]) -> str | None:
    """
    Map an expression token to a full component id.

    Exact id match always applies. Otherwise, return the unique component id such that
    ``id.startswith(token)``.

    Parameters
    ----------
    token : str
        Identifier from an expression (short prefix or full id).
    component_ids : Iterable[str]
        Known component ids for the current optimization scope.

    Returns
    -------
    str or None
        Resolved full id, or ``None`` if unknown or ambiguous.
    """
    return match_component_reference(token, component_ids).component_id


def parse_parameter_expression(
    expr: str,
    *,
    owner_component_id: str,
    parameter_name: str,
    known_component_ids: Iterable[str],
    parameter_names_by_component: Mapping[str, Set[str]] | None = None,
) -> ParsedParameterExpression:
    """
    Parse a constraint expression into references, issues, and an lmfit string.

    Bare component tokens mean “the same parameter name on that component”
    (current storage semantics). Numbers and known lmfit builtins are ignored.
    Failed resolution yields per-token :class:`ExpressionIssue` entries and
    ``lmfit_expr is None``.

    Parameters
    ----------
    expr : str
        Raw constraint expression (may be empty).
    owner_component_id : str
        Component that owns the constrained parameter (used only for callers;
        self-references are allowed).
    parameter_name : str
        Name of the constrained parameter; also the parameter read on each
        referenced component until dotted syntax exists.
    known_component_ids : Iterable[str]
        Component ids visible in the current scope.
    parameter_names_by_component : mapping, optional
        ``component_id → set of parameter names``. When provided, references to
        components that lack ``parameter_name`` become ``missing_parameter``
        issues. When omitted, that check is skipped.

    Returns
    -------
    ParsedParameterExpression
        Structured parse result; pure (no graph mutation).
    """
    known_ids = frozenset(known_component_ids)
    references: list[ExpressionComponentRef] = []
    issues: list[ExpressionIssue] = []
    # (start, end, replacement) for successful refs; applied right-to-left.
    replacements: list[tuple[int, int, str]] = []

    for match in _TOKEN_RE.finditer(expr):
        if match.lastgroup == "number":
            continue

        token = match.group("ident")
        start, end = match.span()
        ref_match = match_component_reference(token, known_ids)

        if ref_match.status in {"exact", "prefix"}:
            component_id = ref_match.component_id
            assert component_id is not None
            if (
                parameter_names_by_component is not None
                and parameter_name not in parameter_names_by_component.get(component_id, ())
            ):
                issues.append(
                    ExpressionIssue(
                        token=token,
                        start=start,
                        end=end,
                        code="missing_parameter",
                        message=(f"component '{component_id}' has no parameter '{parameter_name}'"),
                    )
                )
                continue

            references.append(
                ExpressionComponentRef(
                    token=token,
                    start=start,
                    end=end,
                    component_id=component_id,
                    parameter_name=parameter_name,
                )
            )
            replacements.append((start, end, f"{component_id}_{parameter_name}"))
            continue

        if ref_match.status == "ambiguous":
            issues.append(
                ExpressionIssue(
                    token=token,
                    start=start,
                    end=end,
                    code="ambiguous_component",
                    message=f"ambiguous component reference '{token}'",
                )
            )
            continue

        if token in _LMFIT_BUILTIN_NAMES:
            continue

        issues.append(
            ExpressionIssue(
                token=token,
                start=start,
                end=end,
                code="unknown_component",
                message=f"unknown component reference '{token}'",
            )
        )

    if issues:
        return ParsedParameterExpression(
            source=expr,
            references=tuple(references),
            issues=tuple(issues),
            lmfit_expr=None,
        )

    lmfit_expr = expr
    for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
        lmfit_expr = lmfit_expr[:start] + replacement + lmfit_expr[end:]

    return ParsedParameterExpression(
        source=expr,
        references=tuple(references),
        issues=(),
        lmfit_expr=lmfit_expr,
    )
