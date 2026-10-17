"""Continuous minimum-cost path cover, ported from ai-xps-series.

Only adjacent supplied steps can link. Every eligible peak belongs to a track,
including singletons. Doublet roles are neither inferred nor accepted as gates.
"""

from itertools import pairwise
from math import isfinite, log, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from core.post_processing.types import (
    MatchingResult,
    MatchingSeries,
    MatchingSettings,
    PeakDescriptor,
    PeakLink,
    PeakTrack,
)


def _huber(value: float) -> float:
    value = abs(value)
    return value * value / 2 if value <= 1 else value - 0.5


def _features(peaks: tuple[PeakDescriptor, ...]) -> dict[str, tuple[float, float]]:
    positions = [peak.position for peak in peaks]
    total = sum(max(0.0, peak.area or 0.0) for peak in peaks)
    if not isfinite(total):
        raise ValueError("Non-finite area sum; adjust input magnitudes")
    result = {}
    for peak in peaks:
        rank = (
            sum(p < peak.position for p in positions)
            + (sum(p == peak.position for p in positions) - 1) / 2
        ) / max(1, len(peaks) - 1)
        area = sqrt(max(0.0, peak.area or 0.0) / total) if total else 0.0
        result[peak.component_id] = rank, area
    return result


def _transition(
    source: PeakDescriptor,
    target: PeakDescriptor,
    features: dict[str, tuple[float, float]],
    config: MatchingSettings,
    *,
    manual: bool,
) -> PeakLink | None:
    delta = target.position - source.position
    if not manual and abs(delta) > config.energy_gate:
        return None
    terms = [("position", config.energy_weight * _huber(delta / config.energy_scale))]
    missing = []
    for field, name, weight, scale in (
        ("sig", "width", config.width_weight, config.width_scale),
        ("shape_gl", "shape_gl", config.shape_weight, config.shape_scale),
    ):
        if weight == 0:
            continue
        x, y = getattr(source, field), getattr(target, field)
        if x is None or y is None:
            missing.append(field)
            continue
        difference = log(y) - log(x) if field == "sig" else y - x
        terms.append((name, weight * _huber(difference / scale)))
    rank_a, area_a = features[source.component_id]
    rank_b, area_b = features[target.component_id]
    terms.extend(
        (
            ("area_fraction", config.area_weight * (area_a - area_b) ** 2),
            ("energy_rank", config.rank_weight * (rank_a - rank_b) ** 2),
        )
    )
    cost = sum(value for _, value in terms)
    if not isfinite(cost):
        raise ValueError("Non-finite transition cost; adjust scales or input magnitudes")
    return PeakLink(
        source.component_id, target.component_id, cost, tuple(terms), tuple(missing), manual
    )


def _solve(
    matrix: NDArray,
    edges: dict[tuple[int, int], PeakLink],
    fixed: tuple[tuple[int, int], ...],
    config: MatchingSettings,
    forbidden: tuple[int, int] | None = None,
) -> tuple[list[tuple[int, int]], float]:
    n = len(matrix)
    if not n:
        return [], 0.0
    costs = matrix.copy()
    if forbidden is not None:
        costs[forbidden] = np.inf
    for i, j in fixed:
        costs[i, :] = np.inf
        costs[:, j] = np.inf
        costs[i, j] = matrix[i, j]
    rows, columns = linear_sum_assignment(costs)
    selected = [(int(i), int(j)) for i, j in zip(rows, columns, strict=True) if j < n]
    objective = (n - len(selected)) * (config.start_cost + config.end_cost)
    objective += sum(edges[pair].cost for pair in selected)
    if not isfinite(objective):
        raise ValueError("Non-finite matching objective")
    return selected, objective


def validate_constraints(series: MatchingSeries) -> set[tuple[str, str]]:
    """Validate manual restrictions without running assignment or ignoring conflicts."""
    orders = {
        peak.component_id: order for order, step in enumerate(series.steps) for peak in step.peaks
    }
    required = set(series.constraints.required)
    forbidden = set(series.constraints.forbidden)
    if required & forbidden:
        raise ValueError("A manual link cannot also be forbidden")
    for source, target in required | forbidden:
        if source not in orders or target not in orders:
            raise ValueError(f"Constraint references an unavailable peak: {source} -> {target}")
        if orders[target] != orders[source] + 1:
            raise ValueError("Constraints must connect adjacent supplied steps")
    if len({a for a, _ in required}) != len(required):
        raise ValueError("A component cannot have multiple successors")
    if len({b for _, b in required}) != len(required):
        raise ValueError("A component cannot have multiple predecessors")
    return required


def match_series(series: MatchingSeries) -> MatchingResult:
    """Cover all peaks with adjacent tracks and evaluate global link margins.

    For each selected automatic edge, solve again with that edge forbidden.
    The increase in the complete constrained objective measures ambiguity.
    Required links bypass the automatic energy gate and have no margin.
    Equal-cost ties are resolved deterministically using sorted component IDs.
    """
    nodes = [
        peak for step in series.steps for peak in sorted(step.peaks, key=lambda p: p.component_id)
    ]
    orders = {
        peak.component_id: order for order, step in enumerate(series.steps) for peak in step.peaks
    }
    features = {key: value for step in series.steps for key, value in _features(step.peaks).items()}
    required = validate_constraints(series)
    forbidden = set(series.constraints.forbidden)
    indices = {peak.component_id: index for index, peak in enumerate(nodes)}
    fixed = tuple(sorted((indices[a], indices[b]) for a, b in required))
    n = len(nodes)
    matrix = np.full((n, 2 * n), np.inf)
    matrix[:, n:] = 0
    edges = {}
    config = series.settings
    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            if orders[target.component_id] != orders[source.component_id] + 1:
                continue
            pair = source.component_id, target.component_id
            if pair in forbidden:
                continue
            edge = _transition(source, target, features, config, manual=pair in required)
            if edge is not None:
                edges[i, j] = edge
                matrix[i, j] = edge.cost - config.start_cost - config.end_cost
    selected, objective = _solve(matrix, edges, fixed, config)
    links = {}
    for i, j in selected:
        edge = edges[i, j]
        if edge.manual:
            links[i, j] = edge
        else:
            _, alternative = _solve(matrix, edges, fixed, config, forbidden=(i, j))
            margin = max(0.0, alternative - objective)
            links[i, j] = PeakLink(
                edge.source_id,
                edge.target_id,
                edge.cost,
                edge.terms,
                edge.missing_features,
                cost_margin=margin,
                ambiguous=margin <= config.ambiguity_threshold,
            )
    successors = dict(selected)
    predecessors = {j for _, j in selected}
    tracks = []
    for start in range(n):
        if start in predecessors:
            continue
        chain = [start]
        while chain[-1] in successors:
            chain.append(successors[chain[-1]])
        anchor = nodes[start].component_id
        tracks.append(
            PeakTrack(
                id_=f"track/{anchor}",
                anchor_id=anchor,
                component_ids=tuple(nodes[index].component_id for index in chain),
                links=tuple(links[a, b] for a, b in pairwise(chain)),
            )
        )
    return MatchingResult(series, tuple(tracks), objective, len(edges))
