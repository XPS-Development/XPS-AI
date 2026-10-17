"""CSV serialization of the same full precision projections used by the UI."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from core.post_processing.analysis import AnalyticalTableState

if TYPE_CHECKING:
    from core.dto import AnalysisTable
    from core.post_processing.state import PostProcessingSeries


def table_csv(table: AnalysisTable, *, separator: str = ",", precision: int | None = None) -> str:
    """Serialize IDs, units, values and a status/reason for every column."""
    if len(separator) != 1 or separator in "\r\n":
        raise ValueError("CSV separator must be one character other than a newline")
    if precision is not None and not 0 <= precision <= 15:
        raise ValueError("CSV precision must be between 0 and 15")
    stream = StringIO()
    writer = csv.writer(stream, delimiter=separator, lineterminator="\n")
    writer.writerow(
        [
            "spectrum_id",
            "region_id",
            "component_id",
            *(
                part
                for column in table.columns
                for part in (
                    f"{column.id_} [{column.unit}]" if column.unit else column.id_,
                    f"{column.id_}:status",
                    f"{column.id_}:reason",
                )
            ),
        ]
    )
    for row in table.rows:
        values: list[str | float | int | None] = [row.spectrum_id, row.region_id, row.component_id]
        for cell in row.cells:
            value = cell.value
            if isinstance(value, float) and precision is not None:
                value = f"{value:.{precision}f}"
            values.extend((value, cell.status, cell.reason))
        writer.writerow(values)
    return stream.getvalue()


def calculation_description(
    series: PostProcessingSeries | AnalyticalTableState,
) -> dict[str, object]:
    """Describe operands, units and reference IDs without inventing implicit corrections."""
    if isinstance(series, AnalyticalTableState):
        return {
            "analytical_table": asdict(series),
            "definitions": [{**asdict(metric), "unit": metric.unit} for metric in series.metrics],
            "area_basis": "Original model areas; no RSF or concentration correction",
        }
    return {
        "series_id": series.id_,
        "series_name": series.name,
        "spectrum_ids": series.spectrum_ids,
        "input_fingerprint": series.input_fingerprint,
        "area_basis": "Original fitted model areas; no sensitivity factors or concentration correction",
        "spectrum_area_fraction_denominator": "Sum of all eligible peak areas in that spectrum",
        "track_anchors": {}
        if series.result is None
        else {track.id_: track.anchor_id for track in series.result.tracks},
        "matching_snapshot": None if series.result is None else asdict(series.result),
    }


def export_table(
    path: str | Path,
    table: AnalysisTable,
    series: PostProcessingSeries | AnalyticalTableState,
    *,
    separator: str = ",",
    precision: int | None = None,
    source_series: dict[str, PostProcessingSeries] | None = None,
) -> None:
    """Write values and a companion JSON description of definitions and columns."""
    csv_text = table_csv(table, separator=separator, precision=precision)
    description = {
        **calculation_description(series),
        "precision": precision,
        "source_series": {sid: asdict(saved) for sid, saved in (source_series or {}).items()},
        "columns": [asdict(column) for column in table.columns],
    }
    metadata_text = json.dumps(description, indent=2, ensure_ascii=False, allow_nan=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.with_suffix(".calculations.json").write_text(metadata_text + "\n", encoding="utf-8")
    target.write_text(csv_text, encoding="utf-8")
