"""Attach notebook decompositions while preserving the default VAMAS spectrum tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scripts.ag_hopg_sources import (
    RECIPES,
    ROOT,
    dat_exports,
    measurements,
    phase_sources,
    read_parameters,
    source_record,
    stable_id,
)

from app.orchestration import AppOrchestrator
from app.parameters import AppParameters
from core.collection import CoreCollection
from formats import parse_spectrum_file


def prepare(data_root: Path, output: Path) -> None:
    folder = data_root / "spectra/ag_hopg/Ag HOPG last"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = [folder / "Sample 2 after atm.vms", folder / "New-HOPG-after-clb.vms"]
    app = AppOrchestrator(CoreCollection(), AppParameters())
    source_ids = {}
    all_sources = []
    for path in paths:
        all_sources.extend(measurements(path))
        for index, spectrum in enumerate(parse_spectrum_file(path), 1):
            sid = stable_id(f"native/{path.name}/block/{index}")
            app.create_spectrum(spectrum.x, spectrum.y, spectrum_id=sid)
            app.set_metadata(sid, spectrum.metadata)
            source_ids[path, index] = sid
    anchors = dat_exports(folder, all_sources)
    mappings, series_report = [], []
    for recipe in RECIPES:
        selected = phase_sources(recipe, all_sources)
        for region in ("Ag3d", "C1s"):
            source_path = folder / (
                f"{recipe.prefix} {region} hv20 "
                + ("dub.csv" if recipe.silver_blocked and region == "Ag3d" else "regions.csv")
            )
            parameters = read_parameters(source_path)
            region_anchors = [
                anchor
                for anchor in anchors
                if any(
                    anchor["vamas_path"] == str(source.path) and anchor["block"] == source.block
                    for source in selected[region]
                )
                and anchor["fit_bounds"] is not None
            ]
            if not region_anchors:
                raise ValueError(f"Missing DAT fit-window anchor for {recipe.key}/{region}")
            low = max(anchor["fit_bounds"][0] for anchor in region_anchors)
            high = min(anchor["fit_bounds"][1] for anchor in region_anchors)
            ids, expected_main = [], []
            for index, source in enumerate(selected[region]):
                sid = source_ids[source.path, source.block]
                ids.append(sid)
                spectrum = app.query.get_spectrum_dto(sid)
                indices = np.flatnonzero((spectrum.x >= low - 1e-8) & (spectrum.x <= high + 1e-8))
                start, stop = int(indices[0]), int(indices[-1] + 1)
                rid = stable_id(f"native/{recipe.key}/{region}/{index}/region")
                app.create_region(sid, start, stop, mode="index", region_id=rid)
                if region == "Ag3d":
                    row_indices = (
                        (index, index + recipe.steps)
                        if recipe.silver_blocked
                        else (2 * index, 2 * index + 1)
                    )
                else:
                    row_indices = (3 * index, 3 * index + 1, 3 * index + 2)
                # Keep measured counts untouched. Convert CSV areas to that same
                # acquisition scale instead of changing the spectrum intensities.
                scale = source.dwell * source.scans
                for row_number in row_indices:
                    row = parameters[row_number]
                    pid = stable_id(f"native/{recipe.key}/{region}/row/{row_number}")
                    app.create_peak(
                        rid,
                        "pseudo-voigt",
                        peak_id=pid,
                        parameters={
                            "amp": row["Area"] * scale,
                            "cen": row["Position"],
                            "sig": row["FWHM"] / 2,
                            "frac": row["GL"],
                        },
                    )
                    app.rename_component(pid, f"{region} CSV row {row_number}")
                expected_main.append(
                    stable_id(f"native/{recipe.key}/{region}/row/{row_indices[0]}")
                )
                # Estimate only the missing background; preserve all peak parameters.
                _, components = app.query.get_region_dto_repr(rid)
                from core.evaluation import component_y

                xx, yy = spectrum.x[start:stop], spectrum.y[start:stop]
                profile = sum(component_y(component, xx) for component in components)
                residual = yy - profile
                app.create_background(
                    rid,
                    "shirley",
                    parameters={
                        "i1": max(0, float(residual[:3].mean())),
                        "i2": max(0, float(residual[-3:].mean())),
                    },
                    background_id=stable_id(f"native/{recipe.key}/{region}/{index}/background"),
                )
                mappings.append(
                    {
                        "phase": recipe.label,
                        "line": region,
                        "step": index + 1,
                        "spectrum_id": sid,
                        "file": str(source.path.relative_to(data_root)),
                        "group": source.group,
                        "name": source.name,
                        "block_1_based": source.block,
                        "csv": source_path.name,
                        "csv_rows_0_based": list(row_indices),
                        "area_to_measured_counts_scale": scale,
                        "fit_bounds": [low, high],
                    }
                )
            series_id = app.create_post_processing_series(f"{recipe.label} — {region} hv20", ids)
            app.run_post_processing(series_id)
            result = app.query.get_post_processing_series(series_id).series.result
            assert result is not None
            main = next(track for track in result.tracks if expected_main[0] in track.component_ids)
            app.name_post_processing_track(
                series_id, main.id_, "Ag3d 5/2" if region == "Ag3d" else "C1s main"
            )
            matches = sum(pid in main.component_ids for pid in expected_main)
            if matches != recipe.steps:
                raise ValueError(
                    f"Main track deviates from notebook selection: {recipe.key}/{region}"
                )
            series_report.append(
                {
                    "phase": recipe.label,
                    "line": region,
                    "steps": len(ids),
                    "main_reference_matches": matches,
                }
            )
    for series_id in tuple(app.query.get_post_processing_series_ids()):
        app.remove_post_processing_series(series_id)
    app.dump_collection(output / "ag-hopg-with-decompositions.json.gz")
    report = {
        "structure": "Original VAMAS file/group/spectrum names, block order and individual spectra retained",
        "measurements": "Original parser x/y arrays unchanged, including intensity scale (counts)",
        "decompositions": "136 individual Ag3d/C1s hv20 spectra receive 340 CSV peaks and estimated Shirley backgrounds",
        "parameters": "CSV centers, FWHM and GL unchanged; amp=CSV Area * signal collection time * scans",
        "alignment": "Adjacent numbered acquisitions and notebook CSV row recipes; independently checked DAT anchors; intermediate rows use order",
        "backgrounds": "Estimated, not original CasaXPS backgrounds; current pseudo-Voigt is an approximation of unexported original profile family",
        "spectra": len(source_ids),
        "fitted_spectra": len(mappings),
        "sources": [
            source_record(path, data_root)
            for path in (
                data_root / "plots.ipynb",
                *paths,
                *sorted(folder.glob("*after*hv20*.csv")),
            )
        ],
        "series": series_report,
        "mapping": mappings,
    }
    (output / "provenance.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        Path(__file__).with_name("ag_hopg_native_readme.md").read_text(), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "spectra": len(source_ids),
                "fitted_spectra": len(mappings),
                "validated_series": len(series_report),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "ai-xps-data")
    parser.add_argument("--output", type=Path, default=ROOT / "build/manual-check/ag-hopg")
    args = parser.parse_args()
    prepare(args.data_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
