# scripts/

One-off / unsupported utilities (data conversion, external-app automation).

- Not part of the application runtime.
- Extra dependencies (e.g. `pyautogui`, `uiautomation`) are **not** declared in
  `pyproject.toml`.
- Paths inside these scripts are often machine-specific (Desktop folders, etc.).
- `prepare_ag_hopg_native.py`: prepare the current local Ag/HOPG walkthrough.
  Run `uv run python -m scripts.prepare_ag_hopg_native`; output defaults to
  `build/manual-check/ag-hopg/` (choose a fresh `--output` directory for reruns).
  Preserves the original VAMAS file/group/spectrum structure and measured arrays;
  attaches CSV decompositions to individual Ag3d/C1s hv20 spectra. Produces one
  document without preconfigured series, instructions and a provenance report.
  `ag_hopg_sources.py` provides its source recipes and acquisition metadata.

Prefer adding new one-offs here rather than in `app/` or `core/`.
