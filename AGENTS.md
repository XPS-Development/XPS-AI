# AGENTS.md — XPS-AI

Guidance for AI agents working in this repository.

## What this is

Desktop XPS (X-ray Photoelectron Spectroscopy) analysis app: spectrum import,
region/peak/background fitting (lmfit), NN segmentation (ONNX), PySide6 +
pyqtgraph UI. Entry point: `main.py`.

## Architecture (dependency direction)

```
ui/  →  app/  →  core/
              ↘  formats/  →  core/
              ↘  inference/  →  core/
debug/ → core/evaluation → core/dto   (notebooks / interactive only)
```

| Package | Role |
|---------|------|
| `core/` | Domain: spectra, regions, components, math models, services, array numerics, immutable DTOs, model evaluation, fitting, serialization, CSV export |
| `app/` | Application: orchestrator, commands/undo, usecases, DTOService, adapters over core/formats/inference |
| `ui/` | Presentation: Qt widgets, controller wrapper, signals |
| `formats/` | Spectrum file parsers (casa, dat, VAMAS) and extension dispatcher |
| `inference/` | ONNX segmenter pipeline (preprocess → adapter → postprocess) |
| `debug/` | Matplotlib viewer for notebooks / exploration (`uv sync --group interactive`) |
| `scripts/` | One-off unsupported utilities (not app runtime) |
| `model/` | Training code for the segmenter — **unmaintained / broken; do not extend** |

**Hard rules**

- `core/` must not import `app/`, `ui/`, `formats/`, or `inference/`.
  Array helpers live in `core/numerics.py`.
- `app/` must not import `ui/`. Qt in `app/` only via optional lazy import in
  `error_dump.py`.
- `ui/` talks to the domain through `ControllerWrapper` → `AppOrchestrator`.
  Do not call `core.evaluation` / `ModelRegistry` / mutate core objects from
  widgets when adding new features — put that behind `app/`.
- Mutations go through `Change` → `CommandExecutor` (undo/redo). Do not bypass
  with direct collection/service writes unless document lifecycle
  (`new_collection`, load/replace) genuinely requires it.

**`app/` adapters are not duplicates**

`app/optimization.py`, `app/serialization.py`, `app/csv_export.py`,
`app/automatization.py` wrap library modules and return `Change` objects or track
dirty state. Keep that boundary; do not merge layers or copy logic both ways.

## Tooling

Use **uv** for everything. Do not use bare `python` / `pip` / `pytest`.

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format .
uv run ty check
uv run pytest
```

- Python ≥ 3.11; line length 100; NumPy-style docstrings; type annotations
  required on public APIs (`ruff` ANN + D).
- `model/` and `notebooks/` are excluded from ruff/ty — leave them alone unless
  the task is specifically training/export.
- After editing Python: run `ruff check` (and `ruff format` if needed) on touched
  paths; run relevant pytest modules.

## Where to put new code

| Kind of change | Put it in |
|----------------|-----------|
| Entity / service / math model | `core/` |
| Immutable DTO projections | `core/dto.py` |
| Build DTOs from `CoreContext` | `app/dto_service.py` |
| Stateless model evaluation | `core/evaluation.py` |
| Pure array helpers (interp, index lookup) | `core/numerics.py` |
| Lmfit optimization | `core/fitting/` |
| Document serialization / CSV export | `core/io/` |
| Workflow that returns `Change`s | `app/usecases/` then wire from orchestrator |
| Undoable mutation | `app/command/changes.py` + `commands.py` + registry |
| Initial parameter guess | model `guess_initial` on `core/math_models` (+ helpers in `guess_helpers.py`) |
| File format parse | `formats/` + dispatcher in `__init__.py` |
| ONNX segmenter / inference pipeline | `inference/` |
| Qt widget / dialog / plot | `ui/` — presentation only |
| Matplotlib debug / notebook plotting | `debug/` — not `ui/` |
| One-off scripts | `scripts/` — do not grow top-level packages with scripts |

Prefer extending `EditingUseCases` / `AnalysisUseCases` over growing
`AppOrchestrator` further. Extract `QueryService` / `AppParameters` out of
`orchestration.py` before adding more façade methods if you need those types.

## Known debt — do not make worse

These are intentional temporary states. Avoid reinforcing them.

1. **Do not reintroduce a SPECS parser.** Notebook matplotlib plotting lives in
   `debug/viewer.py`; one-offs in `scripts/`.
2. **UI leakage exists** (`plot_area` → `spectrum_bundle`, dialogs →
   `ModelRegistry`). New features must not add more `core` imports in
   `ui/` for business logic.
3. **`model/` cannot train after `uv sync`** (no torch/lightning deps; broken
   imports; no ONNX export script). App consumes
   `assets/models/model.onnx` only. Do not “fix” training casually; treat
   ONNX as an external artifact unless the task is a full training/export
   revive.
4. **CI** is `.github/workflows/ci.yml` (`ruff` / `ty` / `pytest` on PRs to `dev`/`main`).
   Keep that gate green; do not weaken excludes for unmaintained training scripts without fixing them.

## Commands / undo notes

- New change types need: dataclass in `changes.py`, command in `commands.py`,
  entry in `create_default_registry()`.
- UI refresh flags in `ui/controller.py` use `isinstance` on concrete command
  classes — if you add a command that should not full-refresh the UI, update
  that mapping or over-invalidation will happen.

## Git / PR

- Branch from `dev`; PR into `dev`. See `CONTRIBUTING.md`.
- Commit only when asked. Messages in English, concise, focus on why.
- Do not commit secrets, `error_dumps/`, or notebooks (gitignored).

## Tests

Mirror package layout under `tests/`. Prefer testing `core/` and `app/` without
Qt. UI tests need `QApplication`; keep them thin. When changing behavior, update
or add tests in the matching `tests/<package>/` tree.
