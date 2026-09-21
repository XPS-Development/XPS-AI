# scripts/

One-off / unsupported utilities (data conversion, external-app automation).

- Not part of the application runtime.
- Extra dependencies (e.g. `pyautogui`, `uiautomation`) are **not** declared in
  `pyproject.toml`.
- Paths inside these scripts are often machine-specific (Desktop folders, etc.).

Prefer adding new one-offs here rather than in `app/` or `core/`.
