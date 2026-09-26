# Contributing

## Branches

- **main** — stable release with reviewed, ready features.
- **dev** — the branch day-to-day work lands on.

Start every change from an up-to-date `dev`. Name the branch with a prefix and a short English kebab-case description:

| Prefix | Use |
| --- | --- |
| `feat/` | A new feature |
| `fix/` | A bug fix |
| `refactor/` | Internal restructuring |

```bash
git checkout dev
git pull origin dev
git checkout -b feat/my-new-feature
```

Keep commits small and write the messages in English.

## Pull requests

Push the branch and open a pull request into **`dev`**:

```bash
git push origin feat/my-new-feature
```

Name the issues the pull request closes in the description:

```
Closes #12
```

Pull requests into `dev` and `main` run GitHub Actions (`.github/workflows/ci.yml`): `ruff check`, `ruff format --check`, `ty check` (including `tests/`), and `pytest --cov`.

Run the same checks locally before pushing:

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest --cov --cov-report=term-missing
```

After CI is green, the pull request is reviewed and merged into `dev`.

When a set of changes is ready to release, open a pull request from `dev` into `main`.

## Building

Python 3.11 or newer is required. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

### Development with uv

1. Install [uv](https://docs.astral.sh/uv/).
2. Clone the repository and install the application together with the development tools (tests, lint, type check):

   ```bash
   git clone https://github.com/XPS-Development/XPS-AI.git
   cd XPS-AI
   uv sync --group dev
   ```

3. Run the application:

   ```bash
   uv run python main.py
   ```

For matplotlib-based debugging (`debug/viewer.py`), also sync the interactive group:

```bash
uv sync --group dev --group interactive
```

### Windows installer

The installer is built on 64-bit Windows. PyInstaller freezes the application, and [Inno Setup 6](https://jrsoftware.org/isinfo.php) wraps that folder into `xps-ai_<version>_x64.exe`.

1. Install the runtime dependencies:

   ```powershell
   uv sync
   ```

2. Copy the version from `pyproject.toml` into the installer script:

   ```powershell
   uv run python build/sync_version.py
   ```

3. Freeze the application. `build/XPS-AI.spec` packages `assets/models/model.onnx` and `assets/icons` from the repository, using paths relative to the spec file. PyInstaller is pulled in for this step only, outside the locked dependency groups:

   ```powershell
   cd build
   uv run --with pyinstaller pyinstaller XPS-AI.spec --workpath pyi_build --distpath pyi_dist
   ```

   Git Bash can run the same step with `build/build_app_pyi.sh`. The frozen app is `build/pyi_dist/XPS-AI/XPS-AI.exe`.

4. Compile `build/XPS-AI.iss` with the Inno Setup compiler. Paths in that script are relative to `build/`:

   ```powershell
   & "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" build\XPS-AI.iss
   ```

   The installer is written to `build\iss_build\xps-ai_<version>_x64.exe`. The version in the filename is the `MyAppVersion` define, which step 2 copies from `pyproject.toml`.
