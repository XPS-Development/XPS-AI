"""Sync version between `pyproject.toml` and `build/XPS-AI.iss`.

Reads the canonical version from `[project].version` in `pyproject.toml` and updates the
`#define MyAppVersion "..."` line in the Inno Setup script.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SyncResult:
    """Result of a sync/check operation."""

    pyproject_version: str
    iss_version_before: str
    changed: bool


def _read_pyproject_version(pyproject_path: Path) -> str:
    """Read `[project].version` from a `pyproject.toml`.

    Parameters
    ----------
    pyproject_path
        Path to `pyproject.toml`.

    Returns
    -------
    str
        The version string.
    """
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError("Python 3.11+ is required (tomllib missing).") from e

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project")
    if not isinstance(project, dict) or "version" not in project:
        raise ValueError(f"Missing [project].version in {pyproject_path}")

    version = project["version"]
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"Invalid [project].version in {pyproject_path}: {version!r}")

    return version.strip()


_ISS_DEFINE_RE = re.compile(
    r'^(?P<prefix>\s*#define\s+MyAppVersion\s+")(?P<version>[^"]+)(?P<suffix>"\s*)$',
    re.MULTILINE,
)


def _read_iss_define_version(iss_text: str, iss_path: Path) -> str:
    """Extract the `MyAppVersion` define from an Inno Setup script.

    Parameters
    ----------
    iss_text
        Full `.iss` file contents.
    iss_path
        Path to the `.iss` file (only for error messages).

    Returns
    -------
    str
        The version string currently present in `#define MyAppVersion "..."`.
    """
    match = _ISS_DEFINE_RE.search(iss_text)
    if match is None:
        raise ValueError(f'Could not find `#define MyAppVersion "..."` in {iss_path} (expected one match).')
    return match.group("version").strip()


def sync_version(*, pyproject_path: Path, iss_path: Path, write: bool) -> SyncResult:
    """Sync version from `pyproject.toml` into an Inno Setup `.iss` file.

    Parameters
    ----------
    pyproject_path
        Path to `pyproject.toml`.
    iss_path
        Path to the Inno Setup script, e.g. `build/XPS-AI.iss`.
    write
        If True, write changes back to disk. If False, perform a dry-run.

    Returns
    -------
    SyncResult
        The sync outcome.
    """
    pyproject_version = _read_pyproject_version(pyproject_path)

    # Preserve original newline behavior (notably CRLF) by disabling newline translation.
    with iss_path.open("r", encoding="utf-8", newline="") as f:
        iss_text_before = f.read()

    iss_version_before = _read_iss_define_version(iss_text_before, iss_path)
    if iss_version_before == pyproject_version:
        return SyncResult(
            pyproject_version=pyproject_version,
            iss_version_before=iss_version_before,
            changed=False,
        )

    iss_text_after, n_subs = _ISS_DEFINE_RE.subn(
        rf"\g<prefix>{pyproject_version}\g<suffix>", iss_text_before, count=1
    )
    if n_subs != 1:
        raise ValueError(
            f"Expected to update exactly 1 `MyAppVersion` define in {iss_path}, updated {n_subs}."
        )

    if write:
        with iss_path.open("w", encoding="utf-8", newline="") as f:
            f.write(iss_text_after)

    return SyncResult(
        pyproject_version=pyproject_version,
        iss_version_before=iss_version_before,
        changed=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Sync version between pyproject.toml and build/XPS-AI.iss")
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pyproject.toml",
        help="Path to pyproject.toml (default: repo root pyproject.toml)",
    )
    parser.add_argument(
        "--iss",
        type=Path,
        default=Path(__file__).resolve().parent / "XPS-AI.iss",
        help="Path to Inno Setup .iss file (default: build/XPS-AI.iss)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if versions differ; do not write.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change; do not write.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args(argv)

    pyproject_path: Path = args.pyproject
    iss_path: Path = args.iss

    if not pyproject_path.is_file():
        raise FileNotFoundError(pyproject_path)
    if not iss_path.is_file():
        raise FileNotFoundError(iss_path)

    write = not (args.check or args.dry_run)
    result = sync_version(pyproject_path=pyproject_path, iss_path=iss_path, write=write)

    if args.check:
        if result.changed:
            print(
                f"Version mismatch: pyproject.toml={result.pyproject_version} "
                f"{iss_path.name}={result.iss_version_before}",
                file=sys.stderr,
            )
            return 1
        return 0

    if args.dry_run:
        if result.changed:
            print(
                f"Would update {iss_path} MyAppVersion: "
                f"{result.iss_version_before} -> {result.pyproject_version}"
            )
        else:
            print(f"No changes needed (version {result.pyproject_version}).")
        return 0

    if result.changed:
        print(
            f"Updated {iss_path} MyAppVersion: " f"{result.iss_version_before} -> {result.pyproject_version}"
        )
    else:
        print(f"No changes needed (version {result.pyproject_version}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
