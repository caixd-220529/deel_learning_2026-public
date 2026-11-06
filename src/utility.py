"""Future agents: maintain this file's implementation memory in this region.

# 2026-11-06

Canonical implementation moved here from qichen/src/utility.py. Legacy qichen modules re-export these APIs.
This public module depends only on sibling public modules and installed third-party packages.

Shared utilities for experiment scripts.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path


# region Path validation


def validate_directory(path: Path, description: str) -> None:
    """Raise a clear error when a required directory is unavailable."""

    if not path.is_dir():
        raise FileNotFoundError(f"{description} does not exist or is not a directory: {path}")


# endregion


# region Output archiving


def archive_existing_output(output_dir: Path) -> Path | None:
    """Move an existing run into a timestamped history directory."""

    if not output_dir.is_dir():
        return None

    stale_entries: list[Path] = [entry for entry in output_dir.iterdir() if entry.name != "history"]
    if not stale_entries:
        return None

    entry_mtimes: list[float] = []
    for entry in stale_entries:
        candidate_paths: list[Path] = [entry]
        if entry.is_dir():
            candidate_paths.extend(entry.rglob("*"))
        entry_mtimes.extend(path.stat().st_mtime for path in candidate_paths if path.exists())

    latest_mtime: float = max(entry_mtimes, default=output_dir.stat().st_mtime)
    history_root: Path = output_dir / "history"
    timestamp: str = datetime.fromtimestamp(latest_mtime, tz=UTC).astimezone().strftime("%Y-%m-%d-%H-%M-%S")
    history_dir: Path = history_root / timestamp
    suffix: int = 1
    while history_dir.exists():
        history_dir = history_root / f"{timestamp}-{suffix:02d}"
        suffix += 1

    history_dir.mkdir(parents=True)
    for entry in stale_entries:
        shutil.move(str(entry), str(history_dir / entry.name))
    return history_dir


# endregion
