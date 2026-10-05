"""Shared pipeline modules and import compatibility for existing checkpoints.

Scripts keep their established top-level module names because saved model
metadata and multiprocessing workers refer to them. Importing this package
makes the organized source directories available without root-level shims.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRECTORIES = (PROJECT_ROOT, PROJECT_ROOT / "dependencies",
                      PROJECT_ROOT / "evaluation", PROJECT_ROOT / "benchmark")
for directory in reversed(SOURCE_DIRECTORIES):
    value = str(directory)
    if value not in sys.path:
        sys.path.insert(0, value)


def script_path(filename: str) -> Path:
    """Resolve an entry point independently of the caller's working directory."""
    for directory in SOURCE_DIRECTORIES:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No pipeline script named {filename!r}")
