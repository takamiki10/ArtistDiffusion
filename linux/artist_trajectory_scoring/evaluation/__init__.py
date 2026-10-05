"""Evaluation scripts for ArtistDiffusion."""

# Support direct execution and retain historical checkpoint module names.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[1]
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from dependencies import PROJECT_ROOT
