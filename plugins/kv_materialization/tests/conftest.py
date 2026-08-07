"""Test configuration for the standalone plugin package."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(PACKAGE_ROOT))
