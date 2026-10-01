"""Streamlit entrypoint for KBMOD search planning (recommend a configuration, evaluate a grid)."""

import sys
from pathlib import Path

# Hosted deployments run this file directly; make the src layout importable without an install.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kbmod_angle_rate_tool.kbmod_rate_angle_interactive import run_app  # noqa: E402


if __name__ == "__main__":
    run_app()
