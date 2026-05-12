import sys
from pathlib import Path

# Add the src directory to sys.path so that 'wildfire_simulator' can be imported
src_path = Path(__file__).resolve().parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
