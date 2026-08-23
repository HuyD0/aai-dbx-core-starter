import sys
from pathlib import Path

# Tests import the lesson sources (scripts/notebook_content.py) directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
