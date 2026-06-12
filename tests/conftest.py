"""conftest.py — shared pytest configuration."""
import sys
from pathlib import Path

# Add src and root to path so imports work from tests/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
