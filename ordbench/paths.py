"""Storage locations. Raw data, features, run logs and model caches live outside the repository.

The data root is, in order of precedence:
  1. the SCB_DATA environment variable;
  2. D:/soft-corn-benchmark-data, if it exists (the original development machine, whose C: drive is full);
  3. <repo>/data (git-ignored).
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_LEGACY = Path("D:/soft-corn-benchmark-data")
if os.environ.get("SCB_DATA"):
    DATA = Path(os.environ["SCB_DATA"])
elif os.name == "nt" and _LEGACY.exists():
    DATA = _LEGACY
else:
    DATA = REPO / "data"

RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RUNS = DATA / "results"
FIGURES = REPO / "figures"
TABLES = REPO / "results"

os.environ.setdefault("HF_HOME", str(DATA / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
