"""Configuration constants and default paths for alma_os."""

from pathlib import Path
from typing import List

# Project root inferred from this file's location.
ROOT_DIR = Path(__file__).resolve().parent.parent

# Default paths
SESSIONS_DIR = ROOT_DIR / "sessions"
SESSIONS_CURRENT_DIR = SESSIONS_DIR / "current"
DATA_DIR = ROOT_DIR / "data"
BASELINES_DIR = ROOT_DIR / "baselines"

STATE_STREAM_PATH = SESSIONS_CURRENT_DIR / "state_stream.ndjson"
SPOTIFY_PLAYBACK_LOG_PATH = SESSIONS_CURRENT_DIR / "spotify_playback.ndjson"
MODE_FILE_PATH = SESSIONS_CURRENT_DIR / "adaptive_mode.json"
DB_PATH = DATA_DIR / "alma.db"
BASELINE_DEFAULT_PATH = BASELINES_DIR / "baseline_global_muse_v1_revised.json"

# Defaults
CANONICAL_Q: str = "vibe_focus"
NEUROMETRICS_Q_OPTIONS: List[str] = ["vibe_focus", "vibe", "abs", "vibe_focus_e"]
BUCKET_MINUTES_DEFAULT: int = 10
MIN_VALID_FRACTION_FOR_LABEL: float = 0.6


def ensure_required_paths() -> None:
    """Create required runtime directories if they do not exist."""
    for path in {SESSIONS_CURRENT_DIR, DATA_DIR}:
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ROOT_DIR",
    "SESSIONS_DIR",
    "SESSIONS_CURRENT_DIR",
    "DATA_DIR",
    "BASELINES_DIR",
    "STATE_STREAM_PATH",
    "SPOTIFY_PLAYBACK_LOG_PATH",
    "MODE_FILE_PATH",
    "DB_PATH",
    "BASELINE_DEFAULT_PATH",
    "CANONICAL_Q",
    "NEUROMETRICS_Q_OPTIONS",
    "BUCKET_MINUTES_DEFAULT",
    "MIN_VALID_FRACTION_FOR_LABEL",
    "ensure_required_paths",
]

