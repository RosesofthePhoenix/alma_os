"""Centralized service registry for alma_os.

Services are placeholders for now and will be populated as implementations are
added. Import `registry` or use `get_registry()` to access shared services.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from alma.engine.muse_stream_manager import MuseStreamManager
from alma.engine.state_engine import StateEngine
from alma.integrations.turrell_runner import TurrellRunner
from alma.integrations.spotify_now_playing_logger import SpotifyNowPlayingLogger


@dataclass
class ServiceRegistry:
    state_engine: StateEngine = field(default_factory=StateEngine)
    muse_stream_manager: MuseStreamManager = field(default_factory=MuseStreamManager)
    turrell_runner: TurrellRunner = field(default_factory=TurrellRunner)
    spotify_logger: SpotifyNowPlayingLogger = field(default_factory=SpotifyNowPlayingLogger)


# Singleton-like shared registry instance
registry = ServiceRegistry()


def get_registry() -> ServiceRegistry:
    """Return the shared registry instance."""
    return registry


__all__ = ["ServiceRegistry", "registry", "get_registry"]

