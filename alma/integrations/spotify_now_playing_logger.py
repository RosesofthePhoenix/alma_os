"""Read-only Spotify Now Playing logger."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from spotipy import Spotify, SpotifyException
from spotipy.oauth2 import SpotifyOAuth

from alma import config
from alma.engine import storage


class SpotifyNowPlayingLogger:
    def __init__(self, poll_interval: float = 2.0) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._last_error: Optional[str] = None
        self._latest: Optional[Dict[str, object]] = None
        self._session_id: Optional[str] = None
        self._poll_interval = poll_interval
        self._lock = threading.Lock()

    def start(self, session_id: Optional[str]) -> None:
        with self._lock:
            if self._running:
                return
            self._session_id = session_id
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._running = True
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        with self._lock:
            self._running = False

    def get_latest(self) -> Optional[Dict[str, object]]:
        with self._lock:
            return dict(self._latest) if self._latest else None

    def status(self) -> Dict[str, object]:
        with self._lock:
            return {
                "running": self._running,
                "last_error": self._last_error,
                "latest": dict(self._latest) if self._latest else None,
            }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
                self._last_error = None
            except Exception as exc:  # pragma: no cover - defensive
                with self._lock:
                    self._last_error = str(exc)
            time.sleep(self._poll_interval)
        with self._lock:
            self._running = False

    def _poll_once(self) -> None:
        creds = self._load_creds()
        if not creds:
            with self._lock:
                self._last_error = "Missing Spotify credentials."
            return
        preferred_device = creds.get("preferred_device") or None
        session_id = self._session_id
        auth = SpotifyOAuth(
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            redirect_uri=creds["redirect_uri"],
            scope="user-read-playback-state user-read-currently-playing",
            open_browser=False,
        )
        sp = Spotify(auth_manager=auth)
        try:
            data = sp.current_playback(additional_types="track")
        except SpotifyException as exc:
            with self._lock:
                self._last_error = f"Spotify auth/playback error: {exc}"
            return
        except Exception as exc:
            with self._lock:
                self._last_error = f"Spotify error: {exc}"
            return

        ts = time.time()
        if data is None:
            # Try to surface device info for troubleshooting
            device_note = None
            try:
                devices = sp.devices().get("devices", [])
                if devices:
                    names = ", ".join(d.get("name", "?") for d in devices)
                    device_note = f"No active playback device. Seen devices: {names}"
                    if preferred_device:
                        match = [d for d in devices if preferred_device.lower() in (d.get("name","").lower())]
                        if match:
                            device_note = f"No playback. Preferred device '{preferred_device}' seen but inactive."
                        else:
                            device_note = f"No playback. Preferred device '{preferred_device}' not seen. Devices: {names}"
                else:
                    device_note = "No active playback device."
            except Exception:
                device_note = "No active playback device."
            payload = {
                "ts": ts,
                "is_playing": False,
                "track_id": None,
                "track_name": None,
                "artists": None,
                "album": None,
                "context_uri": None,
                "device_name": None,
                "progress_ms": None,
                "duration_ms": None,
                "mode": self._read_mode(),
            }
            if device_note:
                with self._lock:
                    self._last_error = device_note
        else:
            item = data.get("item") or {}
            artists = item.get("artists") or []
            payload = {
                "ts": ts,
                "is_playing": bool(data.get("is_playing")),
                "track_id": item.get("id"),
                "track_name": item.get("name"),
                "artists": ", ".join(a.get("name", "") for a in artists),
                "album": (item.get("album") or {}).get("name"),
                "context_uri": (data.get("context") or {}).get("uri"),
                "device_name": (data.get("device") or {}).get("name"),
                "progress_ms": data.get("progress_ms"),
                "duration_ms": item.get("duration_ms"),
                "mode": self._read_mode(),
            }

        with self._lock:
            self._latest = payload

        self._append_ndjson(payload)
        if session_id:
            try:
                storage.insert_spotify_row(
                    ts=payload["ts"],
                    session_id=session_id,
                    is_playing=payload["is_playing"],
                    track_id=payload.get("track_id") or "",
                    track_name=payload.get("track_name") or "",
                    artists=payload.get("artists") or "",
                    album=payload.get("album") or "",
                    context_uri=payload.get("context_uri") or "",
                    device_name=payload.get("device_name") or "",
                    progress_ms=payload.get("progress_ms"),
                    duration_ms=payload.get("duration_ms"),
                    mode=payload.get("mode") or "",
                )
            except Exception:
                pass

    def _append_ndjson(self, payload: Dict[str, object]) -> None:
        try:
            path = config.SPOTIFY_PLAYBACK_LOG_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    @staticmethod
    def _load_creds() -> Optional[Dict[str, str]]:
        profile_path = Path(__file__).resolve().parents[2] / "profiles" / "default.json"
        try:
            with profile_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        cid = data.get("SPOTIPY_CLIENT_ID")
        secret = data.get("SPOTIPY_CLIENT_SECRET")
        redirect = data.get("SPOTIPY_REDIRECT_URI")
        preferred_device = data.get("SPOTIFY_PREFERRED_DEVICE")
        if cid and secret and redirect:
            return {
                "client_id": cid,
                "client_secret": secret,
                "redirect_uri": redirect,
                "preferred_device": preferred_device,
            }
        return None

    @staticmethod
    def _read_mode() -> str:
        try:
            with config.MODE_FILE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return str(data.get("mode", "OFF"))
        except Exception:
            return "OFF"


__all__ = ["SpotifyNowPlayingLogger"]

