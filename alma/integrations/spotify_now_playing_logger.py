"""Read-only Spotify Now Playing logger."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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
        self._current_track_session_id: Optional[int] = None
        self._current_track_id: Optional[str] = None
        self._current_start_ts: Optional[float] = None
        self._last_update_ts: float = 0.0
        self._analysis_cache: Dict[str, dict] = {}

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
            now = time.time()
            try:
                self._finalize_current(now)
            except Exception:
                pass
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        with self._lock:
            self._running = False
            self._current_track_session_id = None
            self._current_track_id = None
            self._current_start_ts = None

    def force_poll(self) -> None:
        """Run one poll immediately to refresh latest/NDJSON without waiting for thread loop."""
        try:
            self._poll_once()
        except Exception:
            # best-effort; ignore to keep UI responsive
            pass

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
        try:
            self._update_track_session(payload, session_id=session_id)
        except Exception:
            # Defensive: keep logging alive even if track session update fails
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

    def _update_track_session(self, payload: Dict[str, object], session_id: Optional[str]) -> None:
        track_id = payload.get("track_id")
        title = payload.get("track_name") or ""
        artist = payload.get("artists") or ""
        album = payload.get("album") or ""
        is_playing = bool(payload.get("is_playing"))
        now_ts = payload.get("ts") or time.time()

        # End current session if playback stopped or track lost
        if not is_playing or not track_id:
            self._finalize_current(now_ts)
            return

        # Track changed
        if self._current_track_id and track_id != self._current_track_id:
            self._finalize_current(now_ts)

        # Start if none
        if self._current_track_session_id is None:
            tsid = storage.start_track_session(
                session_id=session_id,
                track_id=track_id,
                title=title,
                artist=artist,
                album=album,
                start_ts=now_ts,
            )
            self._current_track_session_id = tsid
            self._current_track_id = track_id
            self._current_start_ts = now_ts
            self._last_update_ts = now_ts
            # fetch audio analysis lazily
            self._maybe_fetch_analysis(track_id)
            return

        # Periodic metric update while playing
        if now_ts - self._last_update_ts >= 15.0 and self._current_track_session_id:
            storage.update_track_session_metrics(self._current_track_session_id, end_ts=now_ts)
            self._last_update_ts = now_ts

    def _finalize_current(self, end_ts: float) -> None:
        if self._current_track_session_id:
            storage.update_track_session_metrics(self._current_track_session_id, end_ts=end_ts)
            # section-level metrics
            try:
                self._compute_sections(self._current_track_session_id, end_ts)
            except Exception:
                pass
        self._current_track_session_id = None
        self._current_track_id = None
        self._current_start_ts = None
        self._last_update_ts = 0.0

    def _maybe_fetch_analysis(self, track_id: str) -> None:
        if track_id in self._analysis_cache:
            return
        creds = self._load_creds()
        if not creds:
            return
        auth = SpotifyOAuth(
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            redirect_uri=creds["redirect_uri"],
            scope="user-read-playback-state user-read-currently-playing",
            open_browser=False,
        )
        sp = Spotify(auth_manager=auth)
        try:
            analysis = sp.audio_analysis(track_id)
            self._analysis_cache[track_id] = analysis
        except Exception:
            self._analysis_cache[track_id] = {}

    def _compute_sections(self, track_session_id: int, end_ts: float) -> None:
        if not self._current_start_ts or not self._current_track_id:
            return
        start_ts = self._current_start_ts
        track_id = self._current_track_id
        title = ""
        artist = ""
        analysis = self._analysis_cache.get(track_id) or {}
        sections = []
        sec_data = analysis.get("sections") or []
        if sec_data:
            for idx, sec in enumerate(sec_data):
                s = sec.get("start", 0.0)
                d = sec.get("duration", 0.0)
                sections.append((idx, f"Section {idx+1}", s, s + d))
        else:
            # fallback thirds
            total = max(end_ts - start_ts, 1.0)
            step = total / 3.0
            sections = [(i, f"Part {i+1}", i * step, (i + 1) * step) for i in range(3)]

        rows = []
        for idx, label, rel_start, rel_end in sections:
            abs_start = start_ts + rel_start
            abs_end = start_ts + rel_end
            means = storage._compute_means_for_window(abs_start, abs_end, session_id=self._session_id)
            rows.append(
                {
                    "idx": idx,
                    "label": label,
                    "start_ts": abs_start,
                    "end_ts": abs_end,
                    "mean_HCE": means.get("mean_HCE", 0.0),
                    "mean_Q": means.get("mean_Q", 0.0),
                    "mean_X": means.get("mean_X", 0.0),
                }
            )
        storage.upsert_track_sections(track_session_id, track_id, title, artist, rows)


__all__ = ["SpotifyNowPlayingLogger"]

