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

_SPOTIFY_CACHE_PATH = config.ROOT_DIR / ".cache"
_FEATURE_CACHE_PATH = config.SESSIONS_CURRENT_DIR / "audio_features_cache.json"


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
        self._current_duration_sec: float = 0.0
        self._current_title: str = ""
        self._current_artist: str = ""
        self._last_section_compute_ts: float = 0.0
        self._warned_restricted = False

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
            self._last_section_compute_ts = 0.0

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
        if not self._warned_restricted:
            print("[spotify_logger] Spotify advanced features restricted (403)—using local personal resonance estimation", flush=True)
            self._warned_restricted = True
        print(f"[spotify_logger] Found refresh_token: {'yes' if creds.get('refresh_token') else 'no'}", flush=True)
        sp = self._get_spotify_client(creds)
        if not sp:
            with self._lock:
                self._last_error = "Spotify token refresh failed."
            return
        track_id_for_features = None
        preferred_device = creds.get("preferred_device") or None
        session_id = self._session_id
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
            track_id_for_features = payload.get("track_id")

        with self._lock:
            self._latest = payload

        # Advanced Spotify endpoints disabled; skip audio_features/audio_analysis

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
        refresh_token = data.get("SPOTIPY_REFRESH_TOKEN")
        if cid and secret and redirect:
            return {
                "client_id": cid,
                "client_secret": secret,
                "redirect_uri": redirect,
                "preferred_device": preferred_device,
                "refresh_token": refresh_token,
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

    @staticmethod
    def _seed_from_refresh(auth: SpotifyOAuth, refresh_token: Optional[str]) -> None:
        """Seed the cache from a stored refresh token to avoid interactive prompts."""
        if not refresh_token:
            return
        try:
            cache_path = Path(getattr(auth, "cache_path", ""))
            if cache_path.exists():
                return
        except Exception:
            pass
        try:
            token_info = auth.refresh_access_token(refresh_token)
            try:
                auth._save_token_info(token_info)
            except Exception:
                pass
        except Exception:
            # best-effort; avoid raising
            pass

    @staticmethod
    def _get_spotify_client(creds: Dict[str, str]) -> Optional[Spotify]:
        """Build a Spotify client using only refresh_token (non-interactive)."""
        refresh_token = creds.get("refresh_token")
        if not refresh_token:
            return None
        try:
            auth = SpotifyOAuth(
                client_id=creds["client_id"],
                client_secret=creds["client_secret"],
                redirect_uri=creds["redirect_uri"],
                scope=(
                    "playlist-read-private "
                    "user-read-playback-state "
                    "user-library-read "
                    "user-read-currently-playing "
                    "user-modify-playback-state"
                ),
                cache_path=None,
                open_browser=False,
            )
            token_info = auth.refresh_access_token(refresh_token)
            print("[spotify_logger] Access token refreshed successfully", flush=True)
            return Spotify(auth=token_info.get("access_token"))
        except Exception as exc:
            print(f"[spotify_logger] Token refresh failed: {exc}", flush=True)
            return None

    @staticmethod
    def _log_audio_features(sp: Spotify, track_id: str) -> None:
        # Deprecated: advanced features disabled.
        return

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
            progress_ms = payload.get("progress_ms") or 0
            start_ts_aligned = now_ts - (progress_ms / 1000.0)
            tsid = storage.start_track_session(
                session_id=session_id,
                track_id=track_id,
                title=title,
                artist=artist,
                album=album,
                start_ts=start_ts_aligned,
            )
            self._current_track_session_id = tsid
            self._current_track_id = track_id
            self._current_start_ts = start_ts_aligned
            self._last_update_ts = now_ts
            self._current_duration_sec = float((payload.get("duration_ms") or 0) / 1000.0)
            self._current_title = title
            self._current_artist = artist
            self._last_section_compute_ts = 0.0
            return

        # Periodic metric update while playing
        if now_ts - self._last_update_ts >= 15.0 and self._current_track_session_id:
            storage.update_track_session_metrics(self._current_track_session_id, end_ts=now_ts)
            self._last_update_ts = now_ts
            # attempt early section computation so UI fills within ~30-60s
            if now_ts - (self._last_section_compute_ts or 0.0) >= 30.0:
                try:
                    self._compute_sections(self._current_track_session_id, now_ts)
                    self._last_section_compute_ts = now_ts
                except Exception:
                    pass

    def _finalize_current(self, end_ts: float) -> None:
        if self._current_track_session_id:
            storage.update_track_session_metrics(self._current_track_session_id, end_ts=end_ts)
            # section-level metrics
            try:
                self._compute_sections(self._current_track_session_id, end_ts)
            except Exception:
                pass
            try:
                self._compute_waveform(self._current_track_session_id, end_ts)
                self._compute_waveform_points(self._current_track_session_id, end_ts)
            except Exception:
                pass
        self._current_track_session_id = None
        self._current_track_id = None
        self._current_start_ts = None
        self._last_update_ts = 0.0
        self._current_duration_sec = 0.0
        self._current_title = ""
        self._current_artist = ""

    def _maybe_fetch_analysis(self, track_id: str) -> None:
        creds = self._load_creds()
        if not creds:
            return
        sp = self._get_spotify_client(creds)
        if not sp:
            return
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
        title = self._current_title or ""
        artist = self._current_artist or ""
        analysis = self._analysis_cache.get(track_id) or {}
        sections = []
        duration = self._current_duration_sec or (end_ts - start_ts)
        if duration <= 0:
            duration = end_ts - start_ts
        sec_data = analysis.get("sections") or []
        if sec_data:
            for idx, sec in enumerate(sec_data):
                s = sec.get("start", 0.0)
                d = sec.get("duration", 0.0)
                sections.append((idx, f"Section {idx+1}", s, s + d))
        else:
            # fallback pseudo-sections (5 parts)
            total = max(duration, 1.0)
            n_parts = 5
            step = total / float(n_parts)
            sections = [(i, f"Part {i+1} (est.)", i * step, (i + 1) * step) for i in range(n_parts)]

        def _means_zero(m: Dict[str, float]) -> bool:
            return abs(m.get("mean_HCE", 0.0)) < 1e-9 and abs(m.get("mean_Q", 0.0)) < 1e-9 and abs(m.get("mean_X", 0.0)) < 1e-9

        def _build_rows(sec_list):
            built = []
            for idx, label, rel_start, rel_end in sec_list:
                abs_start = start_ts + rel_start
                abs_end = start_ts + rel_end
                buckets = storage.get_buckets_between(abs_start, abs_end, session_id=self._session_id)
                bucket_count = len(buckets)
                means = storage._compute_means_for_window(abs_start, abs_end, session_id=self._session_id)
                source = "exact buckets" if bucket_count > 0 else "no buckets"
                if bucket_count == 0 or _means_zero(means):
                    # session-level fallback
                    session_means = storage._compute_means_for_window(start_ts, end_ts, session_id=self._session_id)
                    if session_means and not _means_zero(session_means):
                        means = session_means
                        source = "session avg"
                    else:
                        # library avg fallback
                        lib_means = storage.get_track_library_avg(track_id)
                        if lib_means and not _means_zero(lib_means):
                            means = lib_means
                            source = "library avg"
                        else:
                            source = "no data"
                built.append(
                    {
                        "idx": idx,
                        "label": label,
                        "start_ts": abs_start,
                        "end_ts": abs_end,
                        "mean_HCE": means.get("mean_HCE", 0.0),
                        "mean_Q": means.get("mean_Q", 0.0),
                        "mean_X": means.get("mean_X", 0.0),
                        "source": source,
                        "bucket_count": bucket_count,
                    }
                )
            return built

        rows = _build_rows(sections)

        # If still zero or no buckets, retry with pseudo sections as a broader fallback
        if rows and all(_means_zero(r) for r in rows):
            total = max(duration, 1.0)
            n_parts = 5
            step = total / float(n_parts)
            pseudo_sections = [(i, f"Part {i+1} (est.)", i * step, (i + 1) * step) for i in range(n_parts)]
            rows = _build_rows(pseudo_sections)

        # Logging for diagnostics
        try:
            total_buckets = len(storage.get_buckets_between(start_ts, end_ts, session_id=self._session_id))
            mapped = sum(1 for r in rows if r.get("bucket_count", 0) > 0)
            print(
                f"[spotify] live mapping: track={track_id} dur={duration:.1f}s mapped_sections={mapped}/{len(rows)} total_buckets={total_buckets}",
                flush=True,
            )
        except Exception:
            pass

        storage.upsert_track_sections(track_session_id, track_id, title, artist, rows)

    def _compute_waveform(self, track_session_id: int, end_ts: float) -> None:
        """Build per-second HCE waveform from tagged buckets and store for reuse."""
        if not self._current_start_ts or not self._current_track_id:
            return
        start_ts = self._current_start_ts
        track_id = self._current_track_id
        duration = max(self._current_duration_sec or (end_ts - start_ts), end_ts - start_ts, 1.0)
        bin_sec = 1.0
        length = int(duration) + 1
        vals = [float("nan")] * length
        buckets = storage.get_buckets_between(start_ts - 5.0, end_ts + 5.0, session_id=self._session_id)
        for b in buckets:
            if b.get("track_uri") and b.get("track_uri") != track_id:
                continue
            rel = b.get("relative_seconds")
            if rel is None:
                rel = float(b.get("bucket_start_ts", 0.0) - start_ts)
            idx = int(rel // bin_sec)
            if 0 <= idx < length:
                vals[idx] = float(b.get("mean_HCE") or 0.0)
        # forward-fill then back-fill to smooth gaps
        last = 0.0
        for i in range(length):
            if vals[i] == vals[i]:  # not NaN
                last = vals[i]
            else:
                vals[i] = last
        for i in range(length - 1, -1, -1):
            if vals[i] == vals[i]:
                last = vals[i]
            else:
                vals[i] = last
        storage.insert_track_waveform(
            track_id=track_id,
            session_id=self._session_id,
            title=self._current_title or "",
            artist=self._current_artist or "",
            start_ts=start_ts,
            end_ts=end_ts,
            duration=duration,
            bin_sec=bin_sec,
            waveform=vals,
        )

    def _compute_waveform_points(self, track_session_id: int, end_ts: float) -> None:
        """Build per-second HCE/Q/X points from buckets and persist rows for deep analysis."""
        if not self._current_start_ts or not self._current_track_id:
            return
        start_ts = self._current_start_ts
        track_id = self._current_track_id
        title = self._current_title or ""
        artist = self._current_artist or ""
        duration = max(self._current_duration_sec or (end_ts - start_ts), end_ts - start_ts, 1.0)
        bin_sec = 1.0
        length = int(duration) + 1
        buckets = storage.get_buckets_between(start_ts - 5.0, end_ts + 5.0, session_id=self._session_id)
        h_vals = [float("nan")] * length
        q_vals = [float("nan")] * length
        x_vals = [float("nan")] * length
        r_vals = [float("nan")] * length
        for b in buckets:
            if b.get("track_uri") and b.get("track_uri") != track_id:
                continue
            rel = b.get("relative_seconds")
            if rel is None:
                rel = float(b.get("bucket_start_ts", 0.0) - start_ts)
            idx_start = int(rel // bin_sec)
            idx_end = int(((b.get("bucket_end_ts") or (start_ts + duration)) - start_ts) // bin_sec)
            idx_end = min(idx_end, length - 1)
            for idx in range(max(0, idx_start), idx_end + 1):
                h_vals[idx] = float(b.get("mean_HCE") or 0.0)
                q_vals[idx] = float(b.get("mean_Q") or 0.0)
                x_vals[idx] = float(b.get("mean_X") or 0.0)
                r_vals[idx] = float(b.get("valid_fraction") or 0.0)
        # simple forward/back fill to smooth gaps
        def _fill(vals, default=0.0):
            last = default
            for i in range(length):
                if vals[i] == vals[i]:
                    last = vals[i]
                else:
                    vals[i] = last
            last = default
            for i in range(length - 1, -1, -1):
                if vals[i] == vals[i]:
                    last = vals[i]
                else:
                    vals[i] = last
            return vals

        h_vals = _fill(h_vals, 0.0)
        q_vals = _fill(q_vals, 0.0)
        x_vals = _fill(x_vals, 0.0)
        r_vals = _fill(r_vals, 0.0)

        points = []
        for i in range(length):
            points.append(
                {
                    "rel_sec": float(i),
                    "hce": h_vals[i],
                    "q": q_vals[i],
                    "x": x_vals[i],
                    "reliability": r_vals[i],
                }
            )
        storage.insert_track_waveform_points(
            track_id=track_id,
            session_id=self._session_id,
            title=title,
            artist=artist,
            start_ts=start_ts,
            points=points,
        )


__all__ = ["SpotifyNowPlayingLogger"]

