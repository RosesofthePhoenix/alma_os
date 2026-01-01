"""Minimal live pipeline that attaches to LSL and emits placeholder state."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
from pylsl import StreamInlet

from alma import config
from alma.engine import lsl_client
from alma.engine.cortex_core import (
    build_state_packet,
    compute_step_if_ready,
    init_cortex_state,
    process_lsl_samples,
)
from alma.engine import storage


class StateEngine:
    """Attach to LSL, maintain rolling state, and optionally emit NDJSON."""

    def __init__(self, history_len: int = 600) -> None:
        config.ensure_required_paths()
        self._history: Deque[Dict[str, object]] = deque(maxlen=history_len)
        self._latest: Optional[Dict[str, object]] = None
        self._inlet: Optional[StreamInlet] = None
        self._inlet_info: Optional[object] = None
        self._stream_name: Optional[str] = None
        self._stream_type: Optional[str] = None
        self._channel_count: Optional[int] = None
        self._nominal_srate: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._stream_alive = False
        self._last_sample_ts_unix: Optional[float] = None
        self._emit_ndjson = False
        self._ndjson_path: Path = config.STATE_STREAM_PATH
        self._lock = threading.Lock()
        self._samples_received_total: int = 0
        self._samples_received_since_tick: int = 0
        self._samples_received_last_interval: int = 0
        self._history_ring = {
            "t": deque(maxlen=history_len),
            "ts_unix": deque(maxlen=history_len),
            "X": deque(maxlen=history_len),
            "Q_abs": deque(maxlen=history_len),
            "Q_vibe": deque(maxlen=history_len),
            "Q_vibe_focus": deque(maxlen=history_len),
            "Q_abs_raw": deque(maxlen=history_len),
            "Q_vibe_raw": deque(maxlen=history_len),
            "Q_vibe_focus_raw": deque(maxlen=history_len),
            "valid": deque(maxlen=history_len),
            "quality_conf": deque(maxlen=history_len),
            "reason_codes": deque(maxlen=history_len),
        }
        self._reliability_pct: Optional[float] = None
        profile = self._load_profile()
        baseline_path = profile.get("baseline_path", str(config.BASELINE_DEFAULT_PATH))
        self._cortex_state = init_cortex_state(baseline_path=baseline_path)
        self._ch_labels: List[str] = []
        self._session_id: Optional[str] = None
        self._profile_path = str(Path(__file__).resolve().parents[2] / "profiles" / "default.json")
        self._bucket_minutes = config.BUCKET_MINUTES_DEFAULT
        self._next_bucket_ts = time.time() + 60.0
        self._next_auto_event_ts = time.time() + 600.0
        self._auto_seen_peaks: set[tuple[str, float]] = set()
        self._auto_seen_streaks: set[tuple[str, float]] = set()

    # Public API
    def start(self) -> None:
        if self._running:
            return
        with self._lock:
            profile = self._load_profile()
            baseline_path = profile.get("baseline_path", str(config.BASELINE_DEFAULT_PATH))
            self._cortex_state = init_cortex_state(baseline_path=baseline_path)
        self._ensure_session_started()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        with self._lock:
            self._running = False
            self._stream_alive = False
            self._inlet = None
            self._inlet_info = None
            self._end_session_if_any()

    def set_emit_ndjson(self, enabled: bool) -> None:
        self._emit_ndjson = bool(enabled)

    def get_latest(self) -> Optional[Dict[str, object]]:
        with self._lock:
            return dict(self._latest) if self._latest else None

    def get_session_id(self) -> Optional[str]:
        with self._lock:
            return self._session_id

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            now = time.time()
            age = (now - self._last_sample_ts_unix) if self._last_sample_ts_unix else None
            return {
                "running": self._running,
                "stream_alive": self._stream_alive,
                "last_sample_age_s": age,
                "last_sample_ts_unix": self._last_sample_ts_unix,
                "samples_received_total": self._samples_received_total,
                "samples_received_last_interval": self._samples_received_last_interval,
                "inlet_info": self._summarize_inlet(self._inlet_info),
                "reliability_pct": self._reliability_pct,
                "latest_snapshot": dict(self._latest) if self._latest else None,
                "history_len": len(self._history_ring["valid"]),
                "emit_ndjson": self._emit_ndjson,
            }

    def get_history(self) -> Dict[str, List[object]]:
        """Return a copy of rolling history arrays for plotting."""
        with self._lock:
            return {k: list(v) for k, v in self._history_ring.items()}

    # Internal
    def _run(self) -> None:
        next_emit = time.time()
        while not self._stop_event.is_set():
            inlet = self._ensure_inlet()
            if inlet is not None:
                try:
                    samples, ts = inlet.pull_chunk(timeout=0.0)
                except Exception:
                    samples, ts = [], []
                if samples:
                    now_sample = time.time()
                    with self._lock:
                        self._last_sample_ts_unix = now_sample
                        self._stream_alive = True
                        self._samples_received_total += len(samples)
                        self._samples_received_since_tick += len(samples)
                    process_lsl_samples(
                        self._cortex_state,
                        samples=samples,
                        lsl_timestamps=ts,
                        fs=self._nominal_srate or 0,
                        ch_names=self._ch_labels or None,
                    )
                self._update_alive_flag()
            else:
                self._update_alive_flag()

            now = time.time()
            if now >= next_emit:
                next_emit = now + 1.0
                with self._lock:
                    self._samples_received_last_interval = self._samples_received_since_tick
                    self._samples_received_since_tick = 0
                snapshot = compute_step_if_ready(self._cortex_state, now_s=now)
                if snapshot:
                    with self._lock:
                        self._latest = snapshot
                        self._append_history(snapshot)
                        self._reliability_pct = self._compute_reliability_pct()
                    if self._emit_ndjson:
                        self._append_ndjson(snapshot)
            if now >= self._next_bucket_ts:
                self._next_bucket_ts = now + 60.0
                self._maybe_bucket(now)
            if now >= self._next_auto_event_ts:
                self._next_auto_event_ts = now + 600.0
                self._maybe_auto_events(now)

        # Cleanup
        try:
            if self._inlet:
                self._inlet.close_stream()
        except Exception:
            pass

    def _ensure_inlet(self) -> Optional[StreamInlet]:
        if self._inlet:
            return self._inlet
        preferred_name = self._get_preferred_stream_name()
        inlet, info = lsl_client.resolve_preferred_stream(preferred_name=preferred_name, timeout=2.0)
        if inlet:
            with self._lock:
                self._inlet = inlet
                self._inlet_info = info
                self._stream_name = info.name() if hasattr(info, "name") else None
                self._stream_type = info.type() if hasattr(info, "type") else None
                self._channel_count = info.channel_count() if hasattr(info, "channel_count") else None
                self._nominal_srate = info.nominal_srate() if hasattr(info, "nominal_srate") else None
                self._ch_labels = self._get_channel_labels(info)
        return inlet

    def _update_alive_flag(self) -> None:
        with self._lock:
            if self._last_sample_ts_unix is None:
                self._stream_alive = False
                return
            age = time.time() - self._last_sample_ts_unix
            self._stream_alive = age <= 2.5

    @staticmethod
    def _get_preferred_stream_name() -> Optional[str]:
        try:
            profile_path = Path(__file__).resolve().parents[2] / "profiles" / "default.json"
            with profile_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            preferred = data.get("preferred_lsl_stream_name")
            if preferred:
                return str(preferred)
        except Exception:
            return None
        return None

    def _build_snapshot(self, ts_unix: float) -> Dict[str, object]:
        # Deprecated placeholder; retained for backward compatibility (not used).
        return {"ts_unix": ts_unix, "stream_alive": self._stream_alive}

    def _append_ndjson(self, snapshot: Dict[str, object]) -> None:
        try:
            packet = build_state_packet(snapshot)
            self._ndjson_path.parent.mkdir(parents=True, exist_ok=True)
            with self._ndjson_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(packet) + "\n")
        except Exception:
            # Swallow errors; leave to future instrumentation.
            pass

    def _append_history(self, snapshot: Dict[str, object]) -> None:
        """Append snapshot fields to rolling history for UI."""
        self._history_ring["t"].append(snapshot.get("t_session"))
        self._history_ring["ts_unix"].append(snapshot.get("ts_unix"))
        self._history_ring["X"].append(snapshot.get("X"))
        self._history_ring["Q_abs"].append(snapshot.get("Q_abs"))
        self._history_ring["Q_vibe"].append(snapshot.get("Q_vibe"))
        self._history_ring["Q_vibe_focus"].append(snapshot.get("Q_vibe_focus"))
        self._history_ring["Q_abs_raw"].append(snapshot.get("Q_abs_raw"))
        self._history_ring["Q_vibe_raw"].append(snapshot.get("Q_vibe_raw"))
        self._history_ring["Q_vibe_focus_raw"].append(snapshot.get("Q_vibe_focus_raw"))
        reliability = snapshot.get("reliability") or {}
        self._history_ring["valid"].append(reliability.get("valid"))
        self._history_ring["quality_conf"].append(reliability.get("quality_conf"))
        self._history_ring["reason_codes"].append(reliability.get("reason_codes"))

    def _compute_reliability_pct(self) -> Optional[float]:
        vals = [bool(v) for v in self._history_ring["valid"] if v is not None]
        if not vals:
            return None
        return 100.0 * sum(vals) / float(len(vals))

    def _maybe_bucket(self, now: float) -> None:
        if not self._session_id:
            return
        window_s = self._bucket_minutes * 60.0
        with self._lock:
            t = list(self._history_ring["t"])
            if not t:
                return
            t_max = t[-1]
            t_start = t_max - window_s
            idx = [i for i, tv in enumerate(t) if tv >= t_start]
            if not idx:
                return

            def take(key):
                arr = list(self._history_ring[key])
                return [arr[i] for i in idx if i < len(arr)]

            q_series = take("Q_vibe_focus")
            x_series = take("X")
            valid_series = take("valid")
        if not q_series or not x_series or not valid_series:
            return
        q_arr = np.array([v for v in q_series if v is not None], dtype=float)
        x_arr = np.array([v for v in x_series if v is not None], dtype=float)
        v_arr = np.array([1.0 if bool(v) else 0.0 for v in valid_series], dtype=float)
        if q_arr.size == 0 or x_arr.size == 0 or v_arr.size == 0:
            return
        mean_X = float(np.nanmean(x_arr))
        mean_Q = float(np.nanmean(q_arr))
        std_Q = float(np.nanstd(q_arr))
        valid_fraction = float(np.nanmean(v_arr))

        t_slice = np.array([t[i] for i in idx], dtype=float)
        try:
            if t_slice.size >= 2 and not np.allclose(t_slice, t_slice[0]):
                slope, _ = np.polyfit(t_slice, q_arr, 1)
                Q_slope = float(slope)
            else:
                Q_slope = 0.0
        except Exception:
            Q_slope = 0.0

        label = self._label_bucket(mean_X, std_Q, Q_slope, valid_fraction)

        try:
            storage.upsert_bucket(
                bucket_start_ts=float(t_slice[0]),
                bucket_end_ts=float(t_slice[-1]),
                session_id=self._session_id,
                mean_X=mean_X,
                mean_Q=mean_Q,
                std_Q=std_Q,
                Q_slope=Q_slope,
                valid_fraction=valid_fraction,
                label=label,
            )
        except Exception:
            pass

    @staticmethod
    def _label_bucket(mean_X: float, std_Q: float, Q_slope: float, valid_fraction: float) -> str:
        if valid_fraction < config.MIN_VALID_FRACTION_FOR_LABEL:
            return "INSUFFICIENT_SIGNAL"
        if mean_X >= 0.65 and std_Q <= 0.12:
            return "DEEP_WORK"
        if std_Q >= 0.20 and Q_slope > 0:
            return "IDEATION"
        if mean_X <= 0.45 and std_Q <= 0.16:
            return "RECOVERY"
        return "ENGAGEMENT"

    def _maybe_auto_events(self, now: float) -> None:
        if not self._session_id:
            return
        ts0 = now - 6 * 3600
        ts1 = now
        try:
            buckets = storage.get_buckets_between(ts0, ts1, session_id=self._session_id)
        except Exception:
            return
        if not buckets:
            return
        # Peaks
        top = sorted(buckets, key=lambda b: b.get("mean_Q") or 0.0, reverse=True)[:5]
        for b in top:
            bstart = b.get("bucket_start_ts")
            if bstart is None:
                continue
            key = (self._session_id, float(bstart))
            if key in self._auto_seen_peaks:
                continue
            self._auto_seen_peaks.add(key)
            mid_ts = (float(b.get("bucket_start_ts", 0)) + float(b.get("bucket_end_ts", 0))) / 2.0
            note = f"Top Q bucket (mean_Q={b.get('mean_Q', 0):.3f} label={b.get('label', '')})"
            try:
                storage.insert_event(
                    ts=mid_ts,
                    session_id=self._session_id,
                    kind="auto_peak",
                    label="Q Peak",
                    note=note,
                    tags_json={"auto": True, "kind": "auto_peak"},
                    context_json={"bucket": b},
                )
            except Exception:
                pass

        # Streaks (DEEP_WORK + std_Q <= 0.12 + mean_X >= 0.65)
        streak: List[Dict[str, object]] = []
        def end_streak():
            if not streak:
                return
            duration = float(streak[-1].get("bucket_end_ts", 0) - streak[0].get("bucket_start_ts", 0))
            if duration < 20 * 60:
                return
            start_ts = float(streak[0].get("bucket_start_ts", 0))
            key = (self._session_id, start_ts)
            if key in self._auto_seen_streaks:
                return
            self._auto_seen_streaks.add(key)
            mean_q = float(np.nanmean([b.get("mean_Q", 0.0) for b in streak]))
            mean_x = float(np.nanmean([b.get("mean_X", 0.0) for b in streak]))
            note = f"Stable focus streak {duration/60:.1f} min, mean_Q={mean_q:.3f}, mean_X={mean_x:.3f}"
            try:
                storage.insert_event(
                    ts=start_ts + duration / 2.0,
                    session_id=self._session_id,
                    kind="auto_streak",
                    label="Stable Focus Streak",
                    note=note,
                    tags_json={"auto": True, "kind": "auto_streak"},
                    context_json={
                        "streak_start": streak[0],
                        "streak_end": streak[-1],
                        "duration_s": duration,
                        "mean_Q": mean_q,
                        "mean_X": mean_x,
                    },
                )
            except Exception:
                pass

        for b in buckets:
            if (
                b.get("label") == "DEEP_WORK"
                and (b.get("std_Q") or 1) <= 0.12
                and (b.get("mean_X") or 0) >= 0.65
            ):
                streak.append(b)
            else:
                end_streak()
                streak = []
        end_streak()

    @staticmethod
    def _summarize_inlet(info: Optional[object]) -> Optional[Dict[str, object]]:
        if info is None:
            return None
        try:
            return {
                "name": info.name() if hasattr(info, "name") else None,
                "type": info.type() if hasattr(info, "type") else None,
                "source_id": info.source_id() if hasattr(info, "source_id") else None,
                "uid": info.uid() if hasattr(info, "uid") else None,
                "channel_count": info.channel_count() if hasattr(info, "channel_count") else None,
                "nominal_srate": info.nominal_srate() if hasattr(info, "nominal_srate") else None,
            }
        except Exception:
            return None

    @staticmethod
    def _get_channel_labels(info) -> List[str]:
        labels: List[str] = []
        try:
            ch = info.desc().child("channels").child("channel")
            while ch.name() == "channel":
                lab = ch.child_value("label")
                labels.append(lab)
                ch = ch.next_sibling()
        except Exception:
            pass
        return labels

    def _ensure_session_started(self) -> None:
        if self._session_id:
            return
        profile = self._load_profile()
        muse_address = profile.get("muse_address", "")
        ndjson_path = profile.get("ndjson_state_path", str(config.STATE_STREAM_PATH))
        lsl_name = self._stream_name or None
        try:
            self._session_id = storage.start_session(
                profile_path=self._profile_path,
                muse_address=muse_address,
                lsl_stream_name=lsl_name,
                ndjson_path=ndjson_path,
            )
        except Exception:
            self._session_id = None

    def _end_session_if_any(self) -> None:
        if not self._session_id:
            return
        try:
            storage.end_session(self._session_id)
        except Exception:
            pass
        self._session_id = None

    @staticmethod
    def _load_profile() -> dict:
        try:
            with Path(__file__).resolve().parents[2].joinpath("profiles", "default.json").open(
                "r", encoding="utf-8"
            ) as f:
                return json.load(f)
        except Exception:
            return {}


__all__ = ["StateEngine"]

