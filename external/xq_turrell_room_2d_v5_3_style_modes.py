#!/usr/bin/env python3
"""xq_turrell_room_2d_v5.py (v5)

Turrell-ish 2D visuals that react to Cortex Console `state_stream.ndjson` (X/Q + reliability).

v5 upgrades vs v4:
- Supports multiple Q drivers (abs, focus, perX, vibe, vibe_focus, vibe_focus_e, meaningful, focus_meaningful).
- Integrates Cortex Console v4.1 fields: `state.raw.*`, `reliability.qualia_valid`, and `artifact_quality`.
- New hotkeys: V cycles Q source (auto/masked/raw), M cycles Q metric.
- LIVE/STALE dot (stream heartbeat) so you can instantly see if Muse packets are arriving.
- Higher-quality defaults ("4K-like"): higher internal render scale + adjustable bloom.

Controls:
  0=OFF  1=MAX_FOCUS  2=MAX_Q  3=STABILIZE
  F=toggle fullscreen
  H=toggle HUD
  D=toggle demo (simulated X/Q)
  V=cycle Q source (auto → masked → raw)
  M=cycle Q metric (abs → focus → vibe → vibe_focus → vibe_focus_e → perX → meaningful → focus_meaningful)
  C=cycle validity gate source (stream → qc → qualia → auto)
  Esc=quit

Tip for TV / extended display:
  Run with --display 1 --fullscreen (or drag window to TV, then press F).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pygame
try:
    import pygame.freetype as ft
except Exception:
    ft = None

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False


# ----------------------------- small utils -----------------------------

def clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else x


def clampf(x: float, lo: float, hi: float) -> float:
    try:
        xf = float(x)
    except Exception:
        return lo
    if xf < lo:
        return lo
    if xf > hi:
        return hi
    return xf


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(a: float, b: float, x):
    # x may be float or numpy array
    if HAVE_NUMPY and isinstance(x, np.ndarray):
        t = np.clip((x - a) / (b - a), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    else:
        t = clamp01((x - a) / (b - a))
        return t * t * (3.0 - 2.0 * t)


def ema(prev: float, x: float, alpha: float) -> float:
    return (1.0 - alpha) * prev + alpha * x


def alpha_from_dt(dt_s: float, alpha_per_sec: float) -> float:
    # exact discretization: alpha = 1 - exp(-k*dt)
    if dt_s <= 0:
        return 0.0
    k = max(0.0, alpha_per_sec)
    return 1.0 - math.exp(-k * dt_s)


def parse_res(s: str) -> Tuple[int, int]:
    s = s.lower().strip().replace(" ", "")
    if "x" not in s:
        raise ValueError("Resolution must look like 1280x720")
    a, b = s.split("x", 1)
    return int(a), int(b)


def pick_display_size(display_idx: int) -> Tuple[int, int]:
    """
    Best-effort fullscreen size for a given display.
    Avoids (0,0) SCALED fullscreen issues on some SDL builds.
    """
    try:
        modes = pygame.display.get_desktop_sizes()
        if isinstance(modes, (list, tuple)) and len(modes) > display_idx:
            w, h = modes[display_idx]
            if w > 0 and h > 0:
                return int(w), int(h)
    except Exception:
        pass
    try:
        info = pygame.display.get_desktop_display_mode(display_idx)
        w, h = getattr(info, "w", 0), getattr(info, "h", 0)
        if w and h:
            return int(w), int(h)
    except Exception:
        pass
    return (0, 0)


# ----------------------------- palettes -----------------------------

TURRELL_5_PALETTE = [
    "#330E21",  # ink (depth)
    "#3A08D2",  # indigo
    "#6D00E4",  # violet
    "#A632FD",  # electric purple
    "#F22BCB",  # fuchsia
    "#FC67C0",  # neon pink
    "#FD540A",  # blaze orange
    "#FF8E24",  # amber
    "#C4B1DB",  # lavender-white bloom
]


# ----------------------------- NDJSON follower -----------------------------

@dataclass
class State:
    # last accepted ("good") raw values (used for visuals)
    x_raw: float = 0.0
    q_raw: float = 0.0
    q_origin: str = "-"  # masked|raw|-
    hce: float = 0.0
    hce_ema: float = 0.0

    # normalized + smoothed values (0..1)
    x: float = 0.0
    q: float = 0.0

    # "visual-valid" (RELAXED + grace) — used for HUD + render gating
    valid: bool = False
    valid_ema: float = 0.0

    # incoming stream validity + quality (for debugging / optional gating)
    stream_valid: bool = False
    stream_valid_ema: float = 0.0

    qc: float = 0.0
    qc_ema: float = 0.0

    # v4.1 qualia/artifact signals (if present)
    artifact_quality: float = 0.0
    artifact_quality_ema: float = 0.0
    qualia_valid: bool = False
    qualia_valid_ema: float = 0.0
    c_pe_n: float = 0.5
    s_flat_n: float = 0.5
    option_e_present: bool = False
    contact_conf: float = 0.0
    contact_conf_ema: float = 0.0

    # reason codes from stream (if present)
    reasons: tuple[str, ...] = ()

    # how long the stream has been continuously invalid (seconds)
    invalid_age: float = 0.0


class NDJSONFollower:
    """Tail-follow a growing NDJSON file and keep a smoothed state."""

    def __init__(
        self,
        path: str,
        x_min: float,
        x_max: float,
        q_min: float,
        q_max: float,
        alpha_x_per_sec: float = 2.0,
        alpha_q_per_sec: float = 2.0,
        alpha_valid_per_sec: float = 3.0,
        alpha_qc_per_sec: float = 3.0,
        # Visual validity behavior (product-grade feel)
        quality_valid_thr: float = 0.25,
        valid_grace_sec: float = 1.50,
        valid_source: str = "auto",  # auto|stream|qc
        freeze_on_invalid: bool = True,
        # Q selection behavior (debug/testing)
        q_source: str = "auto",  # auto|masked|raw
        q_metric: str = "auto",  # auto|abs|focus
        raw_when_invalid: bool = False,
        qualia_valid_artifact_thr: float = 0.55,
    ):
        self.path = path
        self._fh = None
        self._pos = 0

        self.state = State()

        # ranges for normalization
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.q_min = float(q_min)
        self.q_max = float(q_max)

        # smoothing rates
        self.alpha_x_per_sec = float(alpha_x_per_sec)
        self.alpha_q_per_sec = float(alpha_q_per_sec)
        self.alpha_valid_per_sec = float(alpha_valid_per_sec)
        self.alpha_qc_per_sec = float(alpha_qc_per_sec)

        self.last_pkt_walltime: Optional[float] = None

        # visual validity configuration
        self.quality_valid_thr = float(quality_valid_thr)
        self.valid_grace_sec = float(valid_grace_sec)
        self.valid_source = str(valid_source).strip().lower()
        self.freeze_on_invalid = bool(freeze_on_invalid)

        self.q_source = str(q_source).strip().lower()
        if self.q_source not in {"auto", "masked", "raw"}:
            self.q_source = "auto"
        self.q_metric = str(q_metric).strip().lower()
        if self.q_metric not in {"auto", "abs", "focus", "perx", "vibe", "vibe_focus", "vibe_focus_e", "meaningful", "focus_meaningful"}:
            self.q_metric = "auto"

        # allow raw Q ingestion during invalid periods (debug)
        self.raw_when_invalid = bool(raw_when_invalid)
        
        self.qualia_valid_artifact_thr = float(qualia_valid_artifact_thr)

        self._last_stream_valid_true_time: Optional[float] = None

    def _open_if_needed(self) -> None:
        if self._fh is not None:
            return
        if not os.path.exists(self.path):
            return
        self._fh = open(self.path, "r", encoding="utf-8", errors="ignore")
        self._fh.seek(0, os.SEEK_END)
        self._pos = self._fh.tell()

    def poll_latest(self, max_lines: int = 200) -> Optional[Dict[str, Any]]:
        """Read any newly appended lines and return the last parsed JSON object."""
        self._open_if_needed()
        if self._fh is None:
            return None

        try:
            # handle truncation/rotation
            cur_size = os.path.getsize(self.path)
            if cur_size < self._pos:
                self._fh.close()
                self._fh = open(self.path, "r", encoding="utf-8", errors="ignore")
                self._pos = 0

            self._fh.seek(self._pos)
            last_obj = None
            lines = 0
            while lines < max_lines:
                line = self._fh.readline()
                if not line:
                    break
                self._pos = self._fh.tell()
                line = line.strip()
                if not line:
                    continue
                try:
                    last_obj = json.loads(line)
                except Exception:
                    continue
                lines += 1
            return last_obj
        except Exception:
            return None

    @staticmethod
    def _first_present(d: Dict[str, Any], keys) -> Optional[float]:
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return float(d[k])
                except Exception:
                    continue
        return None

    def _extract(self, pkt: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[bool], Tuple[str, ...], str, Optional[bool], Optional[float], Optional[float], Optional[float], Optional[float]]:


        """Extract X/Q and validity signals from an NDJSON packet.

        Cortex Console emits a packet like:
          { state: {...masked metrics...}, reliability: { valid, quality_conf, reason_codes, ... } }

        This v4 visualizer:
        - Prefers `reliability.valid` + `reliability.quality_conf` when present.
        - Supports selecting Q from masked fields, raw/unmasked fields, or auto (masked→raw).
        - Optionally allows raw-Q ingestion even when invalid (debug/testing only).
        """
        # state payload can be nested under "state" or "metrics" etc.
        obj: Dict[str, Any] = pkt if isinstance(pkt, dict) else {}
        for candidate in ("state", "metrics", "data"):
            if isinstance(obj.get(candidate), dict):
                obj = obj[candidate]
                break

        rel = pkt.get("reliability") if isinstance(pkt, dict) and isinstance(pkt.get("reliability"), dict) else None

        # ----- validity / quality -----
        qc = None
        stream_valid = None
        reasons: Tuple[str, ...] = ()

        if rel is not None:
            qc = rel.get("quality_conf", rel.get("qc", None))
            v = rel.get("valid", None)
            if v is not None:
                stream_valid = bool(v)
            rc = rel.get("reason_codes", None)
            if isinstance(rc, list):
                reasons = tuple(str(x) for x in rc)
            elif isinstance(rc, str):
                reasons = tuple(r for r in rc.split(";") if r.strip())

        if qc is None:
            qc = self._first_present(
                obj,
                [
                    "quality_conf",
                    "qc",
                    "quality",
                    "quality_ema",
                    "quality_conf_ema",
                    "qconf",
                ],
            )

        if stream_valid is None:
            v = obj.get("valid", obj.get("is_valid", obj.get("ok", None)))
            if v is not None:
                stream_valid = bool(v)

        # ----- metrics -----
        x = self._first_present(
            obj,
            [
                "X",
                "x",
                "x_focus",
                "X_focus",
                "X_focus_ema",
                "x_focus_ema",
                "X_ema",
                "x_ema",
                "focus",
                "Focus",
                "X_abs",
                "X_abs_ema",
            ],
        )

        # Masked Q (preferred: null when invalid in Cortex Console stream)
        qm = (self.q_metric or "auto").strip().lower()

        # Supported q_metric values:
        #   auto | abs | focus | perx | vibe | vibe_focus | vibe_focus_e | meaningful | focus_meaningful
        if qm in ("abs", "q_abs"):
            masked_keys = ["Q_abs", "q_abs"]
            raw_keys = ["Q_abs_raw", "q_abs_raw", "Q_abs", "q_abs"]
        elif qm in ("focus", "q_focus"):
            masked_keys = ["Q_focus", "q_focus"]
            raw_keys = ["Q_focus_raw", "q_focus_raw", "Q_focus", "q_focus"]
        elif qm in ("perx", "q_perx", "qperx"):
            # Note: in v4.2.1 math, Q_perX ~= R_abs (redundant). We still support it as an explicit signal.
            masked_keys = ["Q_perX", "Q_perx", "q_perx", "q_perX", "R_abs", "r_abs"]
            raw_keys = ["Q_perX_raw", "Q_perx_raw", "q_perx_raw", "q_perX_raw", "Q_perX", "Q_perx", "q_perx", "q_perX", "R_abs_raw", "r_abs_raw", "R_abs", "r_abs"]
        elif qm in ("vibe", "q_vibe"):
            masked_keys = ["Q_vibe", "q_vibe"]
            raw_keys = ["Q_vibe_raw", "q_vibe_raw", "Q_vibe", "q_vibe"]
        elif qm in ("vibe_focus", "q_vibe_focus", "vibefocus"):
            masked_keys = ["Q_vibe_focus", "q_vibe_focus"]
            raw_keys = ["Q_vibe_focus_raw", "q_vibe_focus_raw", "Q_vibe_focus", "q_vibe_focus"]
        elif qm in ("vibe_focus_e", "q_vibe_focus_e", "vibefocus_e"):
            masked_keys = ["Q_vibe_focus_E", "q_vibe_focus_E"]
            raw_keys = ["Q_vibe_focus_E", "q_vibe_focus_E", "Q_vibe_focus_E_raw", "q_vibe_focus_E_raw"]
        elif qm in ("meaningful", "r_meaningful"):
            masked_keys = ["R_meaningful", "r_meaningful"]
            raw_keys = ["R_meaningful_raw", "r_meaningful_raw", "R_meaningful", "r_meaningful"]
        elif qm in ("focus_meaningful", "r_focus_meaningful"):
            masked_keys = ["R_focus_meaningful", "r_focus_meaningful"]
            raw_keys = ["R_focus_meaningful_raw", "r_focus_meaningful_raw", "R_focus_meaningful", "r_focus_meaningful"]
        else:
            # auto: abs-first, then focus, then perX/r_abs, then fallback
            masked_keys = [
                "Q_abs", "q_abs",
                "Q_focus", "q_focus",
                "Q_perX", "Q_perx", "q_perx", "q_perX",
                "R_abs", "r_abs",
                "Q_total", "q_total",
                "Q_abs_ema", "q_abs_ema",
                "Q_focus_ema", "q_focus_ema",
                "R_abs_ema", "r_abs_ema",
                "Q_perX_ema", "q_perx_ema",
            ]
            raw_keys = [
                "Q_abs_raw", "q_abs_raw",
                "Q_focus_raw", "q_focus_raw",
                "Q_perX_raw", "Q_perx_raw", "q_perx_raw", "q_perX_raw",
                "R_abs_raw", "r_abs_raw",
                "Q_abs", "q_abs",
                "Q_focus", "q_focus",
                "Q_perX", "Q_perx", "q_perx", "q_perX",
                "R_abs", "r_abs",
                "Q", "q",
            ]

        q_masked = self._first_present(obj, masked_keys)

        # Raw/unmasked Q: expected under obj["raw"] in Cortex Console streams.
        q_raw_candidate = None
        raw_obj = obj.get("raw") if isinstance(obj.get("raw"), dict) else None
        if isinstance(raw_obj, dict):
            q_raw_candidate = self._first_present(raw_obj, raw_keys)
        if q_raw_candidate is None:
            # allow exporters that put *_raw at top-level
            q_raw_candidate = self._first_present(obj, raw_keys)

        q_origin = "-"
        q_src = (self.q_source or "auto").strip().lower()
        if q_src == "masked":
            q = q_masked
            if q is not None:
                q_origin = "masked"
        elif q_src == "raw":
            q = q_raw_candidate
            if q is not None:
                q_origin = "raw"
        else:
            # auto: prefer masked; fall back to raw if masked missing
            if q_masked is not None:
                q = q_masked
                q_origin = "masked"
            elif q_raw_candidate is not None:
                q = q_raw_candidate
                q_origin = "raw"
            else:
                q = None
                q_origin = "-"

        # HCE (transcendence)
        hce_val = obj.get("HCE", obj.get("hce"))

        # Some pipelines log Q as percent (0-100).
        if q is not None:
            try:
                qf = float(q)
                if qf > 5.0:
                    qf = qf / 100.0
                q = qf
            except Exception:
                q = None

        if x is not None:
            try:
                x = float(x)
            except Exception:
                x = None

        if qc is not None:
            try:
                qc = float(qc)
            except Exception:
                qc = None

        # v4.1 qualia/artifact fields (best-effort)
        artifact_quality = self._first_present(
            obj,
            [
                "artifact_quality",
                "artifactQuality",
                "aq",
                "aq_raw",
                "phi_meaningful",  # sometimes used as a proxy
            ],
        )
        if artifact_quality is None and rel is not None:
            try:
                aqv = rel.get("artifact_quality", None)
                if aqv is not None:
                    artifact_quality = float(aqv)
            except Exception:
                pass

        contact_conf = self._first_present(
            obj,
            [
                "contact_conf",
                "contactConfidence",
                "contact",
                "contact_conf_raw",
            ],
        )
        if contact_conf is None and rel is not None:
            try:
                ccv = rel.get("contact_conf", None)
                if ccv is not None:
                    contact_conf = float(ccv)
            except Exception:
                pass

        raw_obj = obj.get("raw") if isinstance(obj.get("raw"), dict) else {}
        c_pe_n = None
        s_flat_n = None
        try:
            c_pe_n = float(raw_obj.get("C_pe_n")) if raw_obj.get("C_pe_n") is not None else None
        except Exception:
            c_pe_n = None
        try:
            s_flat_n = float(raw_obj.get("S_aperiodic_slope_n")) if raw_obj.get("S_aperiodic_slope_n") is not None else None
        except Exception:
            s_flat_n = None

        qualia_valid = None
        if rel is not None and ("qualia_valid" in rel):
            try:
                qualia_valid = bool(rel.get("qualia_valid"))
            except Exception:
                qualia_valid = None
        if qualia_valid is None and ("qualia_valid" in obj):
            try:
                qualia_valid = bool(obj.get("qualia_valid"))
            except Exception:
                qualia_valid = None
        if qualia_valid is None and artifact_quality is not None:
            try:
                qualia_valid = bool(float(artifact_quality) >= float(self.qualia_valid_artifact_thr))
            except Exception:
                qualia_valid = None

        return x, q, qc, stream_valid, reasons, q_origin, qualia_valid, artifact_quality, contact_conf, c_pe_n, s_flat_n

    def _normalize_x(self, x_raw: float) -> float:
        if self.x_max <= self.x_min:
            return 0.0
        x_n = (x_raw - self.x_min) / (self.x_max - self.x_min)
        x_n = clamp01(x_n)
        # perceptual curve: expand small changes
        return clamp01(x_n ** 0.6)

    def _normalize_q(self, q_raw: float) -> float:
        if self.q_max <= self.q_min:
            return 0.0
        q_n = (q_raw - self.q_min) / (self.q_max - self.q_min)
        q_n = clamp01(q_n)
        return clamp01(q_n ** 0.7)

    def update_from_packet(self, pkt: Dict[str, Any]) -> None:
        now = time.time()
        if self.last_pkt_walltime is None:
            dt = 0.0
        else:
            dt = now - self.last_pkt_walltime
        self.last_pkt_walltime = now

        # clamp dt so if you paused the app, you don't instantly jump
        dt = max(0.0, min(dt, 0.5))

        x_raw, q_raw, qc, stream_valid_in, reasons, q_origin, qualia_valid_in, artifact_quality_in, contact_conf_in, c_pe_n_in, s_flat_n_in = self._extract(pkt)
        self.state.q_origin = q_origin or "-"

        # Keep prior qc if missing
        if qc is None:
            qc = self.state.qc

        # Update v4.1 qualia/artifact fields (best-effort; independent of base_valid)
        if artifact_quality_in is None:
            artifact_quality_in = self.state.artifact_quality
        try:
            self.state.artifact_quality = float(artifact_quality_in)
        except Exception:
            pass

        if contact_conf_in is None:
            contact_conf_in = self.state.contact_conf
        try:
            self.state.contact_conf = float(contact_conf_in)
        except Exception:
            pass

        drivers_present = False
        try:
            if c_pe_n_in is not None and np.isfinite(c_pe_n_in):
                self.state.c_pe_n = float(c_pe_n_in)
                drivers_present = True
            else:
                self.state.c_pe_n = 0.5
        except Exception:
            self.state.c_pe_n = 0.5
        try:
            if s_flat_n_in is not None and np.isfinite(s_flat_n_in):
                self.state.s_flat_n = float(s_flat_n_in)
                drivers_present = True
            else:
                self.state.s_flat_n = 0.5
        except Exception:
            self.state.s_flat_n = 0.5
        self.state.option_e_present = bool(drivers_present)

        # Determine "base valid" (what we trust for accepting new metrics)
        base_valid: Optional[bool] = None
        src = (self.valid_source or "auto").strip().lower()

        if src in ("auto", "stream"):
            if stream_valid_in is not None:
                base_valid = bool(stream_valid_in)

        if base_valid is None and src in ("auto", "qc"):
            if qc is not None:
                base_valid = bool(float(qc) >= self.quality_valid_thr)

        if base_valid is None and src in ("auto", "qualia"):
            if qualia_valid_in is not None:
                base_valid = bool(qualia_valid_in)
            elif self.state.qualia_valid is not None:
                base_valid = bool(self.state.qualia_valid)

        if base_valid is None:
            base_valid = bool(self.state.stream_valid)

        # Visual-valid (RELAXED + grace): hide short artifact bursts in the HUD + gating,
        # while still NOT ingesting bad Q/X.
        if base_valid:
            self._last_stream_valid_true_time = now
            visual_valid = True
        else:
            if self._last_stream_valid_true_time is None:
                visual_valid = False
            else:
                visual_valid = (now - self._last_stream_valid_true_time) <= self.valid_grace_sec

        # Track continuous invalid duration (based on base_valid)
        if base_valid:
            self.state.invalid_age = 0.0
        else:
            self.state.invalid_age = float(self.state.invalid_age + dt)

        # Compute alphas based on packet dt (NOT frame dt!)
        ax = alpha_from_dt(dt, self.alpha_x_per_sec)
        aq = alpha_from_dt(dt, self.alpha_q_per_sec)
        av = alpha_from_dt(dt, self.alpha_valid_per_sec)
        aqc = alpha_from_dt(dt, self.alpha_qc_per_sec)

        # Always update quality (used for gating/dimming)
        try:
            self.state.qc = float(qc)
        except Exception:
            self.state.qc = float(self.state.qc)
        self.state.qc_ema = ema(self.state.qc_ema, self.state.qc, aqc if dt > 0 else 1.0)

        # Smooth artifact/contact too (helps gentle dimming / avoids HUD jitter)
        self.state.artifact_quality_ema = ema(self.state.artifact_quality_ema, self.state.artifact_quality, aqc if dt > 0 else 1.0)
        self.state.contact_conf_ema = ema(self.state.contact_conf_ema, self.state.contact_conf, aqc if dt > 0 else 1.0)

        # Debug: incoming validity + reasons
        self.state.stream_valid = bool(stream_valid_in) if stream_valid_in is not None else bool(base_valid)
        self.state.stream_valid_ema = ema(self.state.stream_valid_ema, 1.0 if base_valid else 0.0, av if dt > 0 else 1.0)
        self.state.reasons = reasons if reasons is not None else ()

        # Qualia-valid is independent of stream valid (v4.1)
        if qualia_valid_in is None:
            qualia_valid_in = bool(self.state.qualia_valid)
        self.state.qualia_valid = bool(qualia_valid_in)
        self.state.qualia_valid_ema = ema(self.state.qualia_valid_ema, 1.0 if self.state.qualia_valid else 0.0, av if dt > 0 else 1.0)

        # Visual-valid (what HUD shows as "valid=")
        self.state.valid = bool(visual_valid)
        self.state.valid_ema = ema(self.state.valid_ema, 1.0 if self.state.valid else 0.0, av if dt > 0 else 1.0)

        # Accept new X/Q only when base_valid (or if freeze is disabled).
        # For qualia-style metrics (vibe / vibe_focus / meaningful), prefer qualia_valid.
        accept_x = bool(base_valid) or (not self.freeze_on_invalid)

        qm = (self.q_metric or "auto").strip().lower()
        qualia_metric = qm in {"vibe", "vibe_focus", "vibe_focus_e", "meaningful", "focus_meaningful"}
        q_gate_ok = bool(self.state.qualia_valid) if qualia_metric else bool(base_valid)
        accept_q = bool(q_gate_ok) or (not self.freeze_on_invalid)

        # Optionally allow RAW-Q ingestion even during invalid periods (debug/testing only).
        if (not accept_q) and self.raw_when_invalid and (self.state.q_origin == "raw") and (q_raw is not None):
            accept_q = True

        if accept_x:
            if x_raw is None:
                x_raw = self.state.x_raw
            try:
                self.state.x_raw = float(x_raw)
            except Exception:
                pass

            x_n = self._normalize_x(self.state.x_raw)
            self.state.x = ema(self.state.x, x_n, ax if dt > 0 else 1.0)

        if accept_q:
            if q_raw is None:
                q_raw = self.state.q_raw
            try:
                self.state.q_raw = float(q_raw)
            except Exception:
                pass

            q_n = self._normalize_q(self.state.q_raw)
            self.state.q = ema(self.state.q, q_n, aq if dt > 0 else 1.0)
        # else: freeze (hold last good X and/or Q)

        # HCE smoothing (scaled values coming from Cortex; often 0..~50)
        try:
            hce_in = locals().get("hce_val")
            if hce_in is None:
                hce_in = self.state.hce
            h_val = float(hce_in)
            h_val = max(0.0, h_val)
            self.state.hce = h_val
            self.state.hce_ema = ema(self.state.hce_ema, h_val, aq if dt > 0 else 1.0)
        except Exception:
            pass


# ----------------------------- mode file -----------------------------

MODE_OFF = "OFF"
MODE_MAX_FOCUS = "MAX_FOCUS"
MODE_MAX_Q = "MAX_Q"
MODE_STABILIZE = "STABILIZE"

KEY_TO_MODE = {
    pygame.K_0: MODE_OFF,
    pygame.K_1: MODE_MAX_FOCUS,
    pygame.K_2: MODE_MAX_Q,
    pygame.K_3: MODE_STABILIZE,
}


# ----------------------------- render quality presets -----------------------------

QUALITY_PRESET_ORDER = ["perf", "balanced", "ultra", "4k"]

# These presets are tuned for TV-friendly “Turrell-ish” softness + high perceived resolution.
# You can still override at launch with --render-scale / --bloom-down / --bloom-blur-passes.
QUALITY_PRESETS = {
    "perf": {"render_scale": 0.35, "bloom_down": 4, "bloom_blur_passes": 3},
    "balanced": {"render_scale": 0.55, "bloom_down": 4, "bloom_blur_passes": 3},
    "ultra": {"render_scale": 0.70, "bloom_down": 3, "bloom_blur_passes": 4},
    "4k": {"render_scale": 0.80, "bloom_down": 3, "bloom_blur_passes": 4},
}


def apply_quality_preset(args: argparse.Namespace, preset: str) -> None:
    """Apply a quality preset live (hotkey) by updating render_scale + bloom params."""
    p = QUALITY_PRESETS.get(preset) or QUALITY_PRESETS["balanced"]
    args.quality = preset
    args.render_scale = float(p["render_scale"])
    args.bloom_down = max(2, int(p["bloom_down"]))
    args.bloom_blur_passes = max(1, int(p["bloom_blur_passes"]))

def read_mode_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            m = obj.get("mode") or obj.get("adaptive_mode") or obj.get("state")
        else:
            m = obj
        if not m:
            return None
        m = str(m).strip().upper()
        if m in ("OFF", "0"):
            return MODE_OFF
        if m in ("MAX_FOCUS", "FOCUS", "1"):
            return MODE_MAX_FOCUS
        if m in ("MAX_Q", "Q", "2"):
            return MODE_MAX_Q
        if m in ("STABILIZE", "STAB", "STABILITY", "3"):
            return MODE_STABILIZE
        return None
    except Exception:
        return None


def write_mode_file(path: str, mode: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mode": mode}, f)
    except Exception:
        pass


# ----------------------------- Turrell-ish renderer -----------------------------

if HAVE_NUMPY:

    def hex_to_srgb01(h: str) -> np.ndarray:
        h = h.strip().lstrip("#")
        return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


    def srgb_to_linear(c: np.ndarray) -> np.ndarray:
        c = np.clip(c, 0.0, 1.0)
        a = 0.055
        return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1.0 + a)) ** 2.4)


    def linear_to_srgb(c: np.ndarray) -> np.ndarray:
        c = np.clip(c, 0.0, None)
        a = 0.055
        return np.where(c <= 0.0031308, 12.92 * c, (1.0 + a) * (c ** (1.0 / 2.4)) - a)


    def make_uv(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        u = (x / (w - 1.0)) * 2.0 - 1.0
        v = (y / (h - 1.0)) * 2.0 - 1.0
        u *= (w / h)  # aspect correction
        return u, v


    def sdf_round_rect(u, v, cx, cy, hx, hy, r):
        qx = np.abs(u - cx) - (hx - r)
        qy = np.abs(v - cy) - (hy - r)
        qx2 = np.maximum(qx, 0.0)
        qy2 = np.maximum(qy, 0.0)
        outside = np.sqrt(qx2 * qx2 + qy2 * qy2) - r
        inside = np.minimum(np.maximum(qx, qy), 0.0)
        return outside + inside


    def sdf_convex_poly(u, v, verts):
        # verts CCW; returns negative inside, positive outside (SDF-ish)
        p0 = u
        p1 = v
        dmax = np.full(u.shape, -1e9, dtype=np.float32)
        for i in range(len(verts)):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % len(verts)]
            ex, ey = (bx - ax), (by - ay)
            nx, ny = (ey, -ex)  # outward normal
            inv = 1.0 / (math.sqrt(nx * nx + ny * ny) + 1e-8)
            nx *= inv
            ny *= inv
            di = (p0 - ax) * nx + (p1 - ay) * ny
            dmax = np.maximum(dmax, di)
        return dmax


    def blur_box(img: np.ndarray, passes: int = 3) -> np.ndarray:
        out = img
        for _ in range(passes):
            out = (
                out
                + np.roll(out, 1, 0)
                + np.roll(out, -1, 0)
                + np.roll(out, 1, 1)
                + np.roll(out, -1, 1)
                + np.roll(np.roll(out, 1, 0), 1, 1)
                + np.roll(np.roll(out, 1, 0), -1, 1)
                + np.roll(np.roll(out, -1, 0), 1, 1)
                + np.roll(np.roll(out, -1, 0), -1, 1)
            ) / 9.0
        return out


    def add_bloom(lin_rgb: np.ndarray, strength: float, threshold: float, down: int, blur_passes: int) -> np.ndarray:
        small = lin_rgb[::down, ::down]
        bright = np.clip(small - threshold, 0.0, None)
        blur = blur_box(bright, passes=blur_passes)
        up = np.repeat(np.repeat(blur, down, axis=0), down, axis=1)
        up = up[: lin_rgb.shape[0], : lin_rgb.shape[1]]
        return lin_rgb + strength * up


    def tonemap_exp(lin_rgb: np.ndarray, exposure: float) -> np.ndarray:
        return 1.0 - np.exp(-lin_rgb * exposure)


    def dither(lin_rgb: np.ndarray, amp: float) -> np.ndarray:
        noise = (np.random.rand(*lin_rgb.shape).astype(np.float32) - 0.5) * amp
        return np.clip(lin_rgb + noise, 0.0, None)


    def palette_cycle(colors_lin, t: float, speed: float) -> np.ndarray:
        n = len(colors_lin)
        p = (t * speed) % n
        i = int(p)
        f = p - i
        f = f * f * (3.0 - 2.0 * f)
        c0 = colors_lin[i]
        c1 = colors_lin[(i + 1) % n]
        return c0 * (1.0 - f) + c1 * f


    class TurrellishRenderer:
        def __init__(self, w: int, h: int, bloom_down: int = 4, bloom_blur_passes: int = 3):
            self.bloom_down = int(bloom_down)
            self.bloom_blur_passes = int(bloom_blur_passes)
            self.resize(w, h)

            # requested palette anchors
            self.turrell_lin = [srgb_to_linear(hex_to_srgb01(h)) for h in TURRELL_5_PALETTE]

            # single anchor for vignette depth
            self.deep = self.turrell_lin[0]
            self.white = self.turrell_lin[-1]

            # Legacy color aliases mapped into the Turrell palette
            self.marj = self.turrell_lin[2]   # violet
            self.mag = self.turrell_lin[4]    # fuchsia
            self.org = self.turrell_lin[6]    # blaze orange
            self.yel = self.turrell_lin[7]    # amber
            self.cyn = self.turrell_lin[8]    # lavender/soft highlight
            self.grn = self.turrell_lin[1]    # indigo proxy

            # Additional palettes (match v3 intent)
            self.palette_cool = [
                self.turrell_lin[1],  # indigo
                self.turrell_lin[2],  # violet
                self.turrell_lin[4],  # fuchsia
                self.turrell_lin[8],  # lavender/soft highlight
            ]
            self.palette_green = [
                self.grn,
                self.cyn,
                self.white,
            ]

        def resize(self, w: int, h: int) -> None:
            self.w = int(w)
            self.h = int(h)
            self.u, self.v = make_uv(self.w, self.h)

        def _room_base(self, wall_col: np.ndarray, t: float) -> np.ndarray:
            u, v = self.u, self.v
            r = np.sqrt((u * 0.95) ** 2 + (v * 0.90) ** 2)
            vign = 1.0 - 0.28 * smoothstep(0.22, 1.60, r)
            vertical = smoothstep(-1.15, 0.25, v)

            room = wall_col[None, None, :] * (0.84 + 0.16 * vertical)[..., None]
            room = room * vign[..., None] + self.deep[None, None, :] * (1.0 - vign[..., None]) * 0.85

            # ultra-subtle temporal breathing in the "air"
            air = 0.015 * math.sin(t * 0.55)
            room *= (1.0 + air)
            return room

        def _apply_post(self, lin: np.ndarray, exposure: float, bloom_strength: float, bloom_thr: float, micro_texture_gain: float, warmth_bias: float) -> np.ndarray:
            lin = add_bloom(lin, strength=bloom_strength, threshold=bloom_thr, down=self.bloom_down, blur_passes=self.bloom_blur_passes)
            lin = tonemap_exp(lin, exposure=exposure * (1.0 + float(warmth_bias)))
            lin = dither(lin, amp=(0.9 / 255.0) * float(micro_texture_gain))
            srgb = linear_to_srgb(lin)
            return (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        def render(
            self,
            mode: str,
            x: float,
            q: float,
            hce: float,
            valid_ema: float,
            qc_ema: float,
            t: float,
            exposure: float,
            bloom_strength: float,
            bloom_thr: float,
            drift_speed: float,
            micro_texture_gain: float,
            warmth_bias: float,
        ) -> np.ndarray:
            # global gating
            valid_factor = lerp(0.18, 1.0, clamp01(valid_ema))
            qc_factor = clamp01((qc_ema - 0.35) / 0.55)  # 0 if qc~0.35, 1 if qc~0.90
            gate = lerp(0.35, 1.0, valid_factor * qc_factor)

            # HCE-driven transcendence scale
            h_norm = clampf(hce / 50.0, 0.0, 1.0)
            peak = smoothstep(0.25, 1.0, h_norm)
            gold_portal_boost = 1.0 + 1.8 * peak
            cosmic_depth = 1.0 + 1.2 * peak

            # Style presets: all share OFF palette; only style parameters vary
            style = {
                "exp": 1.00,
                "bloom": 1.00,
                "bloom_thr": 1.00,
                "drift": 1.00,
                "micro": 1.00,
                "warmth": 0.0,
            }
            if mode == MODE_MAX_FOCUS:
                style.update({"exp": 0.95, "bloom": 0.90, "bloom_thr": 0.95, "drift": 1.15, "micro": 0.95, "warmth": -0.015})
            elif mode == MODE_MAX_Q:
                style.update({"exp": 1.05, "bloom": 1.05, "bloom_thr": 0.90, "drift": 1.10, "micro": 1.00, "warmth": 0.01})
            elif mode == MODE_STABILIZE:
                style.update({"exp": 1.00, "bloom": 1.00, "bloom_thr": 1.05, "drift": 0.90, "micro": 1.05, "warmth": -0.005})

            exposure_eff = clampf(exposure * style["exp"] * (1.0 + 0.5 * peak), 0.60, 2.20)
            bloom_strength_eff = clampf(bloom_strength * style["bloom"] * (1.0 + 0.8 * peak), 0.18, 2.0)
            bloom_thr_eff = clampf(bloom_thr * style["bloom_thr"], 0.12, 0.85)
            drift_eff = clampf(drift_speed * style["drift"], 0.35, 1.60)
            micro_eff = clampf(micro_texture_gain * style["micro"], 0.65, 1.30)
            warmth_eff = clampf(warmth_bias + style.get("warmth", 0.0), -0.08, 0.08)

            return self._render_off(
                x=x,
                q=q,
                hce=h_norm,
                gate=gate,
                t=t,
                exposure=exposure_eff,
                bloom_strength=bloom_strength_eff,
                bloom_thr=bloom_thr_eff,
                drift_speed=drift_eff,
                micro_texture_gain=micro_eff,
                warmth_bias=warmth_eff,
                cosmic_depth=cosmic_depth,
                gold_boost=gold_portal_boost,
            )

        def palette_at(self, t: float) -> np.ndarray:
            t = clamp01(t) * (len(self.turrell_lin) - 1)
            i = int(t)
            f = t - i
            c0 = self.turrell_lin[i]
            c1 = self.turrell_lin[min(i + 1, len(self.turrell_lin) - 1)]
            return c0 * (1.0 - f) + c1 * f

        def palette_q_triplet(self, q: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            q0 = clamp01(q)
            portal_q = clamp01(q0 + 0.12)
            rim_q = clamp01(q0 + 0.22)
            wall = self.palette_at(q0)
            portal = self.palette_at(portal_q)
            rim = self.palette_at(rim_q)
            return wall, portal, rim

        def _render_off(self, x: float, q: float, hce: float, gate: float, t: float, exposure: float, bloom_strength: float, bloom_thr: float, drift_speed: float, micro_texture_gain: float, warmth_bias: float, cosmic_depth: float, gold_boost: float) -> np.ndarray:
            # OFF palette for all modes; gentle motion and clamped brightness to prevent whiteout
            sp = 0.05 * (0.30 + q) * drift_speed  # slow drift
            motion = 0.03 * math.sin(t * 0.25 * drift_speed)
            wobble = 0.02 * math.cos(t * 0.18 * drift_speed)
            q_for_palette = clamp01(0.15 + 0.55 * q + 0.05 * math.sin(t * 0.12))
            wall, portal, rim = self.palette_q_triplet(q_for_palette)

            # Subtle motion in portal position/size to avoid frozen look
            cx = motion
            cy = -0.06 + wobble * 0.5
            aspect = self.w / self.h
            hx = 0.74 * aspect * (1.0 + 0.02 * math.sin(t * 0.31))
            hy = 0.54 * (1.0 + 0.02 * math.cos(t * 0.27))

            lin = self._room_base(wall, t)
            d = sdf_round_rect(self.u, self.v, cx=cx, cy=cy, hx=hx, hy=hy, r=0.09)

            portal_mask = 1.0 - smoothstep(-0.02, 0.065, d)
            halo = np.exp(-(np.maximum(d, 0.0) / (0.22 + 0.08 * q_for_palette)) ** 2)
            rim_mask = np.exp(-(np.abs(d) / 0.010) ** 2)

            base_intensity = (0.35 + 1.10 * q_for_palette) * gate * cosmic_depth
            intensity = clampf(base_intensity, 0.10, 1.65)

            lin += portal_mask[..., None] * portal[None, None, :] * (1.85 * intensity * gold_boost)
            lin += halo[..., None] * portal[None, None, :] * (0.50 * intensity * gold_boost)
            lin += rim_mask[..., None] * rim[None, None, :] * (2.05 * intensity * gold_boost)

            # HCE cosmic bloom (mandala glow) when elevated
            if hce > 0.05:
                r = np.sqrt(self.u ** 2 + self.v ** 2)
                petals = 0.35 + 0.65 * np.cos(12.0 * (math.pi * r + t * 0.3))
                bloom = np.exp(-((r * 1.8) ** 2)) * petals * (0.6 + 1.8 * hce)
                gold = np.array([1.0, 0.82, 0.38], dtype=np.float32)
                lin += bloom[..., None] * gold[None, None, :] * intensity

            # subtle floor cue
            floor = smoothstep(0.10, 1.00, self.v)
            lin *= (1.0 - 0.08 * floor)[..., None]

            # keep exposure/bloom within safe bounds to avoid whiteout
            exposure_eff = clampf(exposure * 1.02, 0.65, 1.85)
            bloom_strength_eff = clampf(bloom_strength * 1.05, 0.18, 1.65)
            bloom_thr_eff = clampf(bloom_thr, 0.18, 0.85)
            micro_eff = clampf(micro_texture_gain, 0.65, 1.25)

            return self._apply_post(lin, exposure=exposure_eff, bloom_strength=bloom_strength_eff, bloom_thr=bloom_thr_eff, micro_texture_gain=micro_eff, warmth_bias=warmth_bias)

        def _render_focus(self, x: float, q: float, gate: float, t: float, exposure: float, bloom_strength: float, bloom_thr: float, drift_speed: float, micro_texture_gain: float, warmth_bias: float) -> np.ndarray:
            # MAX_FOCUS = cool, tighter portal, less motion; brightness responds to X
            sp = 0.06 * drift_speed
            wall = palette_cycle(self.palette_cool, t, speed=sp)
            portal = palette_cycle([self.white, self.cyn, self.marj], t + 1.3, speed=sp * 1.10)
            rim = palette_cycle([self.cyn, self.mag], t + 0.7, speed=sp * 1.25)

            amp = 0.07 + 0.06 * q
            cx = amp * math.sin(t * (0.35 + 0.55 * x))
            cy = -0.05 + amp * math.cos(t * (0.28 + 0.45 * x))

            aspect = self.w / self.h
            hx = (0.58 - 0.10 * q) * aspect
            hy = 0.48 - 0.05 * q

            lin = self._room_base(wall, t)
            d = sdf_round_rect(self.u, self.v, cx=cx, cy=cy, hx=hx, hy=hy, r=0.10)

            portal_mask = 1.0 - smoothstep(-0.02, 0.060, d)
            halo = np.exp(-(np.maximum(d, 0.0) / 0.18) ** 2)
            rim_mask = np.exp(-(np.abs(d) / 0.012) ** 2)

            intensity = (0.70 + 1.25 * x) * gate

            lin += portal_mask[..., None] * portal[None, None, :] * (2.25 * intensity)
            lin += halo[..., None] * portal[None, None, :] * (0.50 * intensity)
            lin += rim_mask[..., None] * rim[None, None, :] * (2.15 * intensity)

            return self._apply_post(lin, exposure=exposure * 1.10, bloom_strength=bloom_strength * 1.05, bloom_thr=bloom_thr * 0.95, micro_texture_gain=micro_texture_gain, warmth_bias=warmth_bias)

        def _render_stabilize(self, x: float, q: float, gate: float, t: float, exposure: float, bloom_strength: float, bloom_thr: float, drift_speed: float, micro_texture_gain: float, warmth_bias: float) -> np.ndarray:
            # STABILIZE = green/cyan room with slow "breath"; Q influences pulse amplitude
            sp = 0.05 * drift_speed
            wall = palette_cycle(self.palette_green, t, speed=sp)
            portal = palette_cycle([self.grn, self.cyn, self.white], t + 2.0, speed=sp * 1.05)
            rim = palette_cycle([self.cyn, self.marj], t + 0.8, speed=sp * 1.20)

            breath = 0.5 + 0.5 * math.sin(t * (0.22 + 0.10 * x) + 1.5 * q)
            breath = breath ** 1.6

            cx = 0.02 * math.sin(t * 0.30)
            cy = -0.07 + 0.04 * math.cos(t * 0.22)

            aspect = self.w / self.h
            hx = (0.72 + 0.06 * breath) * aspect
            hy = 0.50 + 0.05 * breath

            lin = self._room_base(wall, t)
            d = sdf_round_rect(self.u, self.v, cx=cx, cy=cy, hx=hx, hy=hy, r=0.11)

            portal_mask = 1.0 - smoothstep(-0.02, 0.070, d)
            halo = np.exp(-(np.maximum(d, 0.0) / (0.22 + 0.08 * breath)) ** 2)
            rim_mask = np.exp(-(np.abs(d) / 0.011) ** 2)

            intensity = (0.78 + 0.55 * breath + 0.35 * q) * gate

            lin += portal_mask[..., None] * portal[None, None, :] * (1.95 * intensity)
            lin += halo[..., None] * portal[None, None, :] * (0.62 * intensity)
            lin += rim_mask[..., None] * rim[None, None, :] * (1.90 * intensity)

            return self._apply_post(lin, exposure=exposure * 1.00, bloom_strength=bloom_strength * 1.15, bloom_thr=bloom_thr, micro_texture_gain=micro_texture_gain, warmth_bias=warmth_bias)

        def _render_max_q(self, x: float, q: float, gate: float, t: float, exposure: float, bloom_strength: float, bloom_thr: float, drift_speed: float, micro_texture_gain: float, warmth_bias: float) -> np.ndarray:
            # MAX_Q = Wedgework-ish nested frame + angled slab + neon edges
            sp = 0.08 * drift_speed + 0.06 * q
            field = palette_cycle([self.deep, srgb_to_linear(hex_to_srgb01("#2F0101")), self.deep], t, speed=sp)
            slab = palette_cycle([self.org, self.yel, self.mag], t + 1.2, speed=sp * 1.10)
            rim = palette_cycle([self.mag, self.marj, self.yel], t + 0.5, speed=sp * 1.30)

            lin = np.zeros((self.h, self.w, 3), dtype=np.float32) + self.deep

            aspect = self.w / self.h

            # big frame
            d_big = sdf_round_rect(self.u, self.v, 0.0, 0.05, hx=0.80 * aspect, hy=0.72, r=0.03)
            inside_big = 1.0 - smoothstep(0.0, 0.035, d_big)
            halo_big = np.exp(-(np.maximum(d_big, 0.0) / 0.15) ** 2)

            intensity = (0.85 + 0.85 * q + 0.25 * x) * gate

            lin += inside_big[..., None] * field[None, None, :] * (1.20 * intensity)
            lin += halo_big[..., None] * field[None, None, :] * (0.70 * intensity)

            border = np.exp(-(np.abs(d_big) / 0.007) ** 2)
            lin += border[..., None] * rim[None, None, :] * (3.20 * intensity)

            # inner trapezoid slab
            tilt = lerp(0.05, 0.22, q)
            verts = [
                (-0.42 * aspect, -0.18 + tilt),
                (0.33 * aspect, -0.27),
                (0.35 * aspect, 0.24),
                (-0.42 * aspect, 0.22 - tilt),
            ]
            d_poly = sdf_convex_poly(self.u, self.v, verts)
            poly = 1.0 - smoothstep(0.0, 0.03, d_poly)

            gx = smoothstep(-0.5 * aspect, 0.55 * aspect, self.u)
            slab_fill = slab[None, None, :] * (0.70 + 0.45 * gx)[..., None]
            lin += poly[..., None] * slab_fill * (1.70 * intensity)

            # neon edge glows (two edges)
            # edge 1 (right)
            a = np.array(verts[1], dtype=np.float32)
            b = np.array(verts[2], dtype=np.float32)
            e = b - a
            n = np.array([e[1], -e[0]], dtype=np.float32)
            n /= (np.linalg.norm(n) + 1e-8)
            d_edge = (self.u - a[0]) * n[0] + (self.v - a[1]) * n[1]
            lin += (np.exp(-(np.abs(d_edge) / 0.010) ** 2) * poly)[..., None] * self.marj[None, None, :] * (2.70 * intensity)

            # edge 2 (top)
            a2 = np.array(verts[0], dtype=np.float32)
            b2 = np.array(verts[1], dtype=np.float32)
            e2 = b2 - a2
            n2 = np.array([e2[1], -e2[0]], dtype=np.float32)
            n2 /= (np.linalg.norm(n2) + 1e-8)
            d_edge2 = (self.u - a2[0]) * n2[0] + (self.v - a2[1]) * n2[1]
            lin += (np.exp(-(np.abs(d_edge2) / 0.010) ** 2) * poly)[..., None] * self.mag[None, None, :] * (1.90 * intensity)

            return self._apply_post(lin, exposure=exposure * 0.98, bloom_strength=bloom_strength * 1.25, bloom_thr=bloom_thr * 0.90, micro_texture_gain=micro_texture_gain, warmth_bias=warmth_bias)


# ----------------------------- HUD -----------------------------




def draw_live_dot(screen: pygame.Surface, alive: bool, stream_valid: bool, qualia_valid: bool, x: int, y: int) -> None:
    """Draw a small status dot:
    - red: no packets recently (STALE)
    - amber: packets arriving but invalid / qualia-invalid
    - green: packets arriving and valid
    """
    if not alive:
        color = (220, 60, 60)
    else:
        if stream_valid and qualia_valid:
            color = (60, 220, 120)
        elif stream_valid:
            color = (240, 200, 80)
        else:
            color = (240, 140, 60)
    pygame.draw.circle(screen, color, (x, y), 10)
    pygame.draw.circle(screen, (0, 0, 0), (x, y), 10, 2)
def draw_hud(
    screen: pygame.Surface,
    font,
    mode: str,
    follower: NDJSONFollower,
    ndjson_path: str,
    demo: bool,
    args: argparse.Namespace,
) -> None:
    if font is None:
        return

    def _render_text(txt: str, color=(230, 230, 230)):
        try:
            surf = font.render(txt, True, color)
            # pygame.freetype returns (surf, rect)
            if isinstance(surf, tuple):
                surf = surf[0]
            return surf
        except TypeError:
            try:
                surf, _ = font.render(txt, color)
                return surf
            except Exception:
                return None
        except Exception:
            return None

    st = follower.state
    hce = getattr(st, "hce", 0.0) or 0.0
    hce_ema = getattr(st, "hce_ema", 0.0) or 0.0
    hce_color = (255, 215, 120) if hce_ema >= 20 else (200, 230, 255)
    hce_glow = hce_ema >= 20

    lines = [
        f"mode={mode}    demo={demo}",
        f"valid={st.valid} (visual)  v_ema={st.valid_ema:.2f}  |  stream_valid={st.stream_valid}  s_ema={st.stream_valid_ema:.2f}  qc={st.qc:.2f} qc_ema={st.qc_ema:.2f}",
        f"valid_source={follower.valid_source}  qc_thr={follower.quality_valid_thr:.2f}  grace={follower.valid_grace_sec:.2f}s  freeze={'on' if follower.freeze_on_invalid else 'off'}  |  q_source={follower.q_source} q_metric={follower.q_metric} raw_when_invalid={'on' if follower.raw_when_invalid else 'off'}",
        f"render: preset={getattr(args, 'quality', 'balanced')}  scale={float(getattr(args, 'render_scale', 0.0)):.2f}  bloom_down={int(getattr(args, 'bloom_down', 0))}  blur={int(getattr(args, 'bloom_blur_passes', 0))}",
        f"Xraw={st.x_raw:.3f}  Qraw={st.q_raw:.3f} (q_src={st.q_origin})  |  Xn={st.x:.3f}  Qn={st.q:.3f}  HCE={hce:.2f}  HCE_ema={hce_ema:.2f}",
        f"C_pe_n: {(st.c_pe_n if st.option_e_present else float('nan')):.3f}" if st.option_e_present else "C_pe_n: --",
        f"S_n: {(st.s_flat_n if st.option_e_present else float('nan')):.3f}" if st.option_e_present else "S_n: --",
        f"invalid_age={st.invalid_age:.2f}s  reasons={('|'.join(st.reasons) if st.reasons else '-')}",
        f"ranges: X[{follower.x_min:.2f},{follower.x_max:.2f}]  Q[{follower.q_min:.3f},{follower.q_max:.3f}]",
        f"stream_age: { (time.time() - follower.last_pkt_walltime):.2f}s" if follower.last_pkt_walltime else "stream_age: n/a",
        f"ndjson={ndjson_path}",
        "keys: 1=FOCUS  2=Q  3=STAB  0=OFF  |  F=fullscreen  H=HUD  D=demo  V=Q-source  M=Q-metric  C=valid-source  P=quality  Esc=quit",
    ]

    y = 14
    for s in lines:
        surf = _render_text(s)
        if surf:
            screen.blit(surf, (14, y))
            y += surf.get_height() + 2

    # HCE emphasis (glow) in upper-right
    try:
        x = screen.get_width() - 200
        y0 = 18
        hce_text = f"HCE {hce_ema:.1f}"
        base_color = hce_color
        surf = _render_text(hce_text, color=base_color)
        if surf:
            if hce_glow:
                glow_color = (255, 215, 120)
                for off in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    glow = _render_text(hce_text, color=glow_color)
                    if glow:
                        screen.blit(glow, (x + 4 + off[0], y0 + off[1]))
            screen.blit(surf, (x + 4, y0))
    except Exception:
        pass


# ----------------------------- main -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndjson", default="state_stream.ndjson", help="Path to state_stream.ndjson")
    ap.add_argument("--mode-file", default="spotify_proto/adaptive_mode.json", help="Path to adaptive_mode.json")

    ap.add_argument("--display", type=int, default=0, help="Which display index to open on (0=main, 1=TV)")
    ap.add_argument("--fullscreen", action="store_true", help="Start fullscreen")
    ap.add_argument("--window", default="1280x720", help="Window size if not fullscreen, e.g. 1280x720")

    ap.add_argument("--render-scale", type=float, default=None, help="Internal render scale vs window size (0.25-0.6 recommended)")

    ap.add_argument("--x-min", type=float, default=0.85)
    ap.add_argument("--x-max", type=float, default=2.10)
    ap.add_argument("--q-min", type=float, default=0.00)
    ap.add_argument("--q-max", type=float, default=0.18)

    # Visual validity behavior (for "magical" feel)
    ap.add_argument("--quality-valid-thr", type=float, default=0.25, help="QC threshold used when valid_source=qc (and as fallback)")
    ap.add_argument("--valid-grace-sec", type=float, default=1.50, help="Treat brief invalid bursts as visually-valid for this long (seconds)")
    ap.add_argument("--valid-source", default="auto", choices=["auto", "stream", "qc", "qualia"], help="How to determine base validity for accepting new metrics")
    ap.add_argument("--no-freeze-on-invalid", action="store_true", help="If set, keep ingesting X/Q even when invalid (NOT recommended)")
    
    # Q selection (masked vs raw) — for testing/debug
    ap.add_argument("--q-source", default="auto", choices=["auto", "masked", "raw"], help="Which Q signal to use: masked (recommended), raw (unmasked), or auto (masked then raw)")
    ap.add_argument("--q-metric", default="auto", choices=["auto", "abs", "focus", "perx", "vibe", "vibe_focus", "vibe_focus_e", "meaningful", "focus_meaningful"], help="Prefer Q_abs or Q_focus when both exist (auto=abs-first, matches v3)")
    ap.add_argument("--raw-when-invalid", action="store_true", help="Allow raw-Q to update even when invalid (debug/testing; may look jittery)")
    ap.add_argument("--qualia-valid-artifact-thr", type=float, default=0.55, help="Threshold for artifact_quality used to derive qualia_valid when not provided by stream")

    ap.add_argument("--quality", default="balanced", choices=["perf", "balanced", "ultra", "4k"], help="Render quality preset (controls internal scale + bloom)")
    ap.add_argument("--bloom-down", type=int, default=4, help="Bloom downsample factor (lower=sharper but slower)")
    ap.add_argument("--bloom-blur-passes", type=int, default=3, help="Bloom blur passes (higher=softer but slower)")

    ap.add_argument("--exposure", type=float, default=1.70)
    ap.add_argument("--bloom-strength", type=float, default=1.35)
    ap.add_argument("--bloom-threshold", type=float, default=0.36)
    ap.add_argument("--drift-speed", type=float, default=1.00)

    ap.add_argument("--hud", action="store_true", help="Start with HUD on")
    ap.add_argument("--demo", action="store_true", help="Start in demo mode (simulated X/Q)")

    args = ap.parse_args()

    # Quality preset (unless user explicitly sets --render-scale)
    if args.render_scale is None:
        if args.quality == "perf":
            args.render_scale = 0.35
            args.bloom_down = max(4, int(args.bloom_down))
            args.bloom_blur_passes = min(3, int(args.bloom_blur_passes))
        elif args.quality == "ultra":
            args.render_scale = 0.70
            args.bloom_down = max(3, int(args.bloom_down))
            args.bloom_blur_passes = max(4, int(args.bloom_blur_passes))
        elif args.quality == "4k":
            args.render_scale = 0.80
            args.bloom_down = max(3, int(args.bloom_down))
            args.bloom_blur_passes = max(4, int(args.bloom_blur_passes))
        else:
            # balanced
            args.render_scale = 0.55

    args.bloom_down = max(2, int(args.bloom_down))
    args.bloom_blur_passes = max(1, int(args.bloom_blur_passes))

    if not HAVE_NUMPY:

        print("ERROR: This upgraded renderer needs numpy. Install it in your venv:  pip install numpy")
        return 2

    pygame.init()
    # font setup with fallbacks (freetype if available)
    font = None
    try:
        pygame.font.init()
        font = pygame.font.SysFont("Menlo", 18)
        if font is None:
            raise RuntimeError("SysFont returned None")
    except Exception:
        try:
            if ft:
                font = ft.SysFont("Menlo", 18)
        except Exception:
            font = None

    window_w, window_h = parse_res(args.window)

    flags = pygame.SCALED
    if args.fullscreen:
        flags |= pygame.FULLSCREEN
        size = pick_display_size(args.display)
        if size == (0, 0):
            size = (0, 0)  # last resort fallback
    else:
        flags |= pygame.RESIZABLE
        size = (window_w, window_h)

    try:
        screen = pygame.display.set_mode(size, flags, display=args.display)
    except TypeError:
        # older pygame without display arg
        if args.fullscreen:
            os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(args.display)
        screen = pygame.display.set_mode(size, flags)

    pygame.display.set_caption("X/Q Turrell Room (2D) — v5")
    clock = pygame.time.Clock()

    follower = NDJSONFollower(
        path=args.ndjson,
        x_min=args.x_min,
        x_max=args.x_max,
        q_min=args.q_min,
        q_max=args.q_max,
        quality_valid_thr=args.quality_valid_thr,
        valid_grace_sec=args.valid_grace_sec,
        valid_source=args.valid_source,
        freeze_on_invalid=(not args.no_freeze_on_invalid),
        q_source=args.q_source,
        q_metric=args.q_metric,
        raw_when_invalid=bool(args.raw_when_invalid),
        qualia_valid_artifact_thr=float(args.qualia_valid_artifact_thr),
    )

    mode = MODE_OFF
    last_mode_poll = 0.0

    show_hud = True if font else False
    demo = bool(args.demo)

    renderer: Optional[TurrellishRenderer] = None
    render_w = render_h = 0

    t0 = time.time()

    running = True
    q_vis = 0.0
    q_vis_initialized = False
    toast_msg = ""
    toast_until = 0.0
    while running:
        now = time.time()
        t = now - t0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key in KEY_TO_MODE:
                    mode = KEY_TO_MODE[e.key]
                    write_mode_file(args.mode_file, mode)
                    toast_msg = f"mode → {mode}"
                    toast_until = time.time() + 1.5
                    pygame.display.set_caption(f"X/Q Turrell Room (2D) — v5  |  {mode}")
                elif e.key == pygame.K_h:
                    show_hud = not show_hud
                    toast_msg = "HUD on" if show_hud else "HUD off"
                    toast_until = time.time() + 1.0
                elif e.key == pygame.K_d:
                    demo = not demo
                elif e.key == pygame.K_v:
                    # cycle Q source: auto → masked → raw
                    order = ["auto", "masked", "raw"]
                    cur = (follower.q_source or "auto").strip().lower()
                    if cur not in order:
                        cur = "auto"
                    nxt = order[(order.index(cur) + 1) % len(order)]
                    follower.q_source = nxt
                elif e.key == pygame.K_m:
                    # cycle Q metric
                    order = ["abs", "focus", "vibe", "vibe_focus", "vibe_focus_e", "perx", "meaningful", "focus_meaningful", "auto"]
                    cur = (follower.q_metric or "auto").strip().lower()
                    if cur not in order:
                        cur = "auto"
                    nxt = order[(order.index(cur) + 1) % len(order)]
                    follower.q_metric = nxt
                elif e.key == pygame.K_c:
                    # cycle base validity source: stream → qc → qualia → auto
                    order = ["stream", "qc", "qualia", "auto"]
                    cur = (follower.valid_source or "auto").strip().lower()
                    if cur not in order:
                        cur = "auto"
                    nxt = order[(order.index(cur) + 1) % len(order)]
                    follower.valid_source = nxt
                elif e.key == pygame.K_p:
                    # cycle render quality presets live: perf → balanced → ultra → 4k
                    order = QUALITY_PRESET_ORDER
                    cur = str(getattr(args, "quality", "balanced") or "balanced").strip().lower()
                    if cur not in order:
                        cur = "balanced"
                    nxt = order[(order.index(cur) + 1) % len(order)]
                    apply_quality_preset(args, nxt)
                    # force renderer rebuild on next frame
                    renderer = None
                    render_w = render_h = 0
                    toast_msg = f"quality: {nxt}  (scale={args.render_scale:.2f}, bloom_down={args.bloom_down}, blur={args.bloom_blur_passes})"
                    toast_until = time.time() + 2.0
                    pygame.display.set_caption(f"X/Q Turrell Room (2D) — v5  |  {toast_msg}")
                elif e.key == pygame.K_f:
                    # toggle fullscreen
                    args.fullscreen = not args.fullscreen
                    flags = pygame.SCALED
                    if args.fullscreen:
                        flags |= pygame.FULLSCREEN
                        size = pick_display_size(args.display)
                        if size == (0, 0):
                            size = (0, 0)
                    else:
                        flags |= pygame.RESIZABLE
                        size = (window_w, window_h)
                    try:
                        screen = pygame.display.set_mode(size, flags, display=args.display)
                    except TypeError:
                        screen = pygame.display.set_mode(size, flags)

        # poll mode-file (unless you want manual keys only)
        if (now - last_mode_poll) > 0.25:
            mf = read_mode_file(args.mode_file)
            if mf:
                mode = mf
            last_mode_poll = now

        # update from ndjson
        pkt = follower.poll_latest()
        if pkt is not None and not demo:
            follower.update_from_packet(pkt)

        # demo motion for quick testing
        if demo:
            # x: 0..1, q: 0..1
            follower.state.x = 0.15 + 0.85 * (0.5 + 0.5 * math.sin(t * 0.42))
            follower.state.q = 0.10 + 0.90 * (0.5 + 0.5 * math.sin(t * 0.33 + 1.1))
            follower.state.x_raw = lerp(args.x_min, args.x_max, follower.state.x)
            follower.state.q_raw = lerp(args.q_min, args.q_max, follower.state.q)
            follower.state.valid = True
            follower.state.valid_ema = 1.0
            follower.state.qc = 0.85
            follower.state.qc_ema = 0.85

        # internal render size
        sw, sh = screen.get_size()
        target_w = max(240, int(sw * args.render_scale))
        target_h = max(160, int(sh * args.render_scale))
        if renderer is None or target_w != render_w or target_h != render_h:
            renderer = TurrellishRenderer(target_w, target_h, bloom_down=int(args.bloom_down), bloom_blur_passes=int(args.bloom_blur_passes))
            render_w, render_h = target_w, target_h

        # slow, smooth Q just for visuals (sensual)
        if not q_vis_initialized:
            q_vis = clamp01(float(follower.state.q))
            q_vis_initialized = True
        alpha_vis = 0.035  # slow butter
        q_vis = ema(q_vis, clamp01(float(follower.state.q)), alpha_vis)

        drivers_present = bool(follower.state.option_e_present)
        c_pe_n = float(follower.state.c_pe_n) if drivers_present else 0.5
        s_flat_n = float(follower.state.s_flat_n) if drivers_present else 0.5
        micro_texture_gain = max(0.70, min(1.20, 0.85 + 0.30 * (c_pe_n - 0.5))) if drivers_present else 1.0
        warmth_bias = max(-0.05, min(0.05, 0.06 * (s_flat_n - 0.5))) if drivers_present else 0.0

        img = renderer.render(
            mode=mode,
            x=float(follower.state.x),
            q=float(q_vis),
            hce=float(follower.state.hce_ema),
            valid_ema=float(follower.state.valid_ema * (0.25 + 0.75 * (follower.state.qualia_valid_ema if ((follower.q_metric or "auto").strip().lower() in {"vibe", "vibe_focus", "vibe_focus_e", "meaningful", "focus_meaningful"}) else 1.0))),
            qc_ema=float(follower.state.qc_ema),
            t=t,
            exposure=float(args.exposure),
            bloom_strength=float(args.bloom_strength),
            bloom_thr=float(args.bloom_threshold),
            drift_speed=float(args.drift_speed),
            micro_texture_gain=float(micro_texture_gain),
            warmth_bias=float(warmth_bias),
        )

        surf = pygame.surfarray.make_surface(np.swapaxes(img, 0, 1))
        surf = pygame.transform.smoothscale(surf, (sw, sh))
        screen.blit(surf, (0, 0))

        # Stream heartbeat (LIVE/STALE)
        alive = (follower.last_pkt_walltime is not None) and ((now - follower.last_pkt_walltime) < 1.5)
        # If we're driving a qualia metric, show qualia_valid too; otherwise mirror stream_valid.
        qm = (follower.q_metric or "auto").strip().lower()
        qualia_metric = qm in {"vibe", "vibe_focus", "vibe_focus_e", "meaningful", "focus_meaningful"}
        qv = bool(follower.state.qualia_valid) if qualia_metric else True
        draw_live_dot(screen, alive=alive, stream_valid=bool(follower.state.stream_valid), qualia_valid=qv, x=sw - 24, y=24)

        if show_hud:
            draw_hud(screen, font, mode, follower, args.ndjson, demo, args)

        if toast_msg and (now < toast_until) and font:
            ts_surf = None
            try:
                ts_surf = font.render(toast_msg, True, (235, 235, 235))
                if isinstance(ts_surf, tuple):
                    ts_surf = ts_surf[0]
            except Exception:
                ts_surf = None
            if ts_surf:
                screen.blit(ts_surf, (14, sh - ts_surf.get_height() - 14))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
