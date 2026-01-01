"""Core Cortex compute + packet building (UI-free).

This is a distilled, non-Dash version of the Cortex Console compute pipeline,
keeping packet schema compatibility (Turrell v5.2) and preserving diagnostic
semantics:
- Channel agreement remains diagnostic-only (does not hard invalidate).
- Q_raw history uses seconds (not ms).
- Reliability object and raw metrics are preserved.
- Option-E is optional/neutral unless provided.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch
from scipy.spatial.distance import jensenshannon

from alma import config
from alma.engine import xq_metrics_v4_2_1 as xq_metrics

# Core constants (mirroring external script defaults)
FS_TARGET = 256
WINDOW_SEC = 5
STEP_SEC = 1
WINDOW_N = FS_TARGET * WINDOW_SEC
STEP_N = FS_TARGET * STEP_SEC
EPS = 1e-12
EMA_ALPHA_X = 0.18
HCE_DENOM_FLOOR = 1e-6
HCE_SCALE = 10000.0  # Super-linear scaled HCE for enhanced transcendent sensitivity (2026-01-01)

# Baseline fallbacks (used if no external baseline is loaded)
F_BASE = np.linspace(1, 45, 45, dtype=float)
P_REF = np.ones_like(F_BASE, dtype=float)
P_REF /= P_REF.sum()
MU_JS = 0.15
P90_JS = 0.35
TAU_PHI = (P90_JS - MU_JS) / np.log(2.0) if (P90_JS > MU_JS) else 0.10
GAMMA_REF = 0.25
BG_REF = 0.25
BASELINE_VERSION = "embedded_fallback"
BASELINE_WARNING: Optional[str] = None
BASELINE_LOADED: bool = False
BASELINE_PATH_USED: Optional[str] = None
CODE_VERSION = "cortex_core"
METRICS_VERSION = "v4_2_1"
_BASELINE_LOGGED = False


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def _band_mass(P_mean: np.ndarray, f_bins: np.ndarray, lo: float, hi: float) -> float:
    return float(P_mean[(f_bins >= lo) & (f_bins < hi)].sum())


def _compute_metrics(window_4ch: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute minimal X/Q metrics and diagnostics from a 5s window."""
    P_ch_list = []
    P_ch_raw_list = []
    f_all = None
    for ch_data in window_4ch:
        f, psd = welch(ch_data, fs=fs, nperseg=max(32, int(fs)))
        mask = (f >= 1) & (f <= 45)
        f_mask = f[mask]
        psd_mask = psd[mask]
        if f_all is None:
            f_all = f_mask
        P_ch_raw_list.append(psd_mask)
        P = psd_mask / (np.sum(psd_mask) + EPS)
        P_ch_list.append(P)

    if f_all is None or not P_ch_list:
        raise ValueError("No PSD available for metrics compute.")

    P_ch = np.vstack(P_ch_list)
    P_mean_raw = np.mean(P_ch_raw_list, axis=0)
    P_mean = np.mean(P_ch, axis=0)
    P_mean = np.maximum(P_mean, 0.0)
    P_mean = P_mean / (np.sum(P_mean) + EPS)

    # Align bins to baseline if different
    P_i = np.interp(F_BASE, f_all, P_mean)
    P_i = np.maximum(P_i, 0.0)
    P_i = P_i / (np.sum(P_i) + EPS)

    JS_dist = float(jensenshannon(P_i, P_REF))
    d = max(0.0, JS_dist - MU_JS)
    phi = float(1.0 - np.exp(-d / (TAU_PHI + EPS)))
    phi = float(np.clip(phi, 0.0, 1.0))

    theta = _band_mass(P_i, F_BASE, 4, 8)
    alpha = _band_mass(P_i, F_BASE, 8, 13)
    beta = _band_mass(P_i, F_BASE, 13, 30)
    gamma = _band_mass(P_i, F_BASE, 30, 45)
    total_bands = theta + alpha + beta + gamma
    theta_n = theta / (total_bands + EPS)
    alpha_n = alpha / (total_bands + EPS)
    beta_n = beta / (total_bands + EPS)
    gamma_n = gamma / (total_bands + EPS)

    # Simple engagement proxy (Option D-ish)
    R_abs = alpha + beta
    Q_abs_raw = alpha_n + beta_n
    Q_vibe_raw = alpha_n
    Q_vibe_focus_raw = (alpha_n + beta_n) / 2.0

    # EMG proxy via high-band mass (diagnostic only)
    emg_index = gamma_n
    w_emg = float(1.0 - _sigmoid((emg_index - 0.3) / 0.1))

    # Channel agreement diagnostic (not gating validity)
    try:
        ch_corr = float(np.corrcoef(window_4ch)[np.triu_indices(4, 1)].mean())
    except Exception:
        ch_corr = float("nan")
    ch_conf_raw = float(0.5 + 0.5 * np.clip(ch_corr, -1.0, 1.0))

    # Quality confidence (use band mass + EMG proxy)
    band_conf = float(np.clip(total_bands, 0.0, 1.0))
    quality_conf = float(np.clip(0.5 * band_conf + 0.5 * w_emg, 0.0, 1.0))
    valid = bool(quality_conf >= 0.25)

    return {
        "JS_dist": JS_dist,
        "phi": phi,
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "theta_n": theta_n,
        "alpha_n": alpha_n,
        "beta_n": beta_n,
        "gamma_n": gamma_n,
        "R_abs": R_abs,
        "Q_abs_raw": Q_abs_raw,
        "Q_vibe_raw": Q_vibe_raw,
        "Q_vibe_focus_raw": Q_vibe_focus_raw,
        "Q_abs": Q_abs_raw if valid else 0.0,
        "Q_vibe": Q_vibe_raw if valid else 0.0,
        "Q_vibe_focus": Q_vibe_focus_raw if valid else 0.0,
        "quality_conf": quality_conf,
        "valid": valid,
        "qualia_valid": valid,
        "reason_codes": "" if valid else "low_quality_conf",
        "w_emg": w_emg,
        "ch_conf_raw": ch_conf_raw,
        "band_conf": band_conf,
        "ch_corr": ch_corr,
    }


@dataclass
class CortexState:
    buffers: List[Deque[float]]
    ts: Deque[float]
    fs: float
    t0: float
    n_total_samples: int
    last_compute_samples: int
    history: Dict[str, List[float]]
    stream_meta: Dict[str, object]
    ema: Dict[str, Optional[float]]


def load_baseline(baseline_path: Optional[str] = None) -> None:
    """Load baseline with robust path resolution; record status."""
    global F_BASE, P_REF, MU_JS, P90_JS, TAU_PHI, BASELINE_VERSION, BASELINE_WARNING, BASELINE_LOADED, BASELINE_PATH_USED, GAMMA_REF, BG_REF
    try:
        xq_metrics.load_baseline(baseline_path)
        status = xq_metrics.baseline_status()
        F_BASE = np.array(xq_metrics.F_BASE, dtype=float)
        P_REF = np.array(xq_metrics.P_REF, dtype=float)
        MU_JS = float(xq_metrics.MU_JS)
        P90_JS = float(xq_metrics.P90_JS)
        TAU_PHI = float(xq_metrics.TAU_PHI)
        GAMMA_REF = float(status.get("gamma_ref", GAMMA_REF))
        BG_REF = float(status.get("bg_ref", BG_REF))
        BASELINE_VERSION = str(status.get("baseline_version", BASELINE_VERSION))
        BASELINE_WARNING = status.get("baseline_warning")
        BASELINE_LOADED = bool(status.get("baseline_loaded"))
        BASELINE_PATH_USED = status.get("baseline_path_used")
    except Exception:
        BASELINE_WARNING = "Using fallback baseline (xq load exception)"
        BASELINE_LOADED = False
        BASELINE_PATH_USED = None


def init_cortex_state(fs: float = FS_TARGET, baseline_path: Optional[str] = None) -> Dict[str, object]:
    """Initialize cortex state (buffers, history, config)."""
    load_baseline(baseline_path)
    global _BASELINE_LOGGED
    if not _BASELINE_LOGGED:
        status = {
            "baseline_loaded": BASELINE_LOADED,
            "baseline_version": BASELINE_VERSION,
            "baseline_path": BASELINE_PATH_USED,
            "gamma_ref": GAMMA_REF,
            "bg_ref": BG_REF,
            "metrics_version": METRICS_VERSION,
        }
        print(f"[cortex_core] baseline status: {status}")
        _BASELINE_LOGGED = True
    state = CortexState(
        buffers=[deque(maxlen=int(fs * 600)) for _ in range(4)],
        ts=deque(maxlen=int(fs * 600)),
        fs=float(fs),
        t0=time.time(),
        n_total_samples=0,
        last_compute_samples=0,
        history={
            "t": [],
            "X_raw": [],
            "X_ema": [],
            "HCE": [],
            "HCE_raw": [],
            "Q_abs_raw": [],
            "Q_vibe_raw": [],
            "Q_vibe_focus_raw": [],
        },
        stream_meta={},
        ema={"X": None},
    )
    return {"state": state}


def process_lsl_samples(
    cortex_state: Dict[str, object],
    samples: List[List[float]],
    lsl_timestamps: List[float],
    fs: float,
    ch_names: Optional[List[str]] = None,
) -> None:
    """Append incoming samples into rolling buffers (seconds-based)."""
    state: CortexState = cortex_state["state"]
    fs = float(fs) if fs else state.fs
    state.fs = fs
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n_ch = arr.shape[1] if arr.ndim == 2 else 0
    if n_ch < 4:
        # pad missing channels with zeros to keep shape
        pad_width = 4 - n_ch
        arr = np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")
    arr = arr[:, :4]  # take first 4 channels

    ts_arr = np.asarray(lsl_timestamps, dtype=float)
    if ts_arr.size != arr.shape[0]:
        # fallback: synthesize timestamps based on fs
        start = state.ts[-1] if state.ts else time.time()
        ts_arr = start + np.arange(arr.shape[0]) / fs

    for i in range(arr.shape[0]):
        for ch in range(4):
            state.buffers[ch].append(float(arr[i, ch]))
        state.ts.append(float(ts_arr[i]))
        state.n_total_samples += 1


def compute_step_if_ready(cortex_state: Dict[str, object], now_s: Optional[float] = None) -> Optional[Dict[str, object]]:
    """Compute metrics if enough new samples accumulated; return snapshot dict or None."""
    state: CortexState = cortex_state["state"]
    fs = state.fs
    now_s = now_s if now_s is not None else time.time()

    if state.n_total_samples - state.last_compute_samples < int(STEP_SEC * fs):
        return None
    if len(state.ts) < int(WINDOW_SEC * fs):
        return None

    window_n = int(WINDOW_SEC * fs)
    window_ts = list(state.ts)[-window_n:]
    window = np.vstack([np.array(b)[-window_n:] for b in state.buffers])

    metrics_version = METRICS_VERSION
    try:
        metrics = xq_metrics.compute_metrics_from_window(window, fs=fs)
    except Exception:
        metrics = _compute_metrics(window, fs=fs)
        metrics_version = CODE_VERSION
    state.last_compute_samples = state.n_total_samples

    ts_unix = now_s
    t_session = ts_unix - state.t0

    # Append history (seconds-based)
    state.history["t"].append(t_session)

    X_raw = float(metrics.get("X", metrics.get("R_abs", 0.0)))
    prev_ema = state.ema.get("X")
    X_ema = prev_ema
    if np.isfinite(X_raw):
        if prev_ema is None or not np.isfinite(prev_ema):
            X_ema = X_raw
        else:
            X_ema = float(EMA_ALPHA_X * X_raw + (1.0 - EMA_ALPHA_X) * prev_ema)
    state.ema["X"] = X_ema
    X_val = X_ema if (X_ema is not None and np.isfinite(X_ema)) else X_raw

    state.history["X_raw"].append(X_raw)
    state.history["X_ema"].append(X_ema)
    state.history["Q_abs_raw"].append(metrics.get("Q_abs_raw"))
    state.history["Q_vibe_raw"].append(metrics.get("Q_vibe_raw"))
    state.history["Q_vibe_focus_raw"].append(metrics.get("Q_vibe_focus_raw"))
    q_abs = float(metrics.get("Q_abs", 0.0))
    q_vibe = float(metrics.get("Q_vibe", 0.0))
    q_vf = float(metrics.get("Q_vibe_focus", 0.0))
    q_abs_raw = float(metrics.get("Q_abs_raw", 0.0))
    q_vibe_raw = float(metrics.get("Q_vibe_raw", 0.0))
    q_vf_raw = float(metrics.get("Q_vibe_focus_raw", 0.0))
    validity = bool(metrics.get("valid", False))
    quality_conf = float(metrics.get("quality_conf", 0.0))
    reason_codes = metrics.get("reason_codes", "")
    band_conf = float(metrics.get("band_conf", 0.0))
    ch_conf_raw_val = metrics.get("ch_conf_raw")
    if ch_conf_raw_val is None:
        ch_conf_raw_val = float("nan")
    ch_conf_used_val = metrics.get("ch_conf_used", ch_conf_raw_val)
    if ch_conf_used_val is None:
        ch_conf_used_val = ch_conf_raw_val
    blink_conf_val = metrics.get("blink_conf")
    blink_conf = float(1.0 if blink_conf_val is None else blink_conf_val)
    contact_conf_val = metrics.get("contact_conf")
    contact_conf = float(1.0 if contact_conf_val is None else contact_conf_val)
    drift_conf_val = metrics.get("drift_conf")
    drift_conf = float(1.0 if drift_conf_val is None else drift_conf_val)

    # HCE metric (masked) and diagnostic raw
    X_for_hce = X_ema if (X_ema is not None and np.isfinite(X_ema)) else X_raw
    q = max(0.0, q_vf)
    x_hce = float(X_for_hce)
    denom = max(x_hce, HCE_DENOM_FLOOR)
    hce = ((q / denom) ** 1.5) * q
    if not validity:
        hce = 0.0
    if not np.isfinite(hce):
        hce = 0.0

    q_r = max(0.0, q_vf_raw)
    x_r = max(float(X_raw), HCE_DENOM_FLOOR)
    hce_raw = ((q_r / x_r) ** 1.5) * q_r
    if not np.isfinite(hce_raw):
        hce_raw = 0.0

    # Scale for visibility in downstream charts/heatmaps.
    hce *= HCE_SCALE
    hce_raw *= HCE_SCALE

    state.history["HCE"].append(hce)
    state.history["HCE_raw"].append(hce_raw)

    snapshot = {
        "ts_unix": ts_unix,
        "t_session": t_session,
        "X": X_val,
        "X_raw": X_raw,
        "X_ema": X_ema,
        "HCE": hce,
        "HCE_raw": hce_raw,
        "Q_vibe_focus": q_vf,
        "Q_vibe": q_vibe,
        "Q_abs": q_abs,
        "Q_vibe_focus_raw": q_vf_raw,
        "Q_vibe_raw": q_vibe_raw,
        "Q_abs_raw": q_abs_raw,
        "reliability": {
            "valid": validity,
            "quality_conf": quality_conf,
            "reason_codes": reason_codes,
            "qualia_valid": bool(metrics.get("qualia_valid", validity)),
            "ch_conf_raw": float(ch_conf_raw_val),
            "band_conf": band_conf,
            "ch_conf_used": float(ch_conf_used_val),
            "blink_conf": blink_conf,
            "contact_conf": contact_conf,
            "drift_conf": drift_conf,
        },
        "raw": {
            "JS_dist": float(metrics.get("JS_dist", 0.0)),
            "phi": float(metrics.get("phi", 0.0)),
            "theta": float(metrics.get("theta", 0.0)),
            "alpha": float(metrics.get("alpha", 0.0)),
            "beta": float(metrics.get("beta", 0.0)),
            "gamma": float(metrics.get("gamma", 0.0)),
            "theta_n": float(metrics.get("theta_n", 0.0)),
            "alpha_n": float(metrics.get("alpha_n", 0.0)),
            "beta_n": float(metrics.get("beta_n", 0.0)),
            "gamma_n": float(metrics.get("gamma_n", 0.0)),
            "w_emg": float(metrics.get("w_emg", 0.0)),
            "ch_corr": float(metrics.get("ch_corr", 0.0)),
            "X_raw": X_raw,
            "X_ema": X_ema if (X_ema is not None and np.isfinite(X_ema)) else X_raw,
            "HCE_raw": hce_raw,
        },
        "meta": {
            "fs": fs,
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "baseline_version": BASELINE_VERSION,
            "baseline_warning": BASELINE_WARNING,
            "baseline_loaded": BASELINE_LOADED,
            "baseline_path_used": BASELINE_PATH_USED,
            "code_version": CODE_VERSION,
            "metrics_version": metrics_version,
            "gamma_ref": GAMMA_REF,
            "bg_ref": BG_REF,
        },
    }
    return snapshot


def build_state_packet(
    snapshot: Dict[str, object],
    participant_id: str = "",
    asset_id: str = "",
    cut_id: str = "",
) -> Dict[str, object]:
    """Assemble state packet compatible with Turrell v5.2 schema."""
    if snapshot is None:
        return {}
    packet = {
        "ts_unix": snapshot.get("ts_unix"),
        "t_session": snapshot.get("t_session"),
        "participant_id": participant_id,
        "asset_id": asset_id,
        "cut_id": cut_id,
        "X": snapshot.get("X"),
        "Q_abs": snapshot.get("Q_abs"),
        "Q_vibe": snapshot.get("Q_vibe"),
        "Q_vibe_focus": snapshot.get("Q_vibe_focus"),
        "Q_vibe_focus_E": snapshot.get("Q_vibe_focus_E", snapshot.get("Q_vibe_focus")),
        "HCE": snapshot.get("HCE"),
        "reliability": snapshot.get("reliability", {}),
        "raw": snapshot.get("raw", {}),
        "meta": snapshot.get("meta", {}),
    }
    # Preserve raw qualia fields for compatibility
    raw_section = packet["raw"]
    raw_section.update(
        {
            "Q_abs_raw": snapshot.get("Q_abs_raw"),
            "Q_vibe_raw": snapshot.get("Q_vibe_raw"),
            "Q_vibe_focus_raw": snapshot.get("Q_vibe_focus_raw"),
            "Q_vibe_focus_E_raw": snapshot.get("Q_vibe_focus_E_raw", snapshot.get("Q_vibe_focus_raw")),
            "X_raw": snapshot.get("X_raw"),
            "X_ema": snapshot.get("X_ema"),
            "HCE_raw": snapshot.get("HCE_raw", (snapshot.get("raw") or {}).get("HCE_raw")),
        }
    )
    return packet


def baseline_status() -> Dict[str, object]:
    """Return current baseline load status for diagnostics."""
    return {
        "baseline_loaded": BASELINE_LOADED,
        "baseline_path_used": BASELINE_PATH_USED,
        "baseline_warning": BASELINE_WARNING,
        "baseline_version": BASELINE_VERSION,
        "gamma_ref": GAMMA_REF,
        "bg_ref": BG_REF,
    }


__all__ = [
    "init_cortex_state",
    "load_baseline",
    "baseline_status",
    "process_lsl_samples",
    "compute_step_if_ready",
    "build_state_packet",
]

