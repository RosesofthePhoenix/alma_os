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

# Core constants (mirroring external script defaults)
FS_TARGET = 256
WINDOW_SEC = 5
STEP_SEC = 1
WINDOW_N = FS_TARGET * WINDOW_SEC
STEP_N = FS_TARGET * STEP_SEC
EPS = 1e-12

# Baseline fallbacks (used if no external baseline is loaded)
F_BASE = np.linspace(1, 45, 45, dtype=float)
P_REF = np.ones_like(F_BASE, dtype=float)
P_REF /= P_REF.sum()
MU_JS = 0.15
P90_JS = 0.35
TAU_PHI = (P90_JS - MU_JS) / np.log(2.0) if (P90_JS > MU_JS) else 0.10
BASELINE_VERSION = "embedded_fallback"
BASELINE_WARNING: Optional[str] = None
BASELINE_LOADED: bool = False
BASELINE_PATH_USED: Optional[str] = None
CODE_VERSION = "cortex_core"


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


def load_baseline(baseline_path: Optional[str] = None) -> None:
    """Load baseline with robust path resolution; record status."""
    global F_BASE, P_REF, MU_JS, P90_JS, TAU_PHI, BASELINE_VERSION, BASELINE_WARNING, BASELINE_LOADED, BASELINE_PATH_USED
    tried = []
    path_candidates: List[Path] = []
    if baseline_path:
        p = Path(baseline_path).expanduser()
        path_candidates.append(p if p.is_absolute() else config.ROOT_DIR / p)
        path_candidates.append(Path.cwd() / p)
    # default baseline
    default_path = Path(config.BASELINE_DEFAULT_PATH).expanduser()
    path_candidates.append(default_path if default_path.is_absolute() else config.ROOT_DIR / default_path)

    for p in path_candidates:
        tried.append(str(p))
        try:
            if p and p.exists():
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                f_base = np.array(data.get("f_bins_hz") or [], dtype=float)
                p_ref = np.array(data.get("P_ref") or [], dtype=float)
                if f_base.size == 0 or p_ref.size == 0:
                    continue
                f_base = np.maximum(f_base, 0.0)
                p_ref = np.maximum(p_ref, 0.0)
                p_ref = p_ref / (np.sum(p_ref) + EPS)
                mu_js = float(data.get("mu_JS_dist", MU_JS))
                p90_js = float(data.get("p90_JS_dist", P90_JS))
                tau_phi = (p90_js - mu_js) / np.log(2.0) if (p90_js > mu_js) else 0.10

                F_BASE = f_base
                P_REF = p_ref
                MU_JS = mu_js
                P90_JS = p90_js
                TAU_PHI = tau_phi
                BASELINE_VERSION = str(data.get("version", "unknown"))
                BASELINE_WARNING = None
                BASELINE_LOADED = True
                BASELINE_PATH_USED = str(p)
                return
        except Exception:
            continue

    # fallback
    BASELINE_WARNING = f"Using fallback baseline; tried: {tried}"
    BASELINE_LOADED = False
    BASELINE_PATH_USED = None


def init_cortex_state(fs: float = FS_TARGET, baseline_path: Optional[str] = None) -> Dict[str, object]:
    """Initialize cortex state (buffers, history, config)."""
    load_baseline(baseline_path)
    state = CortexState(
        buffers=[deque(maxlen=int(fs * 600)) for _ in range(4)],
        ts=deque(maxlen=int(fs * 600)),
        fs=float(fs),
        t0=time.time(),
        n_total_samples=0,
        last_compute_samples=0,
        history={
            "t": [],
            "Q_abs_raw": [],
            "Q_vibe_raw": [],
            "Q_vibe_focus_raw": [],
        },
        stream_meta={},
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

    metrics = _compute_metrics(window, fs=fs)
    state.last_compute_samples = state.n_total_samples

    ts_unix = now_s
    t_session = ts_unix - state.t0

    # Append history (seconds-based)
    state.history["t"].append(t_session)
    state.history["Q_abs_raw"].append(metrics["Q_abs_raw"])
    state.history["Q_vibe_raw"].append(metrics["Q_vibe_raw"])
    state.history["Q_vibe_focus_raw"].append(metrics["Q_vibe_focus_raw"])

    snapshot = {
        "ts_unix": ts_unix,
        "t_session": t_session,
        "X": float(metrics["R_abs"]),
        "Q_vibe_focus": float(metrics["Q_vibe_focus"]),
        "Q_vibe": float(metrics["Q_vibe"]),
        "Q_abs": float(metrics["Q_abs"]),
        "Q_vibe_focus_raw": float(metrics["Q_vibe_focus_raw"]),
        "Q_vibe_raw": float(metrics["Q_vibe_raw"]),
        "Q_abs_raw": float(metrics["Q_abs_raw"]),
        "reliability": {
            "valid": bool(metrics["valid"]),
            "quality_conf": float(metrics["quality_conf"]),
            "reason_codes": metrics["reason_codes"],
            "qualia_valid": bool(metrics["qualia_valid"]),
            "ch_conf_raw": float(metrics["ch_conf_raw"]),
            "band_conf": float(metrics["band_conf"]),
            "ch_conf_used": float(metrics["ch_conf_raw"]),
        },
        "raw": {
            "JS_dist": float(metrics["JS_dist"]),
            "phi": float(metrics["phi"]),
            "theta": float(metrics["theta"]),
            "alpha": float(metrics["alpha"]),
            "beta": float(metrics["beta"]),
            "gamma": float(metrics["gamma"]),
            "theta_n": float(metrics["theta_n"]),
            "alpha_n": float(metrics["alpha_n"]),
            "beta_n": float(metrics["beta_n"]),
            "gamma_n": float(metrics["gamma_n"]),
            "w_emg": float(metrics["w_emg"]),
            "ch_corr": float(metrics["ch_corr"]),
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
        }
    )
    return packet


__all__ = [
    "init_cortex_state",
    "load_baseline",
    "process_lsl_samples",
    "compute_step_if_ready",
    "build_state_packet",
]

