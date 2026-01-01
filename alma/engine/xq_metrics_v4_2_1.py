"""V4.2.1 metrics core (compute-only, no UI).

This is a trimmed, pure-Python port of the Cortex Console v4.2.1
metric stack. It preserves key behaviors required by ALMA OS:
- Robust baseline loading (no silent fallback).
- Channel agreement is diagnostic-only (never gates validity).
- Returns both raw and masked Q fields plus reliability hints.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import welch
from scipy.spatial.distance import jensenshannon

from alma import config

# Defaults / constants (mirroring v4.2.1 where relevant)
FS_TARGET = 256
WINDOW_SEC = 5
EPS = 1e-12
QUALITY_VALID_THR_DEFAULT = 0.25
BG_SIGMA = 0.20
CH_CONF_SOFT_FLOOR = 0.35
CH_CONF_HARD_MIN = 0.08

# Baseline globals
F_BASE: np.ndarray = np.linspace(1, 45, 45, dtype=float)
P_REF: np.ndarray = np.ones_like(F_BASE, dtype=float) / len(F_BASE)
MU_JS: float = 0.15
P90_JS: float = 0.35
TAU_PHI: float = (P90_JS - MU_JS) / np.log(2.0) if (P90_JS > MU_JS) else 0.10
GAMMA_REF: float = 0.25
BG_REF: float = 0.25
BASELINE_VERSION: str = "embedded_fallback"
BASELINE_WARNING: Optional[str] = None
BASELINE_LOADED: bool = False
BASELINE_PATH_USED: Optional[str] = None


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _band_mass(P_mean: np.ndarray, f_bins: np.ndarray, lo: float, hi: float) -> float:
    return float(P_mean[(f_bins >= lo) & (f_bins < hi)].sum())


def revised_solve_x(a: float, b: float, c: float = 1.0 / math.e) -> float:
    """Solve a + b*exp(-X) - c*X = 0 using Newton iterations (as in v4.x)."""
    x = max(0.0, a + b)  # seed
    for _ in range(32):
        fx = a + b * math.exp(-x) - c * x
        dfx = -b * math.exp(-x) - c
        if abs(dfx) < 1e-8:
            break
        x_new = x - fx / dfx
        x = max(0.0, x_new)
        if abs(fx) < 1e-6:
            break
    return max(0.0, x)


def load_baseline(baseline_path: Optional[str] = None) -> None:
    """Load baseline with deterministic resolution (absolute, ROOT/relative, CWD/relative)."""
    global F_BASE, P_REF, MU_JS, P90_JS, TAU_PHI, GAMMA_REF, BG_REF, BASELINE_VERSION, BASELINE_WARNING, BASELINE_LOADED, BASELINE_PATH_USED
    tried = []
    candidates = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        if str(p) in seen:
            return
        candidates.append(p)
        seen.add(str(p))

    if baseline_path:
        p_req = Path(baseline_path).expanduser()
        if p_req.is_absolute():
            _add(p_req)
        else:
            _add(config.ROOT_DIR / p_req)
            _add(Path.cwd() / p_req)
    default_p = Path(config.BASELINE_DEFAULT_PATH).expanduser()
    if default_p.is_absolute():
        _add(default_p)
    else:
        _add(config.ROOT_DIR / default_p)
        _add(Path.cwd() / default_p)

    for p in candidates:
        tried.append(str(p))
        try:
            if not p.exists():
                continue
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
            GAMMA_REF = float(data.get("gamma_ref", GAMMA_REF))
            BG_REF = float(data.get("bg_ref", BG_REF))
            BASELINE_VERSION = str(data.get("version", "unknown"))
            BASELINE_WARNING = None
            BASELINE_LOADED = True
            BASELINE_PATH_USED = str(p)
            return
        except Exception:
            continue

    BASELINE_WARNING = f"Using fallback baseline; tried: {tried}"
    BASELINE_LOADED = False
    BASELINE_PATH_USED = None


def _channel_conf(window_4ch: np.ndarray) -> Tuple[float, float]:
    """Return (ch_conf_raw, ch_conf_used) with soft floor; diagnostic-only."""
    try:
        corr = float(np.corrcoef(window_4ch)[np.triu_indices(4, 1)].mean())
        ch_conf_raw = float(0.5 + 0.5 * np.clip(corr, -1.0, 1.0))
    except Exception:
        ch_conf_raw = float("nan")
    ch_conf_used = float(np.clip(ch_conf_raw, CH_CONF_HARD_MIN, 1.0))
    ch_conf_used = max(ch_conf_used, CH_CONF_SOFT_FLOOR)
    return ch_conf_raw, ch_conf_used


def compute_metrics_from_window(window_4ch: np.ndarray, fs: float) -> Dict[str, object]:
    """Compute X/Q metrics and reliability (simplified v4.2.1 port)."""
    fs = float(fs) if fs else float(FS_TARGET)
    arr = np.asarray(window_4ch, dtype=float)
    if arr.shape[0] != 4 and arr.shape[1] == 4:
        arr = arr.T
    # PSD
    P_ch_list = []
    f_all = None
    for ch in arr:
        f, psd = welch(ch, fs=fs, nperseg=max(32, int(fs)))
        mask = (f >= 1) & (f <= 45)
        f_mask = f[mask]
        psd_mask = psd[mask]
        if f_all is None:
            f_all = f_mask
        P = psd_mask / (np.sum(psd_mask) + EPS)
        P_ch_list.append(P)
    P_ch = np.vstack(P_ch_list)
    P_mean = np.mean(P_ch, axis=0)
    P_mean = np.maximum(P_mean, 0.0)
    P_mean = P_mean / (np.sum(P_mean) + EPS)

    # Align to baseline
    P_i = np.interp(F_BASE, f_all, P_mean)
    P_i = np.maximum(P_i, 0.0)
    P_i = P_i / (np.sum(P_i) + EPS)

    JS_dist = float(jensenshannon(P_i, P_REF))
    d = max(0.0, JS_dist - MU_JS)
    phi = float(1.0 - math.exp(-d / (TAU_PHI + EPS)))
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

    band_conf = float(np.clip(total_bands, 0.0, 1.0))
    w_emg = float(1.0 - _sigmoid((gamma_n - 0.3) / 0.1))
    gamma_scaled = gamma_n * w_emg

    bg_raw = beta_n + gamma_scaled
    E = float(_sigmoid((bg_raw - BG_REF) / BG_SIGMA))
    SAC_raw = float(np.clip(1.0 - theta_n / (theta_n + alpha_n + EPS), 0.0, 1.0))
    k = float(np.log1p(beta_n))
    a = 0.5 * beta_n + 0.5 * (1.0 - alpha_n) - 0.1 * theta_n
    b = 0.5 * gamma_scaled
    X = float(revised_solve_x(a, b))

    # Provenance / reliability
    artifact_quality = float(np.clip(0.5 * band_conf + 0.5 * w_emg, 0.0, 1.0))
    quality_conf = float(np.clip(artifact_quality, 0.0, 1.0))
    valid = bool(quality_conf >= QUALITY_VALID_THR_DEFAULT)
    reason_codes = "" if valid else "low_quality_conf"

    # Qualia-style splits
    phi_meaningful = float(phi * artifact_quality)
    R_abs = float(k * phi * SAC_raw)
    R_meaningful = float(k * phi_meaningful * SAC_raw)
    R_focus_meaningful = float(R_meaningful * E)
    Q_abs_raw = float(X * R_abs)
    Q_vibe_raw = float(X * R_meaningful)
    Q_vibe_focus_raw = float(X * R_focus_meaningful)

    Q_abs = Q_abs_raw if valid else 0.0
    Q_vibe = Q_vibe_raw if valid else 0.0
    Q_vibe_focus = Q_vibe_focus_raw if valid else 0.0

    ch_conf_raw, ch_conf_used = _channel_conf(arr)
    blink_conf = 1.0
    contact_conf = 1.0
    drift_conf = 1.0

    return {
        "X": X,
        "Q_abs_raw": Q_abs_raw,
        "Q_vibe_raw": Q_vibe_raw,
        "Q_vibe_focus_raw": Q_vibe_focus_raw,
        "Q_abs": Q_abs,
        "Q_vibe": Q_vibe,
        "Q_vibe_focus": Q_vibe_focus,
        "valid": valid,
        "quality_conf": quality_conf,
        "reason_codes": reason_codes,
        "band_conf": band_conf,
        "blink_conf": blink_conf,
        "contact_conf": contact_conf,
        "drift_conf": drift_conf,
        "ch_conf_raw": ch_conf_raw,
        "ch_conf_used": ch_conf_used,
        "artifact_quality": artifact_quality,
        "phi": phi,
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "theta_n": theta_n,
        "alpha_n": alpha_n,
        "beta_n": beta_n,
        "gamma_n": gamma_n,
    }


def baseline_status() -> Dict[str, object]:
    return {
        "baseline_loaded": BASELINE_LOADED,
        "baseline_path_used": BASELINE_PATH_USED,
        "baseline_warning": BASELINE_WARNING,
        "baseline_version": BASELINE_VERSION,
        "gamma_ref": GAMMA_REF,
        "bg_ref": BG_REF,
    }


__all__ = [
    "compute_metrics_from_window",
    "load_baseline",
    "baseline_status",
    "revised_solve_x",
    "F_BASE",
    "P_REF",
    "MU_JS",
    "P90_JS",
    "TAU_PHI",
    "GAMMA_REF",
    "BG_REF",
    "BASELINE_VERSION",
]

