""" 
Cortex Console V4.2.1 (Dash) — Muse 2 (LSL) Absolute X/Q

V4.2 upgrades (high ROI)

12.1 Contact quality improvements
- Stronger horseshoe/contact stream discovery (stream scoring + auto-select best candidate)
- EEG-based contact inference fallback (long-run per-channel amplitude stability + imbalance proxies)
- Channel agreement soft-gating: ch_conf cannot single-handedly invalidate unless extremely low

12.2 Richness-only dashboard clarity
- Export + display both *raw (unmasked)* and *valid (masked)* Q metrics:
  - Q_abs_raw, Q_focus_raw, Q_perX_raw
  - Q_abs (masked), Q_focus (masked), Q_perX (masked)

12.3 Protect φ (phi) with artifact overlap
- Add artifact_quality (artifact-clean confidence excluding ch_conf)
- Add artifact_overlap = 1 - artifact_quality
- Add deviation provenance signals:
  - phi_meaningful = phi * artifact_quality
  - phi_artifact   = phi * artifact_overlap
- Add partner-friendly provenance summary in left sidebar (Option A)

Place in same folder as:
- baseline_global_muse_v1_revised.json

Run:
  pip install dash dash-bootstrap-components plotly numpy scipy pylsl pandas
  python app_v4_2_cortex_console.py

Open:
  http://127.0.0.1:8051
"""

import json
import os
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from pylsl import StreamInlet, resolve_byprop, resolve_streams
from scipy.optimize import fsolve
from scipy.signal import butter, sosfiltfilt, welch
from scipy.spatial.distance import jensenshannon

# ------------------------------
# Versioning / traceability
# ------------------------------
CODE_VERSION = "4.2.1"
SCRIPT_VARIANT = "v4.1"  # dashboard build id (separate from CODE_VERSION)
BASELINE_PATH = "baseline_global_muse_v1_revised.json"

# ------------------------------
# Runtime config
# ------------------------------
PORT = 8051
FS_TARGET = 256

WINDOW_SEC = 5
STEP_SEC = 1
WINDOW_N = FS_TARGET * WINDOW_SEC



# ------------------------------
# State stream (NDJSON)
# ------------------------------
NDJSON_PATH = "state_stream.ndjson"
NDJSON_SCHEMA_VERSION = "state_layer_api_v0.1"
STATE_STREAM_SCHEMA_VERSION = NDJSON_SCHEMA_VERSION
NDJSON_STREAM_PATH = NDJSON_PATH  # backward compat alias
EMIT_NDJSON_DEFAULT = False

# Adaptive controller mode file
MODE_FILE_PATH = "spotify_proto/adaptive_mode.json"

# EMA smoothing (DISPLAY ONLY)
EMA_ALPHA_X = 0.18
EMA_ALPHA_Q = 0.18
EMA_ALPHA_DRV = 0.22

# Default validity threshold (user can tune in UI)
QUALITY_VALID_THR_DEFAULT = 0.25

# EMG gate (tune if needed)
EMG_THR = 1.30
EMG_SCALE = 0.50
EMG_REASON_THR = 0.45  # w_emg below this => high_emg

# Band mass confidence (theta+alpha+beta+gamma within normalized PSD)
BAND_MASS_MIN = 0.15
BAND_MASS_GOOD = 0.55

# Engagement sigmoid width (affects ONLY Q_focus; not validity)
BG_SIGMA = 0.20

# Blink proxy thresholds (heuristic)
BLINK_Z_THR = 6.0            # robust-z threshold for spike
BLINK_RATE_THR = 0.02        # fraction of samples above threshold considered "starting to blink"
BLINK_RATE_SCALE = 0.08      # how fast blink_conf collapses

# Drift proxy thresholds: compare 1–4 Hz mass to baseline
DRIFT_BAND = (1, 4)          # 1 <= f < 4
DRIFT_OVER_THR = 0.06        # above baseline drift mass (absolute fraction) to start penalty
DRIFT_OVER_SCALE = 0.22


# Drift aggressiveness multiplier (UI). 1.0 = default; <1 looser; >1 stricter.
DRIFT_AGGR_DEFAULT = 1.0

# Warmup / stabilization (recommended for clean Session-2 validation)
# During early stream settling, drift proxies can be overly punitive.
# We temporarily ignore drift penalty for the first N seconds to avoid false invalidation.
WARMUP_IGNORE_DRIFT_SEC = 90



# V4.2.1 reliability v2 tuning
VALIDATION_MODE_DEFAULT = "RELAXED"  # RELAXED | ADAPTIVE | STRICT
PSD_HP_HZ_DEFAULT = 1.0  # used for drift/band gating only (does not change JS/phi)
ARTIFACT_QUALITY_UNCERTAIN = 0.75
DRIFT_REF_UPDATE_MIN_CLEAN = 0.70
# Session-adaptive drift reference (EMA), to reduce false "high_drift" if a user's baseline 1–4Hz mass differs
DRIFT_REF_EMA_ALPHA = 0.02
DRIFT_REF_EMA_ALPHA_WARMUP = 0.06
DRIFT_REF_UPDATE_MIN_ARTIFACT_QUALITY = 0.70

# Horseshoe/contact mapping (if stream exists)
# Typical Muse horseshoe values: often 1..4 (lower = better). We map to confidence.
HORSESHOE_GOOD = 1.0
HORSESHOE_BAD = 4.0

# 12.1: Relax ch_conf from invalidating everything
CH_CONF_SOFT_FLOOR = 0.35
CH_CONF_HARD_MIN = 0.08

# 12.1: Contact inference (fallback when no horseshoe stream)
CONTACT_HIST_N = 240            # ~4 minutes at 1 Hz updates
CONTACT_MIN_HIST = 25           # warm-up before trusting z-scores
CONTACT_RATIO_THR = 1.8         # max(rms)/min(rms)
CONTACT_RATIO_SCALE = 0.45
CONTACT_Z_THR = 4.0             # robust z (per-channel)
CONTACT_Z_SCALE = 1.2
CONTACT_FLAT_REL_THR = 0.18     # std too small vs long-run median std

# 12.3: Deviation provenance thresholds
PHI_HIGH_THR = 0.65
ARTIFACT_QUALITY_LOW = 0.55

# Qualia options (A/B/C/D) masking:
# We want “meaningful qualia” to remain meaningful even when overall validity is RELAXED.
# So we keep the global validity gate (quality_conf) for the main Q metrics, but we mask
# the qualia-only options using artifact_quality (artifact-clean confidence).
#
# Practical default:
# - ARTIFACT_QUALITY_LOW (~0.55) corresponds to “artifact-likely” in provenance labeling.
# - Mask qualia options when artifact_quality drops below this threshold.
QUALIA_VALID_ARTIFACT_QUALITY_THR_DEFAULT = ARTIFACT_QUALITY_LOW

EPS = 1e-12

# ------------------------------
# Baseline load
# ------------------------------
if not os.path.exists(BASELINE_PATH):
    raise FileNotFoundError(
        f"Missing {BASELINE_PATH}. Put it in the same folder as this script."
    )

with open(BASELINE_PATH, "r") as f:
    baseline = json.load(f)

P_REF = np.array(baseline["P_ref"], dtype=float)
F_BASE = np.array(baseline["f_bins_hz"], dtype=float)
MU_JS = float(baseline["mu_JS_dist"])
P90_JS = float(baseline["p90_JS_dist"])
GAMMA_REF = float(baseline["gamma_ref"])
BG_REF = float(baseline.get("bg_ref", 0.25))
BASELINE_VERSION = str(baseline.get("version", "unknown"))

TAU_PHI = (P90_JS - MU_JS) / np.log(2.0) if (P90_JS > MU_JS) else 0.10

# Drift reference mass (baseline) in 1–4 Hz
DRIFT_REF = float(P_REF[(F_BASE >= DRIFT_BAND[0]) & (F_BASE < DRIFT_BAND[1])].sum())

# ------------------------------
# Option E (permutation entropy + aperiodic slope) config
# ------------------------------
option_e_cfg = baseline.get("option_e", {}) if isinstance(baseline.get("option_e", {}), dict) else {}
option_e_defaults = option_e_cfg.get("defaults", {}) if isinstance(option_e_cfg.get("defaults", {}), dict) else {}


def _opt_e_default(key, fallback):
    try:
        return option_e_defaults.get(key, fallback)
    except Exception:
        return fallback


OPTION_E_WC = float(_opt_e_default("wC", 0.15))
OPTION_E_WS = float(_opt_e_default("wS", 0.12))
OPTION_E_MULT_CLIP = _opt_e_default("mult_clip", [0.70, 1.35])
if not (isinstance(OPTION_E_MULT_CLIP, (list, tuple)) and len(OPTION_E_MULT_CLIP) == 2):
    OPTION_E_MULT_CLIP = [0.70, 1.35]

OPTION_E_PE_M = int(_opt_e_default("pe_m", 5))
OPTION_E_PE_BAND = _opt_e_default("pe_band_hz", [8, 30])
if not (isinstance(OPTION_E_PE_BAND, (list, tuple)) and len(OPTION_E_PE_BAND) == 2):
    OPTION_E_PE_BAND = [8, 30]
OPTION_E_PE_BAND = [float(OPTION_E_PE_BAND[0]), float(OPTION_E_PE_BAND[1])]

OPTION_E_SLOPE_FIT_HZ = _opt_e_default("slope_fit_hz", [2, 45])
if not (isinstance(OPTION_E_SLOPE_FIT_HZ, (list, tuple)) and len(OPTION_E_SLOPE_FIT_HZ) == 2):
    OPTION_E_SLOPE_FIT_HZ = [2, 45]
OPTION_E_SLOPE_FIT_HZ = [float(OPTION_E_SLOPE_FIT_HZ[0]), float(OPTION_E_SLOPE_FIT_HZ[1])]

OPTION_E_SLOPE_EXCLUDE_HZ = _opt_e_default("slope_exclude_hz", [[8, 13]])
if not isinstance(OPTION_E_SLOPE_EXCLUDE_HZ, (list, tuple)):
    OPTION_E_SLOPE_EXCLUDE_HZ = [[8, 13]]
OPTION_E_SLOPE_EXCLUDE_HZ = [
    [float(a), float(b)] for (a, b) in OPTION_E_SLOPE_EXCLUDE_HZ if isinstance(a, (int, float)) and isinstance(b, (int, float))
]

OPTION_E_PROFILES = option_e_cfg.get("profiles", {}) if isinstance(option_e_cfg.get("profiles", {}), dict) else {}
OPTION_E_PROFILE_KEYS = sorted(list(OPTION_E_PROFILES.keys())) if OPTION_E_PROFILES else ["global"]
ENV_PROFILE_ID = os.getenv("CORTEX_PROFILE_ID", "global")
OPTION_E_PROFILE_DEFAULT = ENV_PROFILE_ID if ENV_PROFILE_ID in OPTION_E_PROFILE_KEYS else (OPTION_E_PROFILE_KEYS[0] if OPTION_E_PROFILE_KEYS else "global")

env_pe_tau = os.getenv("CORTEX_PE_TAU", None)
try:
    env_pe_tau = int(env_pe_tau) if env_pe_tau is not None else None
except Exception:
    env_pe_tau = None
OPTION_E_PE_TAU_DEFAULT = env_pe_tau if env_pe_tau is not None else int(_opt_e_default("pe_tau", 1))
OPTION_E_PE_TAU_DEFAULT = int(max(1, min(10, OPTION_E_PE_TAU_DEFAULT)))

# Global (top-level) stats fallbacks
OPTION_E_STATS_TOP = {
    "C_pe": option_e_cfg.get("C_pe", {}) if isinstance(option_e_cfg.get("C_pe", {}), dict) else {},
    "S_flat": option_e_cfg.get("S_flat", {}) if isinstance(option_e_cfg.get("S_flat", {}), dict) else {},
}

# Precompute Option E bandpass filter
try:
    OPTION_E_SOS = butter(4, OPTION_E_PE_BAND, btype="bandpass", fs=FS_TARGET, output="sos")
except Exception:
    OPTION_E_SOS = None

# ------------------------------
# Helpers
# ------------------------------
def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def revised_solve_x(a, b, c=1 / np.e, initial_guess=1.0) -> float:
    """Preserved X solver style."""

    def equation(x):
        return a + b * np.exp(-x) - c * x

    try:
        sol = fsolve(equation, initial_guess)[0]
        return float(max(0.0, sol))
    except Exception:
        return float("nan")


def get_channel_labels(info) -> List[str]:
    """Attempt to read channel labels from LSL stream info."""
    labels = []
    try:
        ch = info.desc().child("channels").child("channel")
        while ch.name() == "channel":
            lab = ch.child_value("label")
            labels.append(lab)
            ch = ch.next_sibling()
    except Exception:
        pass
    return labels


def pick_muse_eeg_indices(labels: List[str], n_ch: int) -> Tuple[List[int], List[str]]:
    """Pick 4 EEG channels, ignoring AUX. Uses common Muse labels if available."""
    if labels:
        wanted = ["TP9", "AF7", "AF8", "TP10"]
        idx = []
        out_labels = []
        for w in wanted:
            if w in labels:
                i = labels.index(w)
                idx.append(i)
                out_labels.append(w)
        if len(idx) == 4:
            return idx, out_labels

        # else take first 4 non-AUX-ish
        idx = []
        out_labels = []
        for i, lab in enumerate(labels[:n_ch]):
            if lab and "AUX" in lab.upper():
                continue
            idx.append(i)
            out_labels.append(lab or f"ch{i}")
            if len(idx) == 4:
                break
        if len(idx) == 4:
            return idx, out_labels

    idx = list(range(min(4, n_ch)))
    out_labels = [f"ch{i}" for i in idx]
    return idx, out_labels


def pick_frontal_indices(ch_labels: List[str]) -> List[int]:
    """Try to find AF7/AF8 indices for blink proxy."""
    idx = []
    for lab in ["AF7", "AF8"]:
        if lab in ch_labels:
            idx.append(ch_labels.index(lab))
    # If missing, assume middle channels (Muse typical ordering: TP9, AF7, AF8, TP10)
    if not idx and len(ch_labels) >= 4:
        idx = [1, 2]
    return idx


def robust_blink_rate(x: np.ndarray) -> float:
    """Blink/eye-movement proxy: fraction of time-domain outliers (robust z-score)."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + EPS
    z = np.abs(x - med) / (1.4826 * mad + EPS)
    return float(np.mean(z > BLINK_Z_THR))


def horseshoe_to_conf(vals: np.ndarray) -> float:
    """Map Muse horseshoe/contact values to [0..1]."""
    v = np.array(vals, dtype=float).flatten()
    if v.size == 0:
        return 1.0
    # If values look like 0..1 already, treat as confidence directly
    if np.nanmin(v) >= 0.0 and np.nanmax(v) <= 1.0:
        return float(np.nanmean(v))
    # Otherwise assume 1..4 (lower better)
    v = np.clip(v, HORSESHOE_GOOD, HORSESHOE_BAD)
    conf = 1.0 - (np.nanmean(v) - HORSESHOE_GOOD) / (HORSESHOE_BAD - HORSESHOE_GOOD + EPS)
    return float(np.clip(conf, 0.0, 1.0))


def safe_interp_to_baseline(f_src: np.ndarray, p_src: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Interpolate p_src from f_src onto baseline bins."""
    rebinned = False
    if (len(p_src) != len(P_REF)) or (not np.allclose(f_src, F_BASE)):
        rebinned = True
        p_i = np.interp(F_BASE, f_src, p_src)
        p_i = np.maximum(p_i, 0.0)
        p_i = p_i / (np.sum(p_i) + EPS)
        return p_i, rebinned
    return p_src, rebinned


def robust_mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med))) + EPS


def _option_e_mu_sigma(container: dict, key: str):
    if not isinstance(container, dict):
        return None, None
    v = container.get(key)
    if not isinstance(v, dict):
        return None, None
    mu = v.get("mu", None)
    sigma = v.get("sigma", None)
    try:
        mu = float(mu)
        sigma = float(sigma)
    except Exception:
        return None, None
    if not np.isfinite(mu) or not np.isfinite(sigma):
        return None, None
    return mu, sigma


def option_e_pick_stats(profile_id: str):
    """Resolve Option E stats with fallback order."""
    prof_id = (profile_id or "global").strip()
    candidates = [
        (OPTION_E_PROFILES.get(prof_id, {}) if isinstance(OPTION_E_PROFILES, dict) else {}, f"profile:{prof_id}"),
        (OPTION_E_PROFILES.get("global", {}) if isinstance(OPTION_E_PROFILES, dict) else {}, "profile:global"),
        (OPTION_E_STATS_TOP, "top"),
    ]

    def pick(key: str):
        for container, label in candidates:
            mu, sigma = _option_e_mu_sigma(container, key)
            if mu is not None and sigma is not None and sigma > 0:
                return mu, sigma, label
        return None, None, "none"

    mu_c, sig_c, src_c = pick("C_pe")
    mu_s, sig_s, src_s = pick("S_flat")
    stats_src = src_c if src_c != "none" else src_s
    if src_c != "none" and src_s != "none" and src_c != src_s:
        stats_src = f"{src_c}+{src_s}"
    if stats_src == "none":
        stats_src = "none"
    return {
        "C_pe": (mu_c, sig_c, src_c),
        "S_flat": (mu_s, sig_s, src_s),
        "stats_src": stats_src,
    }


def permutation_entropy_bandpassed(window_4ch: np.ndarray, sos, m: int, tau: int) -> float:
    """Permutation entropy averaged across channels after bandpass + z-score."""
    if sos is None or m < 2 or tau < 1:
        return float("nan")
    vals = []
    for ch in window_4ch:
        try:
            y = sosfiltfilt(sos, ch)
        except Exception:
            continue
        y = np.asarray(y, dtype=float)
        if y.size < (m * tau):
            continue
        mu = float(np.mean(y))
        sd = float(np.std(y) + EPS)
        if sd <= EPS:
            continue
        z = (y - mu) / sd
        n = len(z) - (m - 1) * tau
        if n <= 0:
            continue
        eps_tie = 1e-12
        counts = {}
        for i in range(n):
            segment = z[i : i + m * tau : tau]
            if segment.size != m:
                continue
            order = np.argsort(segment + eps_tie * np.arange(m))
            key = tuple(order.tolist())
            counts[key] = counts.get(key, 0) + 1
        total = sum(counts.values())
        if total <= 0:
            continue
        p = np.array(list(counts.values()), dtype=float) / float(total)
        H = float(-np.sum(p * np.log(p + EPS)))
        Hmax = float(math.log(math.factorial(m)))
        if Hmax <= 0:
            continue
        vals.append(float(np.clip(H / Hmax, 0.0, 1.0)))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def fit_aperiodic_slope(f_bins: np.ndarray, p_raw: np.ndarray, fit_range, exclude_ranges) -> float:
    """Fit log–log slope of unnormalized PSD."""
    try:
        f_bins = np.asarray(f_bins, dtype=float)
        p_raw = np.asarray(p_raw, dtype=float)
    except Exception:
        return float("nan")
    if f_bins.shape != p_raw.shape or f_bins.size < 8:
        return float("nan")
    mask = (f_bins >= float(fit_range[0])) & (f_bins <= float(fit_range[1]))
    for lo, hi in exclude_ranges:
        mask &= ~((f_bins >= float(lo)) & (f_bins <= float(hi)))
    f_sel = f_bins[mask]
    p_sel = p_raw[mask]
    if f_sel.size < 3:
        return float("nan")
    pos_mask = p_sel > 0
    f_sel = f_sel[pos_mask]
    p_sel = p_sel[pos_mask]
    if f_sel.size < 3:
        return float("nan")
    logf = np.log10(f_sel)
    logp = np.log10(p_sel + EPS)
    try:
        slope, _ = np.polyfit(logf, logp, 1)
        return float(slope)
    except Exception:
        return float("nan")

def infer_contact_conf(window_4ch: np.ndarray) -> Tuple[float, str, str]:
    """
    EEG-based contact confidence inference.

    Uses long-run per-channel RMS and STD stability + inter-channel imbalance.
    Returns (contact_conf, contact_raw, method).

    Notes:
    - This does NOT try to be a true impedance measure; it is a pragmatic proxy.
    - It is designed to detect: one-sensor slip, saturation/dropout, long-run channel amplitude anomalies.
    """
    ci = global_state["contact_infer"]

    # Per-channel RMS + STD (time-domain)
    rms = np.sqrt(np.mean(np.square(window_4ch), axis=1) + EPS)
    std = np.std(window_4ch, axis=1) + EPS

    # Update histories
    for k in range(4):
        ci["rms_hist"][k].append(float(rms[k]))
        ci["std_hist"][k].append(float(std[k]))
    ratio = float(np.max(rms) / (np.min(rms) + EPS))
    ci["ratio_hist"].append(ratio)

    # Warm-up: if insufficient history, be conservative (near-neutral)
    hist_len = len(ci["ratio_hist"])
    if hist_len < CONTACT_MIN_HIST:
        raw = f"infer:warmup n={hist_len} ratio={ratio:.2f}"
        return 1.0, raw, "inferred"

    # Robust per-channel z vs each channel's history
    z_list = []
    for k in range(4):
        h = np.array(ci["rms_hist"][k], dtype=float)
        med = float(np.median(h))
        mad = robust_mad(h)
        z = float(abs(rms[k] - med) / (1.4826 * mad + EPS))
        z_list.append(z)
    zmax = float(np.max(z_list))

    # Flat / dropout proxy: std too small relative to that channel's long-run median std
    flat_flags = []
    for k in range(4):
        hs = np.array(ci["std_hist"][k], dtype=float)
        med_std = float(np.median(hs)) + EPS
        flat_flags.append(bool(std[k] < CONTACT_FLAT_REL_THR * med_std))
    flat_any = any(flat_flags)

    # Turn features into penalties
    ratio_pen = float(sigmoid((ratio - CONTACT_RATIO_THR) / (CONTACT_RATIO_SCALE + EPS)))
    z_pen = float(sigmoid((zmax - CONTACT_Z_THR) / (CONTACT_Z_SCALE + EPS)))
    flat_pen = 1.0 if flat_any else 0.0

    # Weighted combination (cap at 1)
    penalty = float(np.clip(0.55 * ratio_pen + 0.30 * z_pen + 0.65 * flat_pen, 0.0, 1.0))
    conf = float(np.clip(1.0 - penalty, 0.0, 1.0))

    raw = (
        f"infer:ratio={ratio:.2f} ratio_pen={ratio_pen:.2f} | "
        f"zmax={zmax:.2f} z_pen={z_pen:.2f} | "
        f"flat={int(flat_any)} conf={conf:.2f}"
    )
    return conf, raw, "inferred"


# ------------------------------
# Metrics
# ------------------------------
@dataclass
class Metrics:
    X: float
    R_abs: float

    # meaningful richness (qualia-ish proxies; masked valid-only)
    R_meaningful: float
    R_focus_meaningful: float
    Q_vibe: float

    # Option D: engaged + meaningful + intensity (masked valid-only)
    Q_vibe_focus: float

    # Option E: qualia focus with entropy/slope multiplier
    Q_vibe_focus_E: float
    Q_vibe_focus_E_raw: float
    Q_vibe_focus_E_mult: float

    # meaningful richness (raw/unmasked)
    R_meaningful_raw: float
    R_focus_meaningful_raw: float
    Q_vibe_raw: float

    # Option D (raw/unmasked)
    Q_vibe_focus_raw: float

    # masked (valid only)
    Q_abs: float
    Q_perX: float
    R_focus: float
    Q_focus: float

    # raw (unmasked)
    Q_abs_raw: float
    Q_perX_raw: float
    Q_focus_raw: float

    JS_dist: float
    phi: float

    # validity
    quality_conf: float
    valid: bool
    qualia_valid: bool
    reason_codes: str

    # 12.3 provenance
    artifact_quality: float
    artifact_overlap: float
    phi_meaningful: float
    phi_artifact: float
    provenance: str

    # diagnostics/drivers
    theta: float
    alpha: float
    beta: float
    gamma: float
    total_bands: float
    theta_n: float
    alpha_n: float
    beta_n: float
    gamma_n: float
    emg_index: float
    w_emg: float
    bg_raw: float
    E: float
    gamma_scaled: float
    a: float
    b: float
    k: float
    SAC_raw: float
    SAC_focus: float
    ch_corr: float
    ch_conf_raw: float
    ch_conf_used: float
    band_conf: float

    # artifact proxies
    drift_mass: float
    drift_over: float
    drift_conf: float
    blink_rate: float
    blink_conf: float

    # contact
    contact_conf: float
    contact_raw: str
    contact_method: str
    contact_conf_used: float

    # Option E drivers
    C_pe: float
    C_pe_z: float
    C_pe_n: float
    S_aperiodic_slope: float
    S_flat: float
    S_aperiodic_slope_z: float
    S_aperiodic_slope_n: float
    option_e_profile_id: str
    option_e_stats_src: str
    option_e_pe_tau: float



# ------------------------------
# Reliability v2 helpers
# ------------------------------
def gate_psd(P: np.ndarray, f_bins: np.ndarray, hp_hz: float) -> np.ndarray:
    """Zero out bins < hp_hz and renormalize. Used for drift/band gating only."""
    hp = float(np.clip(hp_hz, 0.0, 2.0))
    Pg = np.array(P, dtype=float).copy()
    Pg[f_bins < hp] = 0.0
    s = float(np.sum(Pg))
    if s > 0:
        Pg /= (s + EPS)
    return Pg

def baseline_drift_ref_for_hp(hp_hz: float) -> float:
    """Baseline drift mass (1–4Hz) computed in the gated PSD space."""
    Pg = gate_psd(P_REF, F_BASE, hp_hz)
    return float(Pg[(F_BASE >= DRIFT_BAND[0]) & (F_BASE < DRIFT_BAND[1])].sum())


def format_reliability(history: dict, window_n: int) -> str:
    """Human-readable rolling reliability summary."""
    try:
        N = int(window_n)
    except Exception:
        N = 600
    N = max(20, min(N, 5000))
    t = history.get("t", [])
    if not t:
        return "No samples yet."

    q = history.get("quality_conf", [])
    valid = history.get("valid", [])
    reasons = history.get("reason_codes", [])

    n = min(N, len(valid))
    q_slice = [float(x) for x in q[-n:] if x is not None and np.isfinite(x)]
    v_slice = [bool(x) for x in valid[-n:]]

    if not q_slice:
        return f"Last {n} samples:\n(no quality_conf)"

    # percentiles
    q_sorted = sorted(q_slice)
    def pct(p: float) -> float:
        if not q_sorted:
            return float("nan")
        k = int(round((p/100.0) * (len(q_sorted)-1)))
        k = max(0, min(k, len(q_sorted)-1))
        return float(q_sorted[k])

    valid_count = int(sum(1 for x in v_slice if x))
    valid_rate = 100.0 * valid_count / float(len(v_slice) + EPS)

    # invalid reason counts (only among invalid samples)
    counts = {}
    for is_valid, rc in zip(v_slice, reasons[-n:]):
        if is_valid:
            continue
        if not rc:
            continue
        parts = [p.strip() for p in str(rc).split(";") if p.strip()]
        for p in parts:
            counts[p] = counts.get(p, 0) + 1

    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    top_str = ", ".join([f"{k}:{v}" for k, v in top]) if top else "(none)"

    return (
        f"Last {n} samples:\n"
        f"Valid rate: {valid_rate:.1f}% ({valid_count}/{len(v_slice)})\n"
        f"quality_conf: min {min(q_slice):.3f}, median {pct(50):.3f}, p10 {pct(10):.3f}, p90 {pct(90):.3f}\n"
        f"Top invalid reasons: {top_str}"
    )


def compute_metrics_from_window(
    window_4ch: np.ndarray,
    ch_labels_4: List[str],
    quality_thr: float,
    validation_mode: str,
    psd_hp_hz: float,
    contact_conf: float,
    contact_raw: str,
    contact_method: str,
    elapsed_s: float,
    option_e_profile_id: str,
    option_e_pe_tau: int,
) -> Metrics:
    """Compute X/Q from a 5s window of 4 channels. window_4ch shape: (4, WINDOW_N)."""

    # Welch per channel -> normalized PSD shape
    P_ch_list = []
    P_ch_raw_list = []
    f_all = None
    for ch_data in window_4ch:
        f, psd = welch(ch_data, fs=FS_TARGET, nperseg=FS_TARGET)
        mask = (f >= 1) & (f <= 45)
        f_mask = f[mask]
        psd_mask = psd[mask]
        if f_all is None:
            f_all = f_mask
        P_ch_raw_list.append(psd_mask)
        P = psd_mask / (np.sum(psd_mask) + EPS)
        P_ch_list.append(P)

    P_ch = np.vstack(P_ch_list)
    P_ch_raw = np.vstack(P_ch_raw_list)
    P_mean_raw = np.mean(P_ch_raw, axis=0)
    P_mean = np.mean(P_ch, axis=0)
    P_mean = np.maximum(P_mean, 0.0)
    P_mean = P_mean / (np.sum(P_mean) + EPS)

    # Align bins to baseline if necessary
    P_mean, rebinned = safe_interp_to_baseline(f_all, P_mean)

    # JS distance and phi
    JS_dist = float(jensenshannon(P_mean, P_REF))
    d = max(0.0, JS_dist - MU_JS)
    phi = float(1.0 - np.exp(-d / (TAU_PHI + EPS)))
    phi = float(np.clip(phi, 0.0, 1.0))

    # Bands (fractions of 1–45 mass)
    f_bins = F_BASE
    theta = float(P_mean[(f_bins >= 4) & (f_bins < 8)].sum())
    alpha = float(P_mean[(f_bins >= 8) & (f_bins < 13)].sum())
    beta = float(P_mean[(f_bins >= 13) & (f_bins < 30)].sum())
    gamma = float(P_mean[(f_bins >= 30) & (f_bins <= 45)].sum())
    total_bands = theta + alpha + beta + gamma

    # Band-normalized
    if total_bands > 0:
        theta_n = theta / (total_bands + EPS)
        alpha_n = alpha / (total_bands + EPS)
        beta_n = beta / (total_bands + EPS)
        gamma_n = gamma / (total_bands + EPS)
    else:
        theta_n = alpha_n = beta_n = gamma_n = 0.0

    # EMG index and suppression
    emg_num = float(P_mean[(f_bins >= 35) & (f_bins <= 45)].sum())
    emg_den = float(P_mean[(f_bins >= 15) & (f_bins < 30)].sum()) + EPS
    emg_index = float(emg_num / emg_den)

    w_emg = float(1.0 - sigmoid((emg_index - EMG_THR) / (EMG_SCALE + EPS)))
    w_emg = float(np.clip(w_emg, 0.0, 1.0))

    gamma_scaled = float((gamma / (GAMMA_REF + EPS)) * w_emg)

    # Engagement (only for focus variant)
    bg_raw = float(beta + gamma_scaled)
    E = float(sigmoid((bg_raw - BG_REF) / (BG_SIGMA + EPS)))

    # X params
    a = float(0.5 * beta_n + 0.5 * (1.0 - alpha_n) - 0.1 * theta_n)
    b = float(0.5 * gamma_scaled)
    X = revised_solve_x(a, b)

    # Q decomposition
    k = float(np.log1p(beta_n))
    SAC_raw = float(1.0 - (theta_n / (theta_n + alpha_n + EPS)))
    SAC_raw = float(np.clip(SAC_raw, 0.0, 1.0))

    R_abs = float(k * phi * SAC_raw)
    Q_abs_raw = float(R_abs * X)

    SAC_focus = float(SAC_raw * E)
    R_focus = float(k * phi * SAC_focus)
    Q_focus_raw = float(R_focus * X)

    Q_perX_raw = float(Q_abs_raw / (X + EPS))

    # Channel agreement (correlation on log PSD shapes)
    logP = np.log(P_ch + EPS)
    corrs = []
    for i in range(logP.shape[0]):
        for j in range(i + 1, logP.shape[0]):
            ci = np.corrcoef(logP[i], logP[j])[0, 1]
            if np.isfinite(ci):
                corrs.append(ci)
    ch_corr = float(np.mean(corrs)) if corrs else 0.0
    ch_conf_raw = float(np.clip((ch_corr - 0.3) / (0.9 - 0.3 + EPS), 0.0, 1.0))

    # 12.1: ch_conf soft gating
    if ch_conf_raw < CH_CONF_HARD_MIN:
        ch_conf_used = ch_conf_raw
    else:
        ch_conf_used = float(max(ch_conf_raw, CH_CONF_SOFT_FLOOR))


    # Gated PSD for drift/band proxies (does not affect JS/phi)
    P_gate = gate_psd(P_mean, f_bins, psd_hp_hz)
    total_bands_gate = float(P_gate[(f_bins >= 4) & (f_bins <= 45)].sum())

    # Band mass confidence (computed on gated PSD to reduce false negatives from slow drift)
    band_conf = float(
        np.clip(
            (total_bands_gate - BAND_MASS_MIN) / (BAND_MASS_GOOD - BAND_MASS_MIN + EPS),
            0.0,
            1.0,
        )
    )

    # Drift proxy (1–4 Hz mass) from gated PSD
    drift_mass = float(P_gate[(f_bins >= DRIFT_BAND[0]) & (f_bins < DRIFT_BAND[1])].sum())
    drift_ref_dyn = float(global_state.get("drift_ref_ema", baseline_drift_ref_for_hp(psd_hp_hz)))
    drift_over = float(max(0.0, drift_mass - drift_ref_dyn))
    # Drift tuning (UI)
    drift_aggr = float(global_state.get("drift_aggr", DRIFT_AGGR_DEFAULT))
    drift_aggr = float(np.clip(drift_aggr, 0.25, 4.0))
    eff_drift_thr = float(DRIFT_OVER_THR / (drift_aggr + EPS))
    eff_drift_scale = float(DRIFT_OVER_SCALE / (drift_aggr + EPS))

    drift_conf = float(1.0 - np.clip((drift_over - eff_drift_thr) / (eff_drift_scale + EPS), 0.0, 1.0))

    # Warmup: ignore drift penalty early to avoid false invalidation during stream settling
    warmup_ignore = float(global_state.get("warmup_ignore_drift_sec", WARMUP_IGNORE_DRIFT_SEC))
    if float(elapsed_s) < warmup_ignore:
        drift_over = 0.0
        drift_conf = 1.0

    # Blink proxy (time-domain) using frontal channels if possible
    frontal_idx = pick_frontal_indices(ch_labels_4)
    blink_rates = [robust_blink_rate(window_4ch[i]) for i in frontal_idx if 0 <= i < 4]
    blink_rate = float(max(blink_rates)) if blink_rates else 0.0
    blink_conf = float(1.0 - np.clip((blink_rate - BLINK_RATE_THR) / (BLINK_RATE_SCALE + EPS), 0.0, 1.0))

    # Contact confidence soft gating: preserve real horseshoe; soften inferred contact
    if contact_method == "horseshoe":
        contact_conf_used = float(contact_conf)
    else:
        contact_conf_used = float(np.clip(0.60 + 0.40 * float(contact_conf), 0.0, 1.0))

    # 12.3: artifact quality excludes channel agreement

    # Soft terms (reduce false negatives from band/drift on Muse-class headsets)
    band_term_soft = float(0.6 + 0.4 * band_conf)
    drift_term_soft = float(0.6 + 0.4 * drift_conf)

    artifact_quality = float(
        np.clip(w_emg * band_term_soft * drift_term_soft * blink_conf * float(contact_conf_used), 0.0, 1.0)
    )
    artifact_overlap = float(np.clip(1.0 - artifact_quality, 0.0, 1.0))
    phi_meaningful = float(phi * artifact_quality)
    phi_artifact = float(phi * artifact_overlap)

    # Qualia-ish / vibe options (A/B/C)
    # Option A: meaningful richness
    R_meaningful_raw = float(k * phi_meaningful * SAC_raw)
    # Option B: meaningful richness gated by engagement
    R_focus_meaningful_raw = float(k * phi_meaningful * SAC_raw * E)
    # Option C: intensity-weighted vibe
    Q_vibe_raw = float(R_meaningful_raw * X)
    # Option D: intensity-weighted meaningful richness gated by engagement
    Q_vibe_focus_raw = float(R_focus_meaningful_raw * X)

    # Overall quality confidence (validity uses this)
    vm = (validation_mode or VALIDATION_MODE_DEFAULT).strip().upper()
    # Channel agreement (ch_conf) is diagnostic only; it must not gate global validity.
    if vm.startswith("RELAX"):
        quality_conf = float(np.clip(w_emg * blink_conf * float(contact_conf_used), 0.0, 1.0))
    elif vm.startswith("STRICT"):
        quality_conf = float(
            np.clip(w_emg * band_conf * drift_conf * blink_conf * float(contact_conf_used), 0.0, 1.0)
        )
    else:  # ADAPTIVE
        quality_conf = float(
            np.clip(w_emg * band_term_soft * drift_term_soft * blink_conf * float(contact_conf_used), 0.0, 1.0)
        )

    valid = bool(quality_conf >= float(quality_thr) and np.isfinite(X) and np.isfinite(JS_dist))

    # Qualia masking (separate from global validity):
    # Even when global validity is RELAXED (high valid%), we still want qualia options to be
    # meaningfully clean. So we mask Options A/B/C/D based on artifact_quality rather than
    # quality_conf. This prevents "stream is present" from being confused with "state is meaningful".
    # (Tunable via constant; can be promoted to a UI slider later if desired.)
    qualia_thr = float(np.clip(QUALIA_VALID_ARTIFACT_QUALITY_THR_DEFAULT, 0.05, 0.99))
    qualia_valid = bool((artifact_quality >= qualia_thr) and np.isfinite(X) and np.isfinite(JS_dist))

    # Mask meaningful/vibe options using qualia_valid
    R_meaningful = float(R_meaningful_raw if qualia_valid else float("nan"))
    R_focus_meaningful = float(R_focus_meaningful_raw if qualia_valid else float("nan"))
    Q_vibe = float(Q_vibe_raw if qualia_valid else float("nan"))
    Q_vibe_focus = float(Q_vibe_focus_raw if qualia_valid else float("nan"))

    # Option E (permutation entropy + aperiodic slope multiplier on D)
    profile_id = (option_e_profile_id or OPTION_E_PROFILE_DEFAULT).strip()
    pe_tau = int(max(1, min(10, option_e_pe_tau)))
    stats = option_e_pick_stats(profile_id)
    mu_c, sig_c, _src_c = stats["C_pe"]
    mu_s, sig_s, _src_s = stats["S_flat"]
    option_e_stats_src = stats.get("stats_src", "none")

    C_pe_val = permutation_entropy_bandpassed(window_4ch, OPTION_E_SOS, OPTION_E_PE_M, pe_tau)
    S_aperiodic_slope = fit_aperiodic_slope(f_all, P_mean_raw, OPTION_E_SLOPE_FIT_HZ, OPTION_E_SLOPE_EXCLUDE_HZ)
    S_flat = float(-S_aperiodic_slope) if np.isfinite(S_aperiodic_slope) else float("nan")

    def _norm_feature(val, mu, sigma):
        if (mu is None) or (sigma is None) or (sigma <= 0) or (val is None) or (not np.isfinite(val)):
            return 0.0, 0.5, False
        z = float((val - mu) / (sigma + EPS))
        n = float(sigmoid(z))
        return z, n, True

    C_pe_z, C_pe_n, stats_c_ok = _norm_feature(C_pe_val, mu_c, sig_c)
    S_flat_z, S_flat_n, stats_s_ok = _norm_feature(S_flat, mu_s, sig_s)

    C_eff = float(0.5 + artifact_quality * (C_pe_n - 0.5))
    S_eff = float(0.5 + artifact_quality * (S_flat_n - 0.5))

    mult_E = float(
        np.clip(
            1.0 + OPTION_E_WC * (2.0 * (C_eff - 0.5)) + OPTION_E_WS * (2.0 * (S_eff - 0.5)),
            float(OPTION_E_MULT_CLIP[0]),
            float(OPTION_E_MULT_CLIP[1]),
        )
    )

    Q_vibe_focus_E_raw = float(Q_vibe_focus_raw * mult_E) if np.isfinite(Q_vibe_focus_raw) else float("nan")
    Q_vibe_focus_E = float(Q_vibe_focus_E_raw if qualia_valid else float("nan"))

    # Reason codes (diagnostic, not necessarily invalidation)
    reason_codes = []
    if (w_emg < EMG_REASON_THR) or (emg_index > (EMG_THR + 0.25)):
        reason_codes.append("high_emg")
    if band_conf < 0.3:
        reason_codes.append("low_band_mass")
    if ch_conf_raw < 0.3:
        reason_codes.append("low_channel_agreement")
    if ch_conf_raw < CH_CONF_HARD_MIN:
        reason_codes.append("ch_conf_extreme")
    if drift_conf < 0.5:
        reason_codes.append("high_drift")
    if blink_conf < 0.5:
        reason_codes.append("high_blink")
    # Qualia options are masked when artifact_quality drops below qualia_thr.
    # This is not necessarily an invalidation for the *global* Q metrics.
    if artifact_quality < qualia_thr:
        reason_codes.append("qualia_low_artifact_quality")
    if contact_method == "horseshoe":
        if float(contact_conf) < 0.5:
            reason_codes.append("low_contact")
    else:
        if float(contact_conf) < 0.25:
            reason_codes.append("low_contact")
    if rebinned:
        reason_codes.append("js_rebinned")
    if (not stats_c_ok) or (not stats_s_ok):
        reason_codes.append("option_e_missing_baseline")

    # Artifact hints for Option E reason codes
    hint_flags = []
    if artifact_quality < 0.60:
        hint_flags.append("aq")
    if w_emg < 0.40:
        hint_flags.append("emg")
    if blink_conf < 0.60:
        hint_flags.append("blink")
    if contact_conf_used < 0.60:
        hint_flags.append("contact")
    if drift_conf < 0.60:
        hint_flags.append("drift")
    hint_suffix = (":" + ",".join(hint_flags)) if hint_flags else ""
    if C_pe_z > 2.5 and hint_flags:
        reason_codes.append(f"C_spike_artifact_likely{hint_suffix}")
    if S_flat_z > 2.5 and hint_flags:
        reason_codes.append(f"S_flat_artifact_likely{hint_suffix}")

    
    provenance = "—"
    if phi >= PHI_HIGH_THR:
        if artifact_quality < ARTIFACT_QUALITY_LOW:
            provenance = "φ-high, artifact-likely"
            reason_codes.append("phi_high_artifact_likely")
        elif artifact_quality < ARTIFACT_QUALITY_UNCERTAIN:
            provenance = "φ-high, uncertain"
            reason_codes.append("phi_high_uncertain")
        else:
            provenance = "φ-high, candidate state"
            reason_codes.append("phi_high_candidate_state")

    return Metrics(
        X=X,
        R_abs=R_abs,

        R_meaningful=R_meaningful,
        R_focus_meaningful=R_focus_meaningful,
        Q_vibe=Q_vibe,
        Q_vibe_focus=Q_vibe_focus,
        Q_vibe_focus_E=Q_vibe_focus_E,
        Q_vibe_focus_E_raw=Q_vibe_focus_E_raw,
        Q_vibe_focus_E_mult=mult_E,

        R_meaningful_raw=R_meaningful_raw,
        R_focus_meaningful_raw=R_focus_meaningful_raw,
        Q_vibe_raw=Q_vibe_raw,
        Q_vibe_focus_raw=Q_vibe_focus_raw,

        Q_abs=(Q_abs_raw if valid else float("nan")),
        Q_perX=(Q_perX_raw if valid else float("nan")),
        R_focus=R_focus,
        Q_focus=(Q_focus_raw if valid else float("nan")),

        Q_abs_raw=Q_abs_raw,
        Q_perX_raw=Q_perX_raw,
        Q_focus_raw=Q_focus_raw,

        JS_dist=JS_dist,
        phi=phi,

        quality_conf=quality_conf,
        valid=valid,
        qualia_valid=qualia_valid,
        reason_codes=";".join(reason_codes),

        artifact_quality=artifact_quality,
        artifact_overlap=artifact_overlap,
        phi_meaningful=phi_meaningful,
        phi_artifact=phi_artifact,
        provenance=provenance,

        theta=theta,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        total_bands=total_bands,
        theta_n=theta_n,
        alpha_n=alpha_n,
        beta_n=beta_n,
        gamma_n=gamma_n,
        emg_index=emg_index,
        w_emg=w_emg,
        bg_raw=bg_raw,
        E=E,
        gamma_scaled=gamma_scaled,
        a=a,
        b=b,
        k=k,
        SAC_raw=SAC_raw,
        SAC_focus=SAC_focus,
        ch_corr=ch_corr,
        ch_conf_raw=ch_conf_raw,
        ch_conf_used=ch_conf_used,
        band_conf=band_conf,

        drift_mass=drift_mass,
        drift_over=drift_over,
        drift_conf=drift_conf,
        blink_rate=blink_rate,
        blink_conf=blink_conf,

        contact_conf=float(contact_conf),
        contact_raw=str(contact_raw),
        contact_method=str(contact_method),
        contact_conf_used=contact_conf_used,

        C_pe=float(C_pe_val) if np.isfinite(C_pe_val) else float("nan"),
        C_pe_z=float(C_pe_z),
        C_pe_n=float(C_pe_n),
        S_aperiodic_slope=float(S_aperiodic_slope) if np.isfinite(S_aperiodic_slope) else float("nan"),
        S_flat=float(S_flat) if np.isfinite(S_flat) else float("nan"),
        S_aperiodic_slope_z=float(S_flat_z),
        S_aperiodic_slope_n=float(S_flat_n),
        option_e_profile_id=str(profile_id),
        option_e_stats_src=str(option_e_stats_src),
        option_e_pe_tau=float(pe_tau),
    )


# ------------------------------
# Dash state
# ------------------------------
def new_session_id() -> str:
    return f"session_{int(time.time())}"


global_state = {
    "inlet_eeg": None,
    "inlet_hs": None,
    "stream_status": "Not connected",
    "last_eeg_wall": None,
    "stream_meta": {},
    "hs_meta": {},
    "ch_indices": [0, 1, 2, 3],
    "ch_labels": ["TP9", "AF7", "AF8", "TP10"],
    "option_e_profile_id": OPTION_E_PROFILE_DEFAULT,
    "option_e_stats_src": "init",
    "option_e_pe_tau": OPTION_E_PE_TAU_DEFAULT,

    "buffers": [deque(maxlen=FS_TARGET * 600) for _ in range(4)],
    "ts": deque(maxlen=FS_TARGET * 600),

    "hs_last_vals": None,
    "hs_last_ts": None,

    "session_id": new_session_id(),
    "t0": time.time(),

    # to avoid duplicate metric rows when no new EEG arrives
    "n_total_samples": 0,
    "last_compute_samples": 0,

    # contact inference state
    "contact_infer": {
        "rms_hist": [deque(maxlen=CONTACT_HIST_N) for _ in range(4)],
        "std_hist": [deque(maxlen=CONTACT_HIST_N) for _ in range(4)],
        "ratio_hist": deque(maxlen=CONTACT_HIST_N),
    },

    # session-adaptive drift reference
    "drift_ref_ema": DRIFT_REF,

    # drift UI tuning
    "drift_aggr": DRIFT_AGGR_DEFAULT,
    "warmup_ignore_drift_sec": WARMUP_IGNORE_DRIFT_SEC,

    # NDJSON state stream (for local adaptive apps)
    "emit_ndjson_enabled": EMIT_NDJSON_DEFAULT,
    "emit_ndjson_last_len": 0,
    "emit_ndjson_path": NDJSON_PATH,


    # raw history (EXPORT)
    "history": {
        "t": [],
        "X": [],
        "R_abs": [],

        # qualia-ish / vibe options (masked valid-only)
        "R_meaningful": [],
        "R_focus_meaningful": [],
        "Q_vibe": [],
        "Q_vibe_focus": [],
        "Q_vibe_focus_E": [],

        # qualia-ish / vibe options (raw/unmasked)
        "R_meaningful_raw": [],
        "R_focus_meaningful_raw": [],
        "Q_vibe_raw": [],
        "Q_vibe_focus_raw": [],
        "Q_vibe_focus_E_raw": [],
        "Q_vibe_focus_E_mult": [],

        # valid (masked)
        "Q_abs": [],
        "Q_perX": [],
        "R_focus": [],
        "Q_focus": [],

        # raw (unmasked)
        "Q_abs_raw": [],
        "Q_perX_raw": [],
        "Q_focus_raw": [],

        "JS_dist": [],
        "phi": [],
        "quality_conf": [],
        "valid": [],
        "qualia_valid": [],
        "reason_codes": [],
        "quality_thr": [],

        # drift tuning (for traceability)
        "drift_aggr": [],
        "warmup_ignore_drift_sec": [],

        # 12.3 provenance
        "artifact_quality": [],
        "artifact_overlap": [],
        "phi_meaningful": [],
        "phi_artifact": [],
        "provenance": [],

        # drivers / diagnostics
        "theta": [],
        "alpha": [],
        "beta": [],
        "gamma": [],
        "total_bands": [],
        "theta_n": [],
        "alpha_n": [],
        "beta_n": [],
        "gamma_n": [],
        "emg_index": [],
        "w_emg": [],
        "bg_raw": [],
        "E": [],
        "gamma_scaled": [],
        "a": [],
        "b": [],
        "k": [],
        "SAC_raw": [],
        "SAC_focus": [],
        "ch_corr": [],
        "ch_conf_raw": [],
        "ch_conf_used": [],
        "band_conf": [],

        # artifact proxies
        "drift_mass": [],
        "drift_over": [],
        "drift_conf": [],
        "blink_rate": [],
        "blink_conf": [],

        # contact
        "contact_conf": [],
        "contact_conf_used": [],
        "contact_raw": [],
        "contact_method": [],

        # Option E drivers
        "C_pe": [],
        "C_pe_z": [],
        "C_pe_n": [],
        "S_aperiodic_slope": [],
        "S_flat": [],
        "S_aperiodic_slope_z": [],
        "S_aperiodic_slope_n": [],
        "option_e_profile_id": [],
        "option_e_stats_src": [],
        "option_e_pe_tau": [],

        # traceability
        "baseline_version": [],
        "code_version": [],
    },

    # ema history (DISPLAY ONLY; not exported)
    "ema": {
        "X": None,
        "Q_abs": None,
        "Q_focus": None,
        "R_abs": None,

        "R_meaningful": None,
        "R_focus_meaningful": None,
        "Q_vibe": None,
        "Q_vibe_focus": None,
        "Q_vibe_focus_E": None,
        "quality_conf": None,
        "phi": None,
        "JS_dist": None,
        "w_emg": None,
        "E": None,
        "artifact_quality": None,
    },
    "history_ema": {
        "ema_X": [],
        "ema_Q_abs": [],
        "ema_Q_focus": [],
        "ema_R_abs": [],

        "ema_R_meaningful": [],
        "ema_R_focus_meaningful": [],
        "ema_Q_vibe": [],
        "ema_Q_vibe_focus": [],
        "ema_Q_vibe_focus_E": [],
        "ema_quality_conf": [],
        "ema_phi": [],
        "ema_JS_dist": [],
        "ema_w_emg": [],
        "ema_E": [],
        "ema_artifact_quality": [],
    },

    "events": [],
}


# ------------------------------
# LSL connection helpers
# ------------------------------
def connect_eeg(prefer_name: str = "Muse") -> None:
    """Connect to LSL EEG stream."""
    streams = []
    try:
        streams = resolve_byprop("name", prefer_name, timeout=2)
    except Exception:
        streams = []

    # filter for EEG if multiple by name
    streams = [s for s in streams if s.type().lower() == "eeg"] or streams

    if not streams:
        try:
            streams = resolve_byprop("type", "EEG", timeout=2)
        except Exception:
            streams = []

    if not streams:
        global_state["inlet_eeg"] = None
        global_state["stream_status"] = "No LSL EEG stream found"
        return

    inlet = StreamInlet(streams[0])
    info = inlet.info()

    n_ch = info.channel_count()
    fs = info.nominal_srate()
    labels = get_channel_labels(info)
    eeg_idx, eeg_labels = pick_muse_eeg_indices(labels, n_ch)

    global_state["inlet_eeg"] = inlet
    global_state["ch_indices"] = eeg_idx
    global_state["ch_labels"] = eeg_labels
    global_state["stream_meta"] = {
        "name": info.name(),
        "type": info.type(),
        "n_ch": n_ch,
        "fs": fs,
        "picked_idx": eeg_idx,
        "picked_labels": eeg_labels,
    }
    global_state["stream_status"] = (
        f"EEG connected: {info.name()} | {n_ch}ch @ {fs:.1f}Hz | idx={eeg_idx} | labels={eeg_labels}"
    )


def _score_contact_stream(s) -> float:
    """Heuristic scoring to pick best horseshoe/contact stream from all visible LSL streams."""
    try:
        name = (s.name() or "").lower()
        typ = (s.type() or "").lower()
        n_ch = int(s.channel_count())
        fs = float(s.nominal_srate())
    except Exception:
        return -1e9

    # Don't accidentally select EEG
    if typ == "eeg":
        return -1e9

    score = 0.0
    tokens = f"{name} {typ}"

    if "horseshoe" in tokens:
        score += 10
    if "contact" in tokens:
        score += 8
    if "hsi" in tokens:
        score += 7
    if "impedance" in tokens:
        score += 6
    if "muse" in tokens:
        score += 3

    if typ in {"horseshoe", "contact", "hsi"}:
        score += 6

    # Typical channel count ~4 (Muse sensors)
    if 3 <= n_ch <= 6:
        score += 2

    # Typical horseshoe rates are low (<= 50 Hz, often ~10 Hz)
    if 0 < fs <= 50:
        score += 2
    if 0 < fs <= 10:
        score += 1

    return score


def connect_horseshoe() -> None:
    """Try to connect to a horseshoe/contact stream if available (robust discovery)."""
    try:
        all_streams = resolve_streams()
    except Exception:
        all_streams = []

    best = None
    best_score = -1e9
    for s in all_streams:
        sc = _score_contact_stream(s)
        if sc > best_score:
            best = s
            best_score = sc

    # Fallback resolution by common type names
    if best is None or best_score < 6:
        candidates = []
        for t in ["Horseshoe", "Contact", "HSI"]:
            try:
                candidates = resolve_byprop("type", t, timeout=1)
            except Exception:
                candidates = []
            if candidates:
                best = candidates[0]
                best_score = 6
                break

    if best is None or best_score < 6:
        global_state["inlet_hs"] = None
        global_state["hs_meta"] = {"status": "No horseshoe/contact stream found"}
        return

    inlet = StreamInlet(best)
    info = inlet.info()

    global_state["inlet_hs"] = inlet
    global_state["hs_meta"] = {
        "name": info.name(),
        "type": info.type(),
        "n_ch": info.channel_count(),
        "fs": info.nominal_srate(),
        "score": best_score,
        "status": "Horseshoe/contact connected",
    }


def get_horseshoe_contact_conf() -> Tuple[float, str, str]:
    """Pull latest horseshoe/contact sample if available. Returns (conf, raw, method)."""
    inlet = global_state.get("inlet_hs")
    if inlet is None:
        return 1.0, "", "none"

    try:
        # horseshoe streams are low-rate; pull most recent chunk
        samples, ts = inlet.pull_chunk(timeout=0.0)
        if samples:
            arr = np.array(samples, dtype=float)
            last = arr[-1]
            global_state["hs_last_vals"] = last
            global_state["hs_last_ts"] = ts[-1] if ts else None
    except Exception:
        pass

    vals = global_state.get("hs_last_vals")
    if vals is None:
        return 1.0, "", "horseshoe"

    conf = horseshoe_to_conf(np.array(vals))
    raw = ",".join([f"{v:.3g}" for v in np.array(vals).flatten().tolist()])
    return float(conf), raw, "horseshoe"


def get_contact_conf_for_window(window_4ch: np.ndarray) -> Tuple[float, str, str]:
    """
    Contact confidence selection:
    - Prefer real horseshoe/contact stream if available.
    - Otherwise infer from EEG window.
    """
    hs_conf, hs_raw, hs_method = get_horseshoe_contact_conf()
    if hs_method == "horseshoe" and (global_state.get("inlet_hs") is not None):
        return float(hs_conf), hs_raw, "horseshoe"
    conf, raw, method = infer_contact_conf(window_4ch)
    return float(conf), raw, method


# ------------------------------
# Dash app layout
# ------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Cortex Console V4.2.1 (Muse 2, Absolute X/Q)"

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row([
            dbc.Col(html.H1(f"Cortex Console V4.2.1 (Muse 2) — Absolute X/Q  |  code={CODE_VERSION}  |  dash={SCRIPT_VARIANT}"), width=9),
            dbc.Col(html.Div(id="lsl-status", style={"textAlign": "right", "marginTop": "18px"}), width=3),
        ]),

        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Div(id="session-box"),
                        html.Hr(),

                        dbc.Button("Connect / Reconnect LSL", id="btn-connect", color="primary", className="me-2"),
                        dbc.Button("New Session (Reset)", id="btn-reset", color="warning", className="me-2"),
                        dbc.Button("Save Session", id="btn-save", color="success", className="me-2"),
                        html.Div(style={"height": "10px"}),

                        html.Div("Event label (optional):", style={"fontSize": "12px"}),
                        dbc.Input(
                            id="event-label",
                            placeholder="baseline_start / jaw_clench_start / song_drop / ...",
                            type="text",
                            value="event",
                        ),
                        dbc.Button("Mark Event", id="btn-event", color="secondary", className="mt-2"),
                        html.Div(style={"height": "10px"}),

                        dbc.Checklist(
                            id="emit-ndjson",
                            options=[{"label": f"Emit State Stream (NDJSON → {NDJSON_STREAM_PATH})", "value": "on"}],
                            value=(["on"] if EMIT_NDJSON_DEFAULT else []),
                            switch=True,
                            className="mt-2",
                        ),

                        html.Hr(),
                        html.Div("Adaptive Mode Control (for Spotify controller):"),
                        dbc.RadioItems(
                            id="adaptive-mode",
                            options=[
                                {"label": "OFF", "value": "OFF"},
                                {"label": "MAX_Q", "value": "MAX_Q"},
                                {"label": "MAX_FOCUS", "value": "MAX_FOCUS"},
                                {"label": "STABILIZE", "value": "STABILIZE"},
                            ],
                            value="OFF",
                            inline=True,
                            className="mb-2",
                        ),
                        dbc.Button("Write Mode", id="btn-write-mode", color="info", className="me-2", size="sm"),
                        html.Div(id="mode-status", style={"fontSize": "12px", "marginTop": "6px"}),

                        html.Div("QUALITY_VALID_THR (tune to inspect false invalidation vs false validity):"),
                        dcc.Slider(
                            id="quality-thr",
                            min=0.05,
                            max=0.70,
                            step=0.01,
                            value=QUALITY_VALID_THR_DEFAULT,
                            marks={0.1: "0.10", 0.25: "0.25", 0.4: "0.40", 0.55: "0.55", 0.7: "0.70"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),

                        dbc.Checklist(
                            id="show-debug-traces",
                            options=[{"label": "Show masked + EMA traces (debug)", "value": "on"}],
                            value=[],
                            switch=True,
                            className="mt-2",
                        ),

                        html.Hr(),
                        html.Div("Option E profile:"),
                        dcc.Dropdown(
                            id="option-e-profile",
                            options=[{"label": k, "value": k} for k in OPTION_E_PROFILE_KEYS],
                            value=OPTION_E_PROFILE_DEFAULT,
                            clearable=False,
                            style={"color": "#000"},
                        ),
                        html.Div("Option E: Permutation Entropy τ (tau)", style={"marginTop": "8px"}),
                        dcc.Slider(
                            id="option-e-pe-tau",
                            min=1,
                            max=10,
                            step=1,
                            value=OPTION_E_PE_TAU_DEFAULT,
                            marks={1: "1", 5: "5", 10: "10"},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag",
                        ),

                        html.Hr(),

                        html.Div("Validation mode (validity gate):"),
                        dbc.RadioItems(
                            id="validation-mode",
                            options=[
                                {"label": "RELAXED (min false negatives)", "value": "RELAXED"},
                                {"label": "ADAPTIVE (soft band/drift)", "value": "ADAPTIVE"},
                                {"label": "STRICT (original)", "value": "STRICT"},
                            ],
                            value=VALIDATION_MODE_DEFAULT,
                            inline=False,
                            className="mb-2",
                        ),

                        html.Div(style={"height": "8px"}),
                        html.Div("PSD high-pass for drift/band gating (Hz):"),
                        dcc.Slider(
                            id="psd-hp-hz",
                            min=0.0,
                            max=2.0,
                            step=0.1,
                            value=PSD_HP_HZ_DEFAULT,
                            marks={0.0: "0.0", 0.5: "0.5", 1.0: "1.0", 1.5: "1.5", 2.0: "2.0"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),

                        html.Hr(),
                        html.Div("Reliability (rolling):"),
                        html.Div(
                            [
                                html.Span("Window (samples): "),
                                dcc.Input(
                                    id="rel-window",
                                    type="number",
                                    value=600,
                                    min=20,
                                    max=5000,
                                    step=10,
                                    style={"width": "120px", "marginLeft": "8px"},
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                        html.Pre(
                            id="reliability-box",
                            style={
                                "whiteSpace": "pre-wrap",
                                "fontFamily": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
                                "fontSize": "12px",
                                "marginTop": "8px",
                            },
                        ),

                        html.Hr(),
                        html.Div("Drift proxy (advanced):"),
                        html.Div("Drift aggressiveness (1.0 = default; <1 looser; >1 stricter):"),
                        dcc.Slider(
                            id="drift-aggr",
                            min=0.5,
                            max=2.0,
                            step=0.05,
                            value=DRIFT_AGGR_DEFAULT,
                            marks={0.5: "0.5×", 1.0: "1.0×", 1.5: "1.5×", 2.0: "2.0×"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.Div(style={"height": "8px"}),
                        html.Div("Ignore drift penalty during warmup (seconds):"),
                        dcc.Slider(
                            id="drift-warmup",
                            min=0,
                            max=180,
                            step=5,
                            value=WARMUP_IGNORE_DRIFT_SEC,
                            marks={0: "0", 60: "60", 90: "90", 120: "120", 180: "180"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.Div(style={"height": "8px"}),
                        html.Div(id="save-status", style={"marginTop": "10px"}),
                        html.Hr(),
                        html.Div([
                            html.Div(f"Baseline: {BASELINE_VERSION}"),
                            html.Div(f"Baseline file: {BASELINE_PATH}"),
                        ], style={"opacity": 0.85}),
                    ])
                ),
                width=4,
            ),

            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Raw Q (Abs / Vibe / Vibe Focus) — last 420s"),
                        dcc.Graph(id="graph-qraw420", config={"displaylogo": False}),
                        html.H4("X (raw + EMA)"),
                        dcc.Graph(id="graph-x", config={"displaylogo": False}),
                        html.H4("Q / Richness (raw + EMA)"),
                        dcc.Graph(id="graph-q", config={"displaylogo": False}),
                        html.H4("Qualia Options (A/B/C/D) — Meaningful Richness & Vibe (raw + EMA)"),
                        dcc.Graph(id="graph-qualia", config={"displaylogo": False}),
                        html.H4("Drivers / Quality (raw + EMA)"),
                        dcc.Graph(id="graph-drivers", config={"displaylogo": False}),
                        html.H4("Bands & Artifact Proxies"),
                        dcc.Graph(id="graph-bands", config={"displaylogo": False}),
                    ])
                ),
                width=8,
            ),
        ]),

        dcc.Interval(id="interval", interval=STEP_SEC * 1000, n_intervals=0),
    ],
)


# ------------------------------
# UI helpers
# ------------------------------
def _fmt(v, nd=4):
    try:
        if v is None:
            return "—"
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            return "NaN"
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)



# ------------------------------
# NDJSON state stream helpers
# ------------------------------
def _jsonable(v):
    """Convert numpy/scalar types to JSON-safe builtins; NaN/inf -> None."""
    try:
        if v is None:
            return None
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            fv = float(v)
            if not np.isfinite(fv):
                return None
            return fv
        return v
    except Exception:
        return None


def _split_reasons(s):
    if s is None:
        return []
    try:
        s = str(s).strip()
        if not s:
            return []
        # reasons are stored as 'a;b;c' in CSV/history
        return [r for r in s.split(";") if r]
    except Exception:
        return []


def build_state_layer_packet_from_history(h, i, _session_id):
    """Build a State Layer API v0.1 packet from flat history arrays."""
    valid = bool(h["valid"][i])

    def _masked(v):
        return _jsonable(v if valid else None)

    pkt = {
        "type": "state_sample",
        "schema_version": NDJSON_SCHEMA_VERSION,
        "t": _jsonable(h["t"][i]),
        "state": {
            "X": _jsonable(h["X"][i]),
            "R_abs": _jsonable(h["R_abs"][i]),
            "Q_abs": _masked(h["Q_abs"][i]),
            "Q_focus": _masked(h["Q_focus"][i]),
            "Q_perX": _masked(h["Q_perX"][i]),

            "R_meaningful": _masked(h["R_meaningful"][i]) if "R_meaningful" in h else None,
            "R_focus_meaningful": _masked(h["R_focus_meaningful"][i]) if "R_focus_meaningful" in h else None,
            "Q_vibe": _masked(h["Q_vibe"][i]) if "Q_vibe" in h else None,
            "Q_vibe_focus": _masked(h["Q_vibe_focus"][i]) if "Q_vibe_focus" in h else None,
            "Q_vibe_focus_E": _jsonable(h.get("Q_vibe_focus_E", [None])[i]) if ("Q_vibe_focus_E" in h and bool(h.get("qualia_valid", [False])[i])) else None,
            "raw": {
                "Q_abs": _jsonable(h["Q_abs_raw"][i]),
                "Q_focus": _jsonable(h["Q_focus_raw"][i]),
                "Q_perX": _jsonable(h["Q_perX_raw"][i]),

                "R_meaningful": _jsonable(h["R_meaningful_raw"][i]) if "R_meaningful_raw" in h else None,
                "R_focus_meaningful": _jsonable(h["R_focus_meaningful_raw"][i]) if "R_focus_meaningful_raw" in h else None,
                "Q_vibe": _jsonable(h["Q_vibe_raw"][i]) if "Q_vibe_raw" in h else None,
                "Q_vibe_focus": _jsonable(h["Q_vibe_focus_raw"][i]) if "Q_vibe_focus_raw" in h else None,
                "Q_vibe_focus_E": _jsonable(h.get("Q_vibe_focus_E_raw", [None])[i]) if "Q_vibe_focus_E_raw" in h else None,
                "Q_vibe_focus_E_mult": _jsonable(h.get("Q_vibe_focus_E_mult", [None])[i]) if "Q_vibe_focus_E_mult" in h else None,
                "C_pe": _jsonable(h.get("C_pe", [None])[i]) if "C_pe" in h else None,
                "C_pe_z": _jsonable(h.get("C_pe_z", [None])[i]) if "C_pe_z" in h else None,
                "C_pe_n": _jsonable(h.get("C_pe_n", [None])[i]) if "C_pe_n" in h else None,
                "S_aperiodic_slope": _jsonable(h.get("S_aperiodic_slope", [None])[i]) if "S_aperiodic_slope" in h else None,
                "S_aperiodic_slope_z": _jsonable(h.get("S_aperiodic_slope_z", [None])[i]) if "S_aperiodic_slope_z" in h else None,
                "S_aperiodic_slope_n": _jsonable(h.get("S_aperiodic_slope_n", [None])[i]) if "S_aperiodic_slope_n" in h else None,
                "R_focus": _jsonable(h["R_focus"][i]),
            },
        },
        "drivers": {
            "phi": _jsonable(h["phi"][i]),
            "JS_dist": _jsonable(h["JS_dist"][i]),
            "k": _jsonable(h["k"][i]) if "k" in h else None,
            "SAC_raw": _jsonable(h["SAC_raw"][i]) if "SAC_raw" in h else None,
            "SAC_focus": _jsonable(h["SAC_focus"][i]) if "SAC_focus" in h else None,
            "bands": {
                "theta_n": _jsonable(h["theta_n"][i]) if "theta_n" in h else None,
                "alpha_n": _jsonable(h["alpha_n"][i]) if "alpha_n" in h else None,
                "beta_n": _jsonable(h["beta_n"][i]) if "beta_n" in h else None,
                "gamma_n": _jsonable(h["gamma_n"][i]) if "gamma_n" in h else None,
            },
        },
        "reliability": {
            "valid": valid,
            "quality_conf": _jsonable(h["quality_conf"][i]),
            "reason_codes": _split_reasons(h["reason_codes"][i]),
            "artifact_quality": _jsonable(h["artifact_quality"][i]) if "artifact_quality" in h else None,
            "artifact_overlap": _jsonable(h["artifact_overlap"][i]) if "artifact_overlap" in h else None,
            "provenance": str(h["provenance"][i]) if "provenance" in h else "",
            "contact_conf_used": _jsonable(h["contact_conf_used"][i]) if "contact_conf_used" in h else None,
            "qualia_valid": bool(h.get("qualia_valid", [False])[i]) if "qualia_valid" in h else None,
        },
        "trace": {
            "baseline_version": str(h["baseline_version"][i]) if "baseline_version" in h else "",
            "code_version": str(h["code_version"][i]) if "code_version" in h else "",
        },
        "meta": {
            "option_e_profile_id": str(h.get("option_e_profile_id", [""])[i]) if "option_e_profile_id" in h else "",
            "option_e_stats_src": str(h.get("option_e_stats_src", [""])[i]) if "option_e_stats_src" in h else "",
            "pe_tau": _jsonable(h.get("option_e_pe_tau", [None])[i]) if "option_e_pe_tau" in h else None,
        },
    }
    return pkt


def append_ndjson_packet(packet, path=NDJSON_PATH):
    """Append a single packet as one NDJSON line."""
    try:
        # Make sure parent dir exists if provided
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(packet, ensure_ascii=False) + "\n")
            f.flush()
    except Exception as e:
        # Keep the instrument running even if streaming fails
        global_state["last_emit_error"] = str(e)


def write_mode_file(mode: str, notes: str = "set from dash"):
    payload = {"mode": str(mode).upper(), "updated_at": time.time(), "notes": notes}
    parent = os.path.dirname(MODE_FILE_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = MODE_FILE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, MODE_FILE_PATH)


# ------------------------------
# Callbacks
# ------------------------------
@app.callback(Output("lsl-status", "children"), Input("interval", "n_intervals"))
def show_status(_):
    hs = global_state.get("hs_meta", {})
    hs_status = hs.get("status", "")

    base = str(global_state.get("stream_status", "") or "")
    if hs_status:
        base = f"{base}  |  {hs_status}"

    now = time.time()
    last = global_state.get("last_eeg_wall", None)
    live = False
    try:
        live = (last is not None) and ((now - float(last)) < 2.5)
    except Exception:
        live = False

    dot_color = "#2ecc71" if live else "#e74c3c"
    dot = html.Span("●", style={"color": dot_color, "fontSize": "18px", "marginRight": "6px"})
    badge = html.Span("LIVE" if live else "STALE", style={"fontSize": "12px", "opacity": 0.9, "marginRight": "10px"})
    return html.Div([dot, badge, html.Span(base)], style={"whiteSpace": "pre-wrap"})


@app.callback(Output("session-box", "children"), Input("interval", "n_intervals"))
def show_session_box(_):
    h = global_state["history"]
    if not h["t"]:
        return html.Div([
            html.Div(f"Session: {global_state['session_id']}"),
            html.Div("Waiting for data…"),
            html.Hr(),
            html.Div("V4: raw-first Q plots + Qualia Options A/B/C/D + live stream indicator. (Use debug toggle to show masked/EMA traces.)")
        ])

    i = -1

    qabs_v = h["Q_abs"][i]
    qfx_v = h["Q_focus"][i]
    qabs_r = h["Q_abs_raw"][i]
    qfx_r = h["Q_focus_raw"][i]
    opt_e_profile = (h.get("option_e_profile_id") or ["?"])[i]
    opt_e_src = (h.get("option_e_stats_src") or ["?"])[i]
    opt_e_tau = (h.get("option_e_pe_tau") or [np.nan])[i]

    return html.Div([
        html.Div(f"Session: {global_state['session_id']}"),
        html.Div(f"t: {_fmt(h['t'][i], 1)}s"),
        html.Hr(),

        html.Div(
            f"X: {_fmt(h['X'][i])}  |  Q_abs(valid): {_fmt(qabs_v, 6)}  |  Q_focus(valid): {_fmt(qfx_v, 6)}"
        ),
        html.Div(
            f"Q_abs_raw: {_fmt(qabs_r, 6)}  |  Q_focus_raw: {_fmt(qfx_r, 6)}"
        ),
        html.Div(
            f"R_abs: {_fmt(h['R_abs'][i], 6)}  |  Q_perX(valid): {_fmt(h['Q_perX'][i], 6)}  |  Q_perX_raw: {_fmt(h['Q_perX_raw'][i], 6)}"
        ),

        html.Div(
            f"R_meaningful(valid): {_fmt(h['R_meaningful'][i], 6)}  |  R_focus_meaningful(valid): {_fmt(h['R_focus_meaningful'][i], 6)}  |  Q_vibe(valid): {_fmt(h['Q_vibe'][i], 6)}  |  Q_vibe_focus(valid): {_fmt(h.get('Q_vibe_focus', [np.nan])[i], 6)}"
        ),
        html.Div(
            f"Option E profile: {opt_e_profile} | src: {opt_e_src} | \u03c4: {_fmt(opt_e_tau, 0)}"
        ),

        html.Hr(),
        html.Div(f"phi: {_fmt(h['phi'][i])}  |  JS_dist: {_fmt(h['JS_dist'][i])}"),

        html.Div(
            f"artifact_quality: {_fmt(h['artifact_quality'][i], 3)}  |  overlap: {_fmt(h['artifact_overlap'][i], 3)}"
        ),
        html.Div(
            f"phi_meaningful: {_fmt(h['phi_meaningful'][i], 3)}  |  phi_artifact: {_fmt(h['phi_artifact'][i], 3)}"
        ),
        html.Div(
            [
                html.Strong("Deviation provenance: "),
                html.Span(str(h["provenance"][i])),
            ],
            style={"marginTop": "4px"},
        ),

        html.Hr(),
        html.Div(
            f"quality_conf: {_fmt(h['quality_conf'][i], 3)}  |  valid: {h['valid'][i]}  |  thr: {_fmt(h['quality_thr'][i], 2)}"
        ),
        html.Div(
            f"w_emg: {_fmt(h['w_emg'][i], 3)}  |  emg_index: {_fmt(h['emg_index'][i], 3)}"
        ),
        html.Div(
            f"ch_conf(raw/used): {_fmt(h['ch_conf_raw'][i], 3)} / {_fmt(h['ch_conf_used'][i], 3)}  |  band_conf: {_fmt(h['band_conf'][i], 3)}"
        ),
        html.Div(
            f"contact_conf: {_fmt(h['contact_conf'][i], 3)} ({h['contact_method'][i]})  |  blink_conf: {_fmt(h['blink_conf'][i], 3)}  |  drift_conf: {_fmt(h['drift_conf'][i], 3)}"
        ),
        html.Div(
            f"drift_aggr: {_fmt(h['drift_aggr'][i], 2)}  |  warmup_ignore_drift_sec: {_fmt(h['warmup_ignore_drift_sec'][i], 0)}"
        ),
        html.Div(
            f"E: {_fmt(h['E'][i], 3)}  |  bg_raw: {_fmt(h['bg_raw'][i], 3)}"
        ),
        html.Div(f"reasons: {h['reason_codes'][i]}"),
    ])


@app.callback(
    Output("save-status", "children"),
    Output("mode-status", "children"),
    Input("btn-connect", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    Input("btn-save", "n_clicks"),
    Input("btn-event", "n_clicks"),
    Input("btn-write-mode", "n_clicks"),
    Input("adaptive-mode", "value"),
    State("event-label", "value"),
    prevent_initial_call=True,
)
def handle_buttons(nc, nr, ns, ne, nmode, mode_value, label):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", ""
    trig = ctx.triggered[0]["prop_id"].split(".")[0]

    if trig == "btn-connect":
        connect_eeg("Muse")
        connect_horseshoe()
        return dbc.Alert("Reconnected streams.", color="info", duration=2000), ""

    if trig == "btn-reset":
        global_state["session_id"] = new_session_id()
        global_state["t0"] = time.time()
        global_state["last_eeg_wall"] = None
        global_state["events"] = []
        global_state["n_total_samples"] = 0
        global_state["last_compute_samples"] = 0

        global_state["ema"] = {k: None for k in global_state["ema"].keys()}

        for k in list(global_state["history"].keys()):
            global_state["history"][k] = []
        for k in list(global_state["history_ema"].keys()):
            global_state["history_ema"][k] = []

        for b in global_state["buffers"]:
            b.clear()
        global_state["ts"].clear()

        # reset NDJSON streaming cursor
        global_state["emit_ndjson_last_len"] = 0

        # reset contact inference
        global_state["contact_infer"] = {
            "rms_hist": [deque(maxlen=CONTACT_HIST_N) for _ in range(4)],
            "std_hist": [deque(maxlen=CONTACT_HIST_N) for _ in range(4)],
            "ratio_hist": deque(maxlen=CONTACT_HIST_N),
        }

        

        # reset session-adaptive drift ref
        global_state["drift_ref_ema"] = DRIFT_REF
        return dbc.Alert("Session reset.", color="warning", duration=2500), ""

    if trig == "btn-event":
        t = time.time() - global_state["t0"]
        lab = (label or "event").strip()
        global_state["events"].append({"t": float(t), "label": lab})
        return dbc.Alert(f"Event marked: '{lab}' at t={t:.1f}s", color="secondary", duration=2500), ""

    if trig == "btn-save":
        os.makedirs("sessions", exist_ok=True)
        sid = global_state["session_id"]

        df = pd.DataFrame(global_state["history"])
        csv_path = os.path.join("sessions", f"{sid}.csv")
        df.to_csv(csv_path, index=False)

        payload = {
            "session_id": sid,
            "created": time.time(),
            "code_version": CODE_VERSION,
            "stream_meta": global_state.get("stream_meta", {}),
            "horseshoe_meta": global_state.get("hs_meta", {}),
            "baseline": {
                "file": BASELINE_PATH,
                "version": BASELINE_VERSION,
                "mu_JS": MU_JS,
                "p90_JS": P90_JS,
                "gamma_ref": GAMMA_REF,
                "bg_ref": BG_REF,
            },
            "params": {
                "QUALITY_VALID_THR_DEFAULT": QUALITY_VALID_THR_DEFAULT,
                "DRIFT_OVER_THR": DRIFT_OVER_THR,
                "DRIFT_OVER_SCALE": DRIFT_OVER_SCALE,
                "DRIFT_AGGR_DEFAULT": DRIFT_AGGR_DEFAULT,
                "WARMUP_IGNORE_DRIFT_SEC_DEFAULT": WARMUP_IGNORE_DRIFT_SEC,
                "drift_aggr": float(global_state.get("drift_aggr", DRIFT_AGGR_DEFAULT)),
                "warmup_ignore_drift_sec": float(global_state.get("warmup_ignore_drift_sec", WARMUP_IGNORE_DRIFT_SEC)),
                "CH_CONF_SOFT_FLOOR": CH_CONF_SOFT_FLOOR,
                "CH_CONF_HARD_MIN": CH_CONF_HARD_MIN,
                "PHI_HIGH_THR": PHI_HIGH_THR,
                "ARTIFACT_QUALITY_LOW": ARTIFACT_QUALITY_LOW,
            },
            "events": global_state.get("events", []),
            "history": global_state["history"],
        }
        json_path = os.path.join("sessions", f"{sid}.json")
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        return dbc.Alert(f"Saved: {csv_path} and {json_path}", color="success", duration=5000), ""

    if trig in {"btn-write-mode", "adaptive-mode"}:
        m = (mode_value or "OFF").upper()
        write_mode_file(m, notes="set from dash")
        return "", f"Mode set to {m} → {MODE_FILE_PATH}"

    return "", ""


@app.callback(
    Output("graph-qraw420", "figure"),
    Output("graph-x", "figure"),
    Output("graph-q", "figure"),
    Output("graph-qualia", "figure"),
    Output("graph-drivers", "figure"),
    Output("graph-bands", "figure"),
    Output("reliability-box", "children"),
    Input("interval", "n_intervals"),
    Input("quality-thr", "value"),
    Input("show-debug-traces", "value"),
    Input("option-e-profile", "value"),
    Input("option-e-pe-tau", "value"),
    Input("validation-mode", "value"),
    Input("psd-hp-hz", "value"),
    Input("rel-window", "value"),
    Input("drift-aggr", "value"),
    Input("drift-warmup", "value"),
    Input("emit-ndjson", "value"),
)
def update_graphs(_, quality_thr, show_debug_value, option_e_profile_value, option_e_pe_tau_value, validation_mode, psd_hp_hz, rel_window, drift_aggr, drift_warmup, emit_ndjson_value):
    # NDJSON emit toggle (for local adaptive apps like Spotify controllers)
    emit_enabled = isinstance(emit_ndjson_value, (list, tuple, set)) and ("on" in emit_ndjson_value)
    prev_enabled = bool(global_state.get("emit_ndjson_enabled", False))
    global_state["emit_ndjson_enabled"] = emit_enabled
    ndjson_path = global_state.get("emit_ndjson_path", NDJSON_PATH)

    # Stream start/stop markers
    if emit_enabled and not prev_enabled:
        global_state["emit_ndjson_last_len"] = len(global_state["history"]["t"])
        append_ndjson_packet({
            "type": "stream_start",
            "schema_version": NDJSON_SCHEMA_VERSION,
            "baseline_version": BASELINE_VERSION,
            "code_version": CODE_VERSION,
            "t_wall": time.time(),
            "session_id": str(global_state.get("session_id", "")),
        }, path=ndjson_path)
    elif (not emit_enabled) and prev_enabled:
        append_ndjson_packet({
            "type": "stream_stop",
            "schema_version": NDJSON_SCHEMA_VERSION,
            "t_wall": time.time(),
            "session_id": str(global_state.get("session_id", "")),
        }, path=ndjson_path)
        global_state["emit_ndjson_last_len"] = len(global_state["history"]["t"])

    # Persist UI tuning knobs
    try:
        global_state["drift_aggr"] = float(drift_aggr) if drift_aggr is not None else DRIFT_AGGR_DEFAULT
    except Exception:
        global_state["drift_aggr"] = DRIFT_AGGR_DEFAULT
    try:
        global_state["warmup_ignore_drift_sec"] = float(drift_warmup) if drift_warmup is not None else WARMUP_IGNORE_DRIFT_SEC
    except Exception:
        global_state["warmup_ignore_drift_sec"] = WARMUP_IGNORE_DRIFT_SEC

    # Persist PSD high-pass knob (for drift/band gating only)
    try:
        hp = float(psd_hp_hz) if psd_hp_hz is not None else PSD_HP_HZ_DEFAULT
    except Exception:
        hp = PSD_HP_HZ_DEFAULT
    hp = float(np.clip(hp, 0.0, 2.0))
    prev_hp = float(global_state.get("psd_hp_hz", PSD_HP_HZ_DEFAULT))
    global_state["psd_hp_hz"] = hp
    # If hp changes, reset drift baseline to remain comparable
    if abs(hp - prev_hp) > 1e-6:
        global_state["drift_ref_ema"] = baseline_drift_ref_for_hp(hp)

    # Persist validation mode (for traceability)
    global_state["validation_mode"] = str(validation_mode or VALIDATION_MODE_DEFAULT)

    # Persist Option E UI (profile + tau)
    try:
        prof_val = str(option_e_profile_value or OPTION_E_PROFILE_DEFAULT)
    except Exception:
        prof_val = OPTION_E_PROFILE_DEFAULT
    if prof_val not in OPTION_E_PROFILE_KEYS:
        prof_val = OPTION_E_PROFILE_DEFAULT
    try:
        pe_tau_val = int(option_e_pe_tau_value) if option_e_pe_tau_value is not None else OPTION_E_PE_TAU_DEFAULT
    except Exception:
        pe_tau_val = OPTION_E_PE_TAU_DEFAULT
    pe_tau_val = int(max(1, min(10, pe_tau_val)))
    global_state["option_e_profile_id"] = prof_val
    global_state["option_e_pe_tau"] = pe_tau_val



    # Ensure connections
    if global_state.get("inlet_eeg") is None:
        connect_eeg("Muse")
        connect_horseshoe()

    inlet = global_state.get("inlet_eeg")

    # Pull EEG
    if inlet is not None:
        try:
            samples, ts = inlet.pull_chunk(timeout=0.0)
            if samples:
                arr = np.array(samples)  # (n_samples, n_channels)
                if arr.ndim == 2:
                    eeg_idx = global_state["ch_indices"]
                    if len(eeg_idx) >= 4 and arr.shape[1] > max(eeg_idx):
                        eeg = arr[:, eeg_idx[:4]].T  # (4, n_samples)
                    else:
                        eeg = arr[:, :4].T

                    n_new = eeg.shape[1]
                    for ch in range(4):
                        global_state["buffers"][ch].extend(eeg[ch].tolist())
                    global_state["ts"].extend(ts)

                    global_state["n_total_samples"] += n_new
                    global_state["last_eeg_wall"] = time.time()
        except Exception:
            pass

    # Compute metrics if enough data and enough new samples since last compute
    if all(len(b) >= WINDOW_N for b in global_state["buffers"]):
        n_total = global_state["n_total_samples"]
        if (n_total - global_state["last_compute_samples"]) >= int(FS_TARGET * STEP_SEC * 0.5):
            window = np.array([np.array(list(b)[-WINDOW_N:]) for b in global_state["buffers"]])
            ch_labels = global_state.get("ch_labels", ["ch0", "ch1", "ch2", "ch3"])[:4]

            t = time.time() - global_state["t0"]

            # Contact confidence: horseshoe if available, else infer
            contact_conf, contact_raw, contact_method = get_contact_conf_for_window(window)

            mets = compute_metrics_from_window(
                window_4ch=window,
                ch_labels_4=ch_labels,
                quality_thr=float(quality_thr),
                validation_mode=str(validation_mode),
                psd_hp_hz=float(psd_hp_hz),
                contact_conf=float(contact_conf),
                contact_raw=str(contact_raw),
                contact_method=str(contact_method),
                elapsed_s=float(t),
                option_e_profile_id=str(global_state.get("option_e_profile_id", OPTION_E_PROFILE_DEFAULT)),
                option_e_pe_tau=int(global_state.get("option_e_pe_tau", OPTION_E_PE_TAU_DEFAULT)),
            )

            global_state["last_compute_samples"] = n_total
            global_state["option_e_profile_id"] = mets.option_e_profile_id
            global_state["option_e_stats_src"] = mets.option_e_stats_src
            global_state["option_e_pe_tau"] = mets.option_e_pe_tau

            # Append RAW history
            t = time.time() - global_state["t0"]
            h = global_state["history"]
            h["t"].append(float(t))
            h["quality_thr"].append(float(quality_thr))
            h["drift_aggr"].append(float(global_state.get("drift_aggr", DRIFT_AGGR_DEFAULT)))
            h["warmup_ignore_drift_sec"].append(float(global_state.get("warmup_ignore_drift_sec", WARMUP_IGNORE_DRIFT_SEC)))

            # Traceability
            h["baseline_version"].append(BASELINE_VERSION)
            h["code_version"].append(CODE_VERSION)

            # Core
            h["X"].append(mets.X)
            h["R_abs"].append(mets.R_abs)

            # Qualia-ish / vibe options (A/B/C)
            h["R_meaningful"].append(mets.R_meaningful)
            h["R_focus_meaningful"].append(mets.R_focus_meaningful)
            h["Q_vibe"].append(mets.Q_vibe)
            h["Q_vibe_focus"].append(mets.Q_vibe_focus)
            h["Q_vibe_focus_E"].append(mets.Q_vibe_focus_E)

            h["R_meaningful_raw"].append(mets.R_meaningful_raw)
            h["R_focus_meaningful_raw"].append(mets.R_focus_meaningful_raw)
            h["Q_vibe_raw"].append(mets.Q_vibe_raw)
            h["Q_vibe_focus_raw"].append(mets.Q_vibe_focus_raw)
            h["Q_vibe_focus_E_raw"].append(mets.Q_vibe_focus_E_raw)
            h["Q_vibe_focus_E_mult"].append(mets.Q_vibe_focus_E_mult)

            # masked
            h["Q_abs"].append(mets.Q_abs)
            h["Q_perX"].append(mets.Q_perX)
            h["R_focus"].append(mets.R_focus)
            h["Q_focus"].append(mets.Q_focus)

            # raw
            h["Q_abs_raw"].append(mets.Q_abs_raw)
            h["Q_perX_raw"].append(mets.Q_perX_raw)
            h["Q_focus_raw"].append(mets.Q_focus_raw)

            h["JS_dist"].append(mets.JS_dist)
            h["phi"].append(mets.phi)
            h["quality_conf"].append(mets.quality_conf)
            h["valid"].append(bool(mets.valid))
            h["qualia_valid"].append(bool(mets.qualia_valid))
            h["reason_codes"].append(mets.reason_codes)

            # 12.3 provenance
            h["artifact_quality"].append(mets.artifact_quality)
            h["artifact_overlap"].append(mets.artifact_overlap)
            h["phi_meaningful"].append(mets.phi_meaningful)
            h["phi_artifact"].append(mets.phi_artifact)
            h["provenance"].append(mets.provenance)
            h["option_e_profile_id"].append(mets.option_e_profile_id)
            h["option_e_stats_src"].append(mets.option_e_stats_src)
            h["option_e_pe_tau"].append(mets.option_e_pe_tau)

            # Drivers
            for k in [
                "theta",
                "alpha",
                "beta",
                "gamma",
                "total_bands",
                "theta_n",
                "alpha_n",
                "beta_n",
                "gamma_n",
                "emg_index",
                "w_emg",
                "bg_raw",
                "E",
                "gamma_scaled",
                "a",
                "b",
                "k",
                "SAC_raw",
                "SAC_focus",
                "ch_corr",
                "ch_conf_raw",
                "ch_conf_used",
                "band_conf",
                "drift_mass",
                "drift_over",
                "drift_conf",
                "blink_rate",
                "blink_conf",
                "contact_conf",
                "contact_conf_used",
                "contact_raw",
                "contact_method",
                "C_pe",
                "C_pe_z",
                "C_pe_n",
                "S_aperiodic_slope",
                "S_flat",
                "S_aperiodic_slope_z",
                "S_aperiodic_slope_n",
            ]:
                h[k].append(getattr(mets, k))


            # Emit NDJSON packet (only once per new compute)
            if global_state.get("emit_ndjson_enabled", False):
                last_len = int(global_state.get("emit_ndjson_last_len", 0))
                cur_len = len(h["t"])
                if cur_len > last_len:
                    i_emit = cur_len - 1
                    pkt = build_state_layer_packet_from_history(h, i_emit, global_state.get("session_id", ""))
                    append_ndjson_packet(pkt, path=global_state.get("emit_ndjson_path", NDJSON_PATH))
                    global_state["emit_ndjson_last_len"] = cur_len

            # Update session-adaptive drift reference (EMA) when artifacts look clean.
            # This helps avoid false "high_drift" if a user's baseline drift mass differs from the global baseline.
            try:
                prev_ref = float(global_state.get("drift_ref_ema", DRIFT_REF))
                warmup_ignore = float(global_state.get("warmup_ignore_drift_sec", WARMUP_IGNORE_DRIFT_SEC))
                alpha_ref = DRIFT_REF_EMA_ALPHA_WARMUP if float(t) < warmup_ignore else DRIFT_REF_EMA_ALPHA
                clean = float(np.clip(float(mets.w_emg) * float(mets.blink_conf) * float(mets.contact_conf_used) * float(mets.ch_conf_used), 0.0, 1.0))
                if clean >= float(DRIFT_REF_UPDATE_MIN_CLEAN):
                    global_state["drift_ref_ema"] = float((1.0 - alpha_ref) * prev_ref + alpha_ref * float(mets.drift_mass))
            except Exception:
                pass


            # Update EMA (DISPLAY ONLY)
            def upd_ema(key: str, val: float, alpha: float):
                if not np.isfinite(val):
                    return global_state["ema"][key]
                prev = global_state["ema"][key]
                if prev is None or (not np.isfinite(prev)):
                    global_state["ema"][key] = float(val)
                else:
                    global_state["ema"][key] = float(alpha * val + (1 - alpha) * prev)
                return global_state["ema"][key]

            ema_X = upd_ema("X", mets.X, EMA_ALPHA_X)
            ema_Qabs = upd_ema("Q_abs", mets.Q_abs if np.isfinite(mets.Q_abs) else np.nan, EMA_ALPHA_Q)
            ema_Qfx = upd_ema("Q_focus", mets.Q_focus if np.isfinite(mets.Q_focus) else np.nan, EMA_ALPHA_Q)
            ema_Rabs = upd_ema("R_abs", mets.R_abs, EMA_ALPHA_Q)

            ema_Rm = upd_ema("R_meaningful", mets.R_meaningful if np.isfinite(mets.R_meaningful) else np.nan, EMA_ALPHA_Q)
            ema_Rfm = upd_ema("R_focus_meaningful", mets.R_focus_meaningful if np.isfinite(mets.R_focus_meaningful) else np.nan, EMA_ALPHA_Q)
            ema_Qv = upd_ema("Q_vibe", mets.Q_vibe if np.isfinite(mets.Q_vibe) else np.nan, EMA_ALPHA_Q)
            ema_Qvf = upd_ema("Q_vibe_focus", mets.Q_vibe_focus if np.isfinite(mets.Q_vibe_focus) else np.nan, EMA_ALPHA_Q)
            ema_Qvfe = upd_ema("Q_vibe_focus_E", mets.Q_vibe_focus_E if np.isfinite(mets.Q_vibe_focus_E) else np.nan, EMA_ALPHA_Q)

            ema_qc = upd_ema("quality_conf", mets.quality_conf, EMA_ALPHA_DRV)
            ema_phi = upd_ema("phi", mets.phi, EMA_ALPHA_DRV)
            ema_js = upd_ema("JS_dist", mets.JS_dist, EMA_ALPHA_DRV)
            ema_wemg = upd_ema("w_emg", mets.w_emg, EMA_ALPHA_DRV)
            ema_E = upd_ema("E", mets.E, EMA_ALPHA_DRV)
            ema_aq = upd_ema("artifact_quality", mets.artifact_quality, EMA_ALPHA_DRV)

            he = global_state["history_ema"]
            he["ema_X"].append(ema_X if ema_X is not None else np.nan)
            he["ema_Q_abs"].append(ema_Qabs if ema_Qabs is not None else np.nan)
            he["ema_Q_focus"].append(ema_Qfx if ema_Qfx is not None else np.nan)
            he["ema_R_abs"].append(ema_Rabs if ema_Rabs is not None else np.nan)

            he["ema_R_meaningful"].append(ema_Rm if ema_Rm is not None else np.nan)
            he["ema_R_focus_meaningful"].append(ema_Rfm if ema_Rfm is not None else np.nan)
            he["ema_Q_vibe"].append(ema_Qv if ema_Qv is not None else np.nan)
            he["ema_Q_vibe_focus"].append(ema_Qvf if ema_Qvf is not None else np.nan)
            he["ema_Q_vibe_focus_E"].append(ema_Qvfe if ema_Qvfe is not None else np.nan)

            he["ema_quality_conf"].append(ema_qc if ema_qc is not None else np.nan)
            he["ema_phi"].append(ema_phi if ema_phi is not None else np.nan)
            he["ema_JS_dist"].append(ema_js if ema_js is not None else np.nan)
            he["ema_w_emg"].append(ema_wemg if ema_wemg is not None else np.nan)
            he["ema_E"].append(ema_E if ema_E is not None else np.nan)
            he["ema_artifact_quality"].append(ema_aq if ema_aq is not None else np.nan)

    # Build figures
    h = global_state["history"]
    he = global_state["history_ema"]

    t = h["t"]

    
    # Raw Q mini graph (last 420s)
    fig_qraw420 = go.Figure()
    if len(t) > 0:
        t_arr = np.asarray(t, dtype=float)

        # Ensure non-negative window
        t_end = float(t_arr[-1])
        t_min = max(float(t_arr[0]), t_end - 420.0)
        mask_420 = t_arr >= t_min

        def _slice_series(series):
            # Align lengths defensively
            n = len(t_arr)
            if series is None:
                return t_arr[mask_420], np.full(np.count_nonzero(mask_420), np.nan, dtype=float)
            if len(series) != n:
                m = min(len(series), n)
                t_use = t_arr[-m:]
                s_use = np.asarray(series[-m:], dtype=float)
                t_end2 = float(t_use[-1])
                t_min2 = max(float(t_use[0]), t_end2 - 420.0)
                mask2 = t_use >= t_min2
                return t_use[mask2], s_use[mask2]
            s_arr = np.asarray(series, dtype=float)
            return t_arr[mask_420], s_arr[mask_420]

        tx, q_abs = _slice_series(h.get("Q_abs_raw", []))
        _, q_vibe = _slice_series(h.get("Q_vibe_raw", []))
        _, q_vf = _slice_series(h.get("Q_vibe_focus_raw", []))

        fig_qraw420.add_trace(go.Scatter(x=tx, y=q_abs, name="Q_abs_raw", line=dict(width=2)))
        fig_qraw420.add_trace(go.Scatter(x=tx, y=q_vibe, name="Q_vibe_raw", line=dict(dash="dot", width=2)))
        fig_qraw420.add_trace(go.Scatter(x=tx, y=q_vf, name="Q_vibe_focus_raw", line=dict(dash="dashdot", width=2)))

    fig_qraw420.update_layout(template="plotly_dark", height=260, margin=dict(l=30, r=10, t=20, b=25))
# X graph
    fig_x = go.Figure()
    fig_x.add_trace(go.Scatter(x=t, y=h["X"], name="X (raw)"))
    if len(he["ema_X"]) == len(t) and len(t) > 0:
        fig_x.add_trace(go.Scatter(x=t, y=he["ema_X"], name="X (EMA)", line=dict(dash="dot")))
    fig_x.update_layout(template="plotly_dark", height=260, margin=dict(l=30, r=10, t=20, b=25))


    # Plot mode
    show_debug = isinstance(show_debug_value, (list, tuple, set)) and ("on" in show_debug_value)

    # Q / Richness graph (RAW-first; masked/EMA optional for debugging)
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(x=t, y=h["Q_abs_raw"], name="Q_abs_raw (unmasked)", line=dict(width=2)))
    fig_q.add_trace(go.Scatter(x=t, y=h["Q_focus_raw"], name="Q_focus_raw (unmasked)", line=dict(dash="dash", width=2)))
    fig_q.add_trace(go.Scatter(x=t, y=h["Q_vibe_raw"], name="Q_vibe_raw (C, unmasked)", line=dict(dash="dot", width=2)))
    fig_q.add_trace(go.Scatter(x=t, y=h["Q_vibe_focus_raw"], name="Q_vibe_focus_raw (D, unmasked)", line=dict(dash="dashdot", width=2)))
    fig_q.add_trace(go.Scatter(x=t, y=h["Q_vibe_focus_E_raw"], name="Q_vibe_focus_E_raw (E, unmasked)", line=dict(dash="solid", width=1)))
    fig_q.add_trace(go.Scatter(x=t, y=h["Q_perX_raw"], name="Q_perX_raw (unmasked)", opacity=0.75, line=dict(width=1)))

    if show_debug:
        # masked (valid-only)
        fig_q.add_trace(go.Scatter(x=t, y=h["Q_abs"], name="Q_abs (valid)", opacity=0.80, line=dict(width=2)))
        fig_q.add_trace(go.Scatter(x=t, y=h["Q_focus"], name="Q_focus (valid)", opacity=0.80, line=dict(dash="dash", width=2)))
        fig_q.add_trace(go.Scatter(x=t, y=h["Q_vibe"], name="Q_vibe (C, valid)", opacity=0.80, line=dict(width=2)))
        if "Q_vibe_focus" in h:
            fig_q.add_trace(go.Scatter(x=t, y=h["Q_vibe_focus"], name="Q_vibe_focus (D, valid)", opacity=0.80, line=dict(dash="dashdot", width=2)))
        if "Q_vibe_focus_E" in h:
            fig_q.add_trace(go.Scatter(x=t, y=h["Q_vibe_focus_E"], name="Q_vibe_focus_E (E, qualia-valid)", opacity=0.80, line=dict(dash="longdash", width=2)))

        # richness / decomposition (debug)
        fig_q.add_trace(go.Scatter(x=t, y=h["R_abs"], name="R_abs (richness)", line=dict(dash="dot")))

        # EMA (masked)
        if len(he["ema_Q_abs"]) == len(t) and len(t) > 0:
            fig_q.add_trace(go.Scatter(x=t, y=he["ema_Q_abs"], name="Q_abs (EMA, valid)", line=dict(dash="dot")))
            fig_q.add_trace(go.Scatter(x=t, y=he["ema_Q_focus"], name="Q_focus (EMA, valid)", line=dict(dash="dot")))
            fig_q.add_trace(go.Scatter(x=t, y=he["ema_Q_vibe"], name="Q_vibe (EMA, valid)", line=dict(dash="dot")))
            if "ema_Q_vibe_focus" in he:
                fig_q.add_trace(go.Scatter(x=t, y=he["ema_Q_vibe_focus"], name="Q_vibe_focus (EMA, valid)", line=dict(dash="dot")))
            if "ema_Q_vibe_focus_E" in he:
                fig_q.add_trace(go.Scatter(x=t, y=he["ema_Q_vibe_focus_E"], name="Q_vibe_focus_E (EMA, valid)", line=dict(dash="dot")))
            fig_q.add_trace(go.Scatter(x=t, y=he["ema_R_abs"], name="R_abs (EMA)", line=dict(dash="dot")))

    fig_q.update_layout(template="plotly_dark", height=290, margin=dict(l=30, r=10, t=20, b=25))

    # Qualia Options graph (A/B/C/D; RAW-first; masked/EMA optional for debugging)
    fig_qualia = go.Figure()

    # Raw (unmasked)
    fig_qualia.add_trace(go.Scatter(x=t, y=h["R_meaningful_raw"], name="A: R_meaningful_raw (unmasked)", line=dict(width=2)))
    fig_qualia.add_trace(go.Scatter(x=t, y=h["R_focus_meaningful_raw"], name="B: R_focus_meaningful_raw (unmasked)", line=dict(dash="dash", width=2)))
    fig_qualia.add_trace(go.Scatter(x=t, y=h["Q_vibe_raw"], name="C: Q_vibe_raw (unmasked)", line=dict(dash="dot", width=2)))
    fig_qualia.add_trace(go.Scatter(x=t, y=h["Q_vibe_focus_raw"], name="D: Q_vibe_focus_raw (unmasked)", line=dict(dash="dashdot", width=2)))
    fig_qualia.add_trace(go.Scatter(x=t, y=h["Q_vibe_focus_E_raw"], name="E: Q_vibe_focus_E_raw (unmasked)", line=dict(dash="longdash", width=2)))

    if show_debug:
        # Masked (valid)
        fig_qualia.add_trace(go.Scatter(x=t, y=h["R_meaningful"], name="A: R_meaningful (valid)", opacity=0.85, line=dict(width=2)))
        fig_qualia.add_trace(go.Scatter(x=t, y=h["R_focus_meaningful"], name="B: R_focus_meaningful (valid)", opacity=0.85, line=dict(dash="dash", width=2)))
        fig_qualia.add_trace(go.Scatter(x=t, y=h["Q_vibe"], name="C: Q_vibe (valid)", opacity=0.85, line=dict(width=2)))
        if "Q_vibe_focus" in h:
            fig_qualia.add_trace(go.Scatter(x=t, y=h["Q_vibe_focus"], name="D: Q_vibe_focus (valid)", opacity=0.85, line=dict(dash="dashdot", width=2)))
        if "Q_vibe_focus_E" in h:
            fig_qualia.add_trace(go.Scatter(x=t, y=h["Q_vibe_focus_E"], name="E: Q_vibe_focus_E (qualia-valid)", opacity=0.85, line=dict(dash="longdash", width=2)))

        # EMA (valid)
        if len(he["ema_R_meaningful"]) == len(t) and len(t) > 0:
            fig_qualia.add_trace(go.Scatter(x=t, y=he["ema_R_meaningful"], name="A: R_meaningful (EMA, valid)", line=dict(dash="dot")))
            fig_qualia.add_trace(go.Scatter(x=t, y=he["ema_R_focus_meaningful"], name="B: R_focus_meaningful (EMA, valid)", line=dict(dash="dot")))
            fig_qualia.add_trace(go.Scatter(x=t, y=he["ema_Q_vibe"], name="C: Q_vibe (EMA, valid)", line=dict(dash="dot")))
            if "ema_Q_vibe_focus" in he:
                fig_qualia.add_trace(go.Scatter(x=t, y=he["ema_Q_vibe_focus"], name="D: Q_vibe_focus (EMA, valid)", line=dict(dash="dot")))
            if "ema_Q_vibe_focus_E" in he:
                fig_qualia.add_trace(go.Scatter(x=t, y=he["ema_Q_vibe_focus_E"], name="E: Q_vibe_focus_E (EMA, valid)", line=dict(dash="dot")))

    fig_qualia.update_layout(template="plotly_dark", height=280, margin=dict(l=30, r=10, t=20, b=25))

    # Drivers / quality graph
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=t, y=h["quality_conf"], name="quality_conf (raw)"))
    fig_d.add_trace(go.Scatter(x=t, y=h["artifact_quality"], name="artifact_quality (raw)", line=dict(dash="dot")))
    fig_d.add_trace(go.Scatter(x=t, y=h["phi"], name="phi (raw)"))
    fig_d.add_trace(go.Scatter(x=t, y=h["phi_meaningful"], name="phi_meaningful", opacity=0.85, line=dict(dash="dash")))
    fig_d.add_trace(go.Scatter(x=t, y=h["phi_artifact"], name="phi_artifact", opacity=0.65, line=dict(dash="dashdot")))

    fig_d.add_trace(go.Scatter(x=t, y=h["JS_dist"], name="JS_dist (raw)"))
    fig_d.add_trace(go.Scatter(x=t, y=h["w_emg"], name="w_emg (raw)"))
    fig_d.add_trace(go.Scatter(x=t, y=h["E"], name="E (raw)"))
    fig_d.add_trace(go.Scatter(x=t, y=h["contact_conf"], name="contact_conf (raw)", line=dict(dash="dash")))

    if len(he["ema_quality_conf"]) == len(t) and len(t) > 0:
        fig_d.add_trace(go.Scatter(x=t, y=he["ema_quality_conf"], name="quality_conf (EMA)", line=dict(dash="dot")))
        fig_d.add_trace(go.Scatter(x=t, y=he["ema_artifact_quality"], name="artifact_quality (EMA)", line=dict(dash="dot")))
        fig_d.add_trace(go.Scatter(x=t, y=he["ema_phi"], name="phi (EMA)", line=dict(dash="dot")))
        fig_d.add_trace(go.Scatter(x=t, y=he["ema_JS_dist"], name="JS_dist (EMA)", line=dict(dash="dot")))
        fig_d.add_trace(go.Scatter(x=t, y=he["ema_w_emg"], name="w_emg (EMA)", line=dict(dash="dot")))
        fig_d.add_trace(go.Scatter(x=t, y=he["ema_E"], name="E (EMA)", line=dict(dash="dot")))

    fig_d.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=30, r=10, t=20, b=25),
        shapes=[dict(
            type="line",
            x0=(t[0] if t else 0),
            x1=(t[-1] if t else 1),
            y0=float(quality_thr),
            y1=float(quality_thr),
            line=dict(dash="dash"),
        )],
    )

    # Bands & artifacts graph
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=t, y=h["theta"], name="theta"))
    fig_b.add_trace(go.Scatter(x=t, y=h["alpha"], name="alpha"))
    fig_b.add_trace(go.Scatter(x=t, y=h["beta"], name="beta"))
    fig_b.add_trace(go.Scatter(x=t, y=h["gamma"], name="gamma"))
    fig_b.add_trace(go.Scatter(x=t, y=h["drift_mass"], name="drift_mass (1–4Hz)", line=dict(dash="dot")))
    fig_b.add_trace(go.Scatter(x=t, y=h["blink_rate"], name="blink_rate", line=dict(dash="dash")))
    fig_b.update_layout(template="plotly_dark", height=300, margin=dict(l=30, r=10, t=20, b=25))

    # Add event vertical lines to all graphs
    for ev in global_state.get("events", []):
        ev_t = float(ev.get("t", 0.0))
        for fig in (fig_x, fig_q, fig_qualia, fig_d, fig_b):
            fig.add_vline(x=ev_t, line_width=1, line_dash="dot", opacity=0.6)

    rel_text = format_reliability(global_state.get("history", {}), rel_window)
    return fig_qraw420, fig_x, fig_q, fig_qualia, fig_d, fig_b, rel_text


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    os.makedirs("sessions", exist_ok=True)
    connect_eeg("Muse")
    connect_horseshoe()
    app.run_server(debug=True, host="127.0.0.1", port=PORT)