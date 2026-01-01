# Cortex Console v4.2 + Turrell Room v5.2
## Option E (Permutation Entropy + Aperiodic Slope) — Change Log + Metric Definitions

This document is a long-form, reusable reference for:

- **Cortex Console**: `app_v4_2_1_cortex_console_drift_ui_stream_ndjson_reliability_v4_2.py`
- **Turrell Room 2D**: `xq_turrell_room_2d_v5_2.py`

It’s designed to be dropped into your workspace so future chats can pick up context immediately.

---

## 1) Executive summary

### What stayed the same (critical compatibility promise)

- **Option D stays unchanged**. Your existing “single dial” candidate (vibe_focus) is not modified.
- **All existing Q/R/X metrics remain present** and behave the same **by default**.
- If the baseline JSON does **not** contain an `option_e` block, the system **does not crash** and Option E becomes **neutral** (multiplier ≈ 1.0).

### What’s new in v4.2

**Option E** adds two new *drivers* and one new *derived Q*:

- **C_pe**: permutation-entropy complexity (time-domain “microstructure/texture”).
- **S_aperiodic_slope**: log–log aperiodic slope of the unnormalized PSD (frequency-domain broadband “arousal tilt”).
  - Normalized via **flatness**: `S_flat = -S_aperiodic_slope`.
- **Q_vibe_focus_E**: a new metric that **enhances** Option D with a gentle multiplier derived from C_pe and S_flat.

In addition:

- **NDJSON state stream now exports `qualia_valid`** in `reliability.qualia_valid`.
- A **profile dropdown** (Option E baseline stats selection) was added.
- A **τ (tau) slider** (range 1–10) was added for Option E permutation entropy.
  - Tooltip is always visible; marks at **1 / 5 / 10**.
  - The slider is the live override (always wins).

### What’s new in Turrell v5.2

- Adds a new selectable `q_metric`: **`vibe_focus_e`**.
  - Reads **masked** `state.Q_vibe_focus_E` if present.
  - Reads **raw** `state.raw.Q_vibe_focus_E` if present.
- Treats `vibe_focus_e` as a **QUALIA metric** for auto validity gating.
- Optionally reads Option E drivers from NDJSON and uses them for subtle, performance-safe modulation:
  - `C_pe_n` influences **micro-texture gain**.
  - `S_aperiodic_slope_n` influences a tiny **warmth/exposure bias**.
- HUD includes `C_pe_n` and `S_n` if present.

---

## 2) v4.2 architecture recap (how it fits your X/Q state layer)

v4.2 continues your core architecture:

1. **Acquire EEG** (Muse LSL stream) into a rolling buffer.
2. Every tick, compute a **fixed-length analysis window** (your existing 5s windowing).
3. Compute core spectral features (PSD shape, JS distance, band masses), plus reliability (quality_conf, contact, drift, blink, EMG, etc.).
4. Compute metrics:
   - X (activation/drive)
   - R (richness-like term)
   - Q family (Q_abs / Q_focus / Q_perX and the qualia-style Q_vibe / Q_vibe_focus)
5. Emit:
   - **Dashboard plots**
   - **Session CSV/JSON**
   - **NDJSON “state layer” packets** (for Turrell + downstream tools)

Option E is intentionally **drop-in**:

- It runs inside the same window compute.
- It produces *additional* fields.
- It never changes your existing D-path computations.

---

## 3) Option E design goals (why it exists)

Your existing “meaningful vibe” stack (Option D) is already strong because it:

- builds a **stable spectral reference** baseline (`P_ref`) and computes **deviation** (`JS_dist → φ`),
- adds **engagement/drive** (X, E), and
- protects output with **qualia_valid** (artifact-quality gating) instead of “invalidating everything.”

Option E adds **two complementary axes**:

1) **Temporal microstructure / texture**
- Some experiential changes show up as “burstiness / irregular modulation / pattern diversity” even when PSD shape doesn’t move much.
- Permutation entropy is a simple, fast proxy for that time-ordering complexity.

2) **Broadband arousal tilt**
- The overall 1/f-like background (aperiodic component) can shift without obvious “band fraction” changes.
- A log–log slope fit provides a separate broadband axis.

Key constraint: **Muse artifacts can mimic both** (especially EMG), so Option E is built to:

- use your **existing artifact machinery**, and
- *soften toward neutral* when artifact_quality is low.

---

## 4) Option E baseline schema + profile selection

v4.2 reads Option E configuration from the baseline JSON **if present**, without requiring it.

### 4.1 Optional baseline structure

At load time:

- `option_e_cfg = baseline.get("option_e", {})`
- `option_e_defaults = option_e_cfg.get("defaults", {})`

Supported optional keys:

**Defaults** (tuning + parameters)
- `wC` (default 0.15)
- `wS` (default 0.12)
- `mult_clip` (default [0.70, 1.35])
- `pe_m` (default 5)
- `pe_tau` (default 1)
- `pe_band_hz` (default [8, 30])
- `slope_fit_hz` (default [2, 45])
- `slope_exclude_hz` (default [[8, 13]])

**Stats**
- Top-level fallbacks:
  - `option_e.C_pe = {mu, sigma}`
  - `option_e.S_flat = {mu, sigma}`  *(flatness = -slope)*
- Per-profile stats:
  - `option_e.profiles[profile_id].C_pe = {mu, sigma}`
  - `option_e.profiles[profile_id].S_flat = {mu, sigma}`

### 4.2 Profile selection precedence

The active profile id is driven by:

1) **Dashboard dropdown** (live) → **wins**
2) `CORTEX_PROFILE_ID` environment variable (startup default seed)
3) Fallback to `"global"` or the first available profile key

The script records:

- `option_e_profile_id` (which profile id is selected)
- `option_e_stats_src` (where stats were resolved from: profile/global/top/none)

### 4.3 Tau selection precedence (explicitly requested behavior)

Tau (PE delay) is driven by:

1) **Dashboard τ slider** (live override) → **always wins**
2) `CORTEX_PE_TAU` environment variable (seed)
3) baseline defaults `option_e.defaults.pe_tau`
4) hard fallback `1`

Tau is clamped to **1..10**.

The per-sample effective tau is stored as `option_e_pe_tau`.

---

## 5) Option E computation details (formulas + algorithms)

### 5.1 Shared window assumptions

Option E computes once per analysis tick using the same window you already compute for PSD / JS / band masses.

Inputs (already in your pipeline):

- `window_4ch`: 4-channel EEG window (shape ~ 4 × (FS×window_sec))
- `artifact_quality` (0..1)
- reliability hints:
  - `w_emg`, `blink_conf`, `drift_conf`, `contact_conf_used` (already computed)

Outputs (new):

- `C_pe` (0..1)
- `S_aperiodic_slope` (typically negative)
- `S_flat = -S_aperiodic_slope` (positive-ish)
- normalized:
  - `C_pe_z`, `C_pe_n`
  - `S_aperiodic_slope_z`, `S_aperiodic_slope_n` *(computed on flatness, stored under slope names)*
- multiplier:
  - `Q_vibe_focus_E_mult`
- new Q:
  - `Q_vibe_focus_E_raw`
  - `Q_vibe_focus_E` (masked by qualia_valid)

### 5.2 C_pe: Permutation entropy on band-limited EEG

**Goal:** capture temporal “microstructure/texture” beyond PSD shape.

#### Step A — bandpass filter (precomputed)

A Butterworth bandpass is precomputed once:

- Order: 4
- Band: default 8–30 Hz (configurable via baseline `pe_band_hz`)
- Form: SOS for stability

`OPTION_E_SOS = butter(4, [lo, hi], btype="bandpass", fs=FS_TARGET, output="sos")`

#### Step B — per-channel standardization

For each channel `x(t)`:

1) `y = sosfiltfilt(OPTION_E_SOS, x)`
2) z-score:

`z(t) = (y(t) - mean(y)) / (std(y) + eps)`

This makes PE more about **shape/time ordering** than amplitude.

#### Step C — ordinal patterns

Given parameters:

- embedding dimension `m` (default 5)
- delay `τ` (tau) (default 1, slider 1..10)

Create sequences:

`segment_i = [ z[i], z[i+τ], z[i+2τ], ..., z[i+(m-1)τ] ]`

For each segment, compute its permutation pattern via argsort.

**Tie handling (deterministic):**

To avoid ambiguous ranks when values are equal (or near-equal), a tiny index-based perturbation is used:

`order = argsort(segment + eps_tie * arange(m))`

#### Step D — entropy + normalization

Let `p(π)` be frequency of each ordinal pattern π.

Shannon entropy:

`H = - Σ p_i · log(p_i + eps)`

Normalize by maximum entropy:

`Hmax = log( factorial(m) )`

**Normalized permutation entropy:**

`C_pe = clip( H / Hmax, 0, 1 )`

Finally average across channels:

`C_pe = mean(C_pe_channel)`

### 5.3 S_aperiodic_slope: log–log slope of unnormalized PSD

**Goal:** capture broadband spectral tilt (“global arousal axis”).

#### Step A — PSD input

Use the **unnormalized** PSD power per frequency bin (mean across channels):

- compute per-channel Welch PSD `P_ch(f)`
- average: `P_raw(f) = mean_ch P_ch(f)`

#### Step B — fit range + exclude bands

Default fit range:

- `fit_hz = [2, 45]` (configurable)

Exclude alpha peak region:

- `exclude_hz = [[8, 13]]`

Mask definition:

- include only `f` within fit range
- exclude any `f` inside excluded bands
- exclude bins with non-positive power

#### Step C — OLS fit

Fit a line in log–log space:

`log10(P_raw(f)) = slope * log10(f) + intercept`

Return:

- `S_aperiodic_slope = slope` (typically negative)

Define flatness:

- `S_flat = -S_aperiodic_slope`

### 5.4 Normalization to (z, n) with safe neutral fallback

Sigmoid:

`sigmoid(z) = 1 / (1 + exp(-z))`

#### C_pe normalization

If baseline stats exist:

- `C_pe_z = (C_pe - mu_C) / (sigma_C + 1e-6)`
- `C_pe_n = sigmoid(C_pe_z)`

If missing stats:

- `C_pe_z = 0`
- `C_pe_n = 0.5` *(neutral)*
- Add reason code: `option_e_missing_baseline`

#### Slope normalization (via flatness)

Compute:

- `S_flat = -S_aperiodic_slope`

If baseline stats exist for S_flat:

- `S_flat_z = (S_flat - mu_Sflat) / (sigma_Sflat + 1e-6)`
- `S_aperiodic_slope_n = sigmoid(S_flat_z)`
- `S_aperiodic_slope_z = S_flat_z`

If missing stats:

- `S_flat_z = 0`
- `S_aperiodic_slope_n = 0.5`
- Add reason code: `option_e_missing_baseline`

### 5.5 Artifact-aware softening (critical for Muse)

Blend toward neutral using artifact_quality:

- `C_eff = 0.5 + artifact_quality * (C_pe_n - 0.5)`
- `S_eff = 0.5 + artifact_quality * (S_aperiodic_slope_n - 0.5)`

### 5.6 Multiplier and new Q

Weights:

- `wC` default 0.15
- `wS` default 0.12

Clip bounds:

- `mult_clip` default `[0.70, 1.35]`

Compute:

`mult_E = clip( 1 + wC * (2*(C_eff - 0.5)) + wS * (2*(S_eff - 0.5)), lo, hi )`

Apply:

- `Q_vibe_focus_E_raw = Q_vibe_focus_raw * mult_E`
- `Q_vibe_focus_E = Q_vibe_focus_E_raw if qualia_valid else NaN`

### 5.7 Reason codes (Option E additions)

- `option_e_missing_baseline`

Artifact-likely spike annotations:

- `C_spike_artifact_likely:<hints>` when `C_pe_z > 2.5` and artifact hints exist
- `S_flat_artifact_likely:<hints>` when `S_flat_z > 2.5` and artifact hints exist

---

## 6) v4.2 data model changes (history + NDJSON)

### 6.1 New history keys

Option E drivers:
- `C_pe`, `C_pe_z`, `C_pe_n`
- `S_aperiodic_slope`, `S_flat`, `S_aperiodic_slope_z`, `S_aperiodic_slope_n`

Option E Q outputs:
- `Q_vibe_focus_E_raw`
- `Q_vibe_focus_E` (masked)
- `Q_vibe_focus_E_mult`

Traceability / UI state:
- `option_e_profile_id`
- `option_e_stats_src`
- `option_e_pe_tau`

### 6.2 NDJSON export additions (safe + additive)

- `reliability.qualia_valid`
- `state.Q_vibe_focus_E` (masked)
- `state.raw.Q_vibe_focus_E` (raw)
- `state.raw.Q_vibe_focus_E_mult`
- `state.raw.C_pe`, `C_pe_z`, `C_pe_n`
- `state.raw.S_aperiodic_slope`, `S_aperiodic_slope_z`, `S_aperiodic_slope_n`
- `meta.option_e_profile_id`, `meta.option_e_stats_src`, `meta.pe_tau`

---

## 7) v4.2 dashboard changes

### 7.1 New Option E controls

- Profile dropdown (`option-e-profile`)
- Tau slider (`option-e-pe-tau`)
  - Label: “Option E: Permutation Entropy τ (tau)”
  - marks 1/5/10
  - tooltip always visible
  - live update (drag)

### 7.2 Plots

Qualia graph adds:
- `Q_vibe_focus_E_raw`
- `Q_vibe_focus_E` (qualia masked)
- EMA of masked

---

## 8) Turrell Room v5.2 changes

### 8.1 New q_metric: `vibe_focus_e`

- masked: `state["Q_vibe_focus_E"]`
- raw: `state["raw"]["Q_vibe_focus_E"]`

Included in:
- M-cycle list
- allowed metric validation

### 8.2 Qualia gating behavior (auto validity)

Qualia metrics include:
- `vibe`, `vibe_focus`, `vibe_focus_e`, `meaningful`, `focus_meaningful`

Auto gating uses:
- `reliability.qualia_valid` (preferred)
- fallback to artifact_quality threshold if missing

### 8.3 Option E driver ingestion (optional)

Reads:
- `C_pe_n = state.raw.C_pe_n`
- `S_n = state.raw.S_aperiodic_slope_n`

Missing → defaults to neutral and behaves like v5.1.

### 8.4 Secondary modulation (subtle, performance-safe)

Let:
- `c = clamp(C_pe_n, 0, 1)` else 0.5
- `s = clamp(S_n, 0, 1)` else 0.5

Compute:
- `micro_texture_gain = clamp(0.85 + 0.30*(c - 0.5), 0.70, 1.20)`
- `warmth_bias = clamp(0.06*(s - 0.5), -0.05, +0.05)`

Apply:
- micro_texture_gain → noise/dither layer only
- warmth_bias → tiny palette/exposure bias only

### 8.5 HUD additions

HUD shows:
- `C_pe_n` and `S_n` if present else “--”
- plus existing q_metric/q_source/validity/preset

---

## 9) Practical notes

- Option E is an **enhancer**, not a replacement for Option D.
- PE and slope are setup-dependent → per-user Option E baseline stats are recommended.
- If slope feels EMG-sensitive, reduce fit band (e.g., 2–35 Hz) via baseline defaults.

---

## 10) Quick reference: key names

### Cortex Console v4.2 — new fields

NDJSON:
- `reliability.qualia_valid`
- `state.Q_vibe_focus_E`
- `state.raw.Q_vibe_focus_E`
- `state.raw.Q_vibe_focus_E_mult`
- `state.raw.C_pe`, `C_pe_z`, `C_pe_n`
- `state.raw.S_aperiodic_slope`, `S_aperiodic_slope_z`, `S_aperiodic_slope_n`
- `meta.option_e_profile_id`, `meta.option_e_stats_src`, `meta.pe_tau`

### Turrell v5.2 — new metric

- `--q-metric vibe_focus_e`
- Hotkey `M` cycles through it
