# ALMA OS — Unified App (Cortex Engine + 5 Products)

This doc is the **single source of truth** for building ALMA OS in Cursor and continuing in future chats.

---

## Core Principles (Non‑Negotiables)

1. **One app experience**
   - One command starts the unified Dash app.
   - The app can **start/stop muselsl**, **start/stop the state engine**, **toggle NDJSON emit**, and **launch/stop Turrell**.

2. **State Layer contract stays stable**
   - NDJSON remains the interoperability bus between Cortex Engine and Turrell.
   - SQLite is the persistence layer for Readiness / Scheduler / Memory / Recipes.

3. **Reliability exposure policy**
   - Reliability/validity details show **only** on **Home + Neurometrics**.
   - Product pages consume **masked canonical metrics** and bucket aggregates.

4. **Canonical Q (for products)**
   - Default canonical Q = **Q_vibe_focus**.
   - Neurometrics only: dropdown to choose displayed Q:
     - `vibe_focus` (default)
     - `vibe` (awe/immersion)
     - `abs` (fundamental qualia×X axis)
     - `vibe_focus_e` (experimental; only after baseline-per-user work)

5. **Environment OS scope (for now)**
   - **Only visuals actuate**.
   - Spotify integration is **tracking only**: log what’s playing + timestamps.
   - Turrell:
     - Use **OFF palette family** for all modes.
     - Modes only change **style parameters** (bloom/exposure/drift/softness), not palette family.
     - Always run with `--no-freeze-on-invalid`.
     - Support display selection **Display 0 / Display 1**.
     - Aim for **native / 4K highest quality**.

---

## System Architecture (Practical)

### Processes

**A) Dash App (ALMA OS)**
- Hosts pages + controls.
- Owns lifecycle:
  - Start/stop muselsl (supervised)
  - Start/stop state engine thread
  - Toggle NDJSON emit
  - Launch/stop Turrell subprocess

**B) State Engine (thread inside Dash process)**
- LSL inlet
- rolling buffers
- 5s window compute
- reliability logic
- appends in‑memory ring history
- optionally emits NDJSON stream
- writes bucket aggregates + events + spotify now-playing to SQLite

**C) Turrell (separate pygame process)**
- tails NDJSON
- reads adaptive_mode.json
- renders fullscreen to chosen display

### Data Stores

**NDJSON stream**
- `sessions/current/state_stream.ndjson`

**Mode file**
- `sessions/current/adaptive_mode.json`

**SQLite**
- `data/alma.db`

---

## UI Structure (Tabs / Pages)

### 1) Home (Dashboard)
- LSL status + dot
- Engine status
- muselsl PID status
- **Reliability (rolling last 600)**
- Current X + canonical Q
- “Now Playing” (Spotify tracking)
- Quick controls:
  - Start/Stop Muse Stream
  - Start/Stop Engine
  - Toggle NDJSON Emit
  - Launch/Stop Turrell
  - Mark Event (bookmark)

### 2) Neurometrics
- Full diagnostic plots
- Q dropdown (vibe_focus / vibe / abs / vibe_focus_e)
- Shows raw plots (including **Q raw last 420s**) and X

### 3) Readiness Map
- Heatmap timeline day view
- buckets: 5–15 min windows
- Labels (MVP): Deep Work / Ideation / Recovery / Social
- “Why panel” explains label using **features**, not QC internals

### 4) Scheduler (Life OS)
- Define flexible blocks
- Suggest swaps based on readiness windows
- Advice-only MVP

### 5) Memory (State‑Indexed Memory)
- Search: “find similar state”
- Uses bucket feature vector similarity
- Shows events + now-playing context

### 6) Environment
- Turrell controls
  - display selector (0/1)
  - quality selector (native/4k)
  - style/mode selector (OFF palettes always)
  - launch/stop

### 7) Recipes
- CRUD recipes
- Targets: X/Q bands + stability
- “Apply now” creates event + sets Environment style

### 8) Settings
- profile selection
- baseline config
- paths

---

## Shared Feature Vector (Buckets)

Used by Readiness/Scheduler/Memory.

For each bucket (e.g., 10 minutes):
- `mean_X`
- `mean_Q` (canonical)
- `std_Q`
- `Q_slope` (linear trend)
- `valid_frac`
- optional: `mean_focus`, `mean_quality_conf`

Labels (MVP thresholds; tweak later):
- **Deep Work:** mean_X high, std_Q low, valid_frac high
- **Ideation:** std_Q high and Q_slope positive
- **Recovery:** mean_X low and std_Q low
- **Social/Engagement:** mean_Q elevated, std_Q moderate

---

# Current Build Status (Repo)

Repo exists and runs.
- LSL connects.
- Engine runs.
- Turrell launches (but visuals currently broken).
- Reliability rolling shows values.

---

# Current Bugs & Fix Plan

## Bug A — Baseline not loading → X too low

**Symptom:** baseline_global_muse_v1_revised.json isn’t being picked up; X collapses.

**Most common causes:**
1) baseline file isn’t at the path the engine is resolving
2) relative paths resolving against the wrong root
3) baseline loads only at import/init and never reloads after you add the file
4) profile baseline_path differs from default

**Required repo placement**
- Create `baselines/`
- Put file at: `baselines/baseline_global_muse_v1_revised.json`
- In `profiles/default.json` set:
  - `"baseline_path": "baselines/baseline_global_muse_v1_revised.json"`

**Engineering fix (make this bulletproof):**
- In `load_baseline()`:
  - expanduser
  - try candidates:
    - baseline_path as absolute
    - ROOT_DIR / baseline_path
    - cwd / baseline_path
  - store `BASELINE_PATH_USED` + `BASELINE_LOADED=True/False` + warning string

**UI fix:**
- Show on Home (small text):
  - `Baseline: LOADED (path)` or `Baseline: FALLBACK (warning)`

**Engine fix:**
- On `Start Engine`, call baseline load again (so adding the file doesn’t require a full app restart).

---

## Bug B — Turrell: no HUD + brightness max + “static” feel

**Root causes (likely):**
1) Turrell runner not passing `--hud` so overlay is off
2) mode defaults to OFF and OFF render is very subtle, so it looks static
3) NDJSON path mismatch OR NDJSON emit is off → Turrell isn’t getting updates
4) exposure/bloom clamps too high

**Fix checklist:**
- In `turrell_runner.py` add `--hud` to the subprocess command.
- Ensure runner passes correct file paths:
  - `--ndjson sessions/current/state_stream.ndjson`
  - `--mode-file sessions/current/adaptive_mode.json`
- Ensure Dash Start Turrell also ensures NDJSON emit is ON (or warns if OFF).
- In Turrell script:
  - keep `--no-freeze-on-invalid`
  - add gentle time‑drift even in OFF mode (micro motion)
  - clamp exposure/bloom to avoid full white saturation

**Display & Quality:**
- Runner must pass display index (0/1).
- Quality selector should default to **native**; if forcing 4K, ensure fullscreen uses display’s resolution.

---

# Cursor Execution Plan (Phased)

## Phase 1 — Stabilize Core Loop (Now)
1) Fix baseline loading + add UI indicator
2) Fix Turrell runner HUD + ensure NDJSON/mode paths match
3) Confirm Turrell responds to canonical Q and does not saturate

## Phase 2 — Persistence & Buckets
4) Create SQLite tables + write buckets
5) Readiness page uses buckets

## Phase 3 — Memory + Bookmarks
6) Events table (manual + auto)
7) Similarity search over buckets

## Phase 4 — Scheduler + Recipes MVP
8) Rule-based suggestions
9) Recipe CRUD + Apply

## Phase 5 — Spotify Tracking (Not adapting)
10) Poll now-playing and log to DB + show on Home

---

# “Super Prompts” for Cursor Agent (Phase 1)

## Prompt 1 — Baseline: make loading bulletproof + visible
"""
Open `alma/engine/cortex_core.py` (or wherever `load_baseline` lives).

Implement a robust baseline loader:
- Baseline file should live at `baselines/baseline_global_muse_v1_revised.json`.
- Profile key `baseline_path` can override it.
- Resolve paths by trying:
  1) absolute path
  2) ROOT_DIR / relative
  3) Path.cwd() / relative
- Use Path(...).expanduser().

Store globals:
- BASELINE_LOADED (bool)
- BASELINE_PATH_USED (str or None)
- BASELINE_WARNING (str or None)

Ensure the engine calls baseline load on **Start Engine** (not only at import time).

Add a small status field to the engine snapshot so UI can display baseline status.
"""

## Prompt 2 — Home UI: show baseline status
"""
Update the Home page to display:
- Baseline status line: "Baseline: LOADED <path>" or "Baseline: FALLBACK <warning>".

Make sure it updates on the same interval used for the other status fields.
"""

## Prompt 3 — Turrell runner: restore HUD + sanity defaults
"""
Open `alma/ui/pages/environment.py` and `alma/turrell_runner.py`.

1) Add `--hud` to the Turrell subprocess command.
2) Verify the runner passes:
   - --ndjson sessions/current/state_stream.ndjson
   - --mode-file sessions/current/adaptive_mode.json
   - --no-freeze-on-invalid
   - --display {0|1}
   - quality default native/high

3) If NDJSON emit is OFF, Environment page should warn and offer to enable it.
"""

## Prompt 4 — Turrell: stop whiteout + add motion in OFF
"""
Open `external/xq_turrell_room_2d_v5_3_style_modes.py`.

- Keep OFF palette family for all modes.
- Ensure OFF mode has subtle but visible continuous motion (slow drift in portal or gradient).
- Clamp exposure and bloom so it never saturates to pure white.
- Keep HUD render.

Do not break:
- --no-freeze-on-invalid
- display selection
- quality selection
"""

---

# Debug Commands (When Things Feel Wrong)

- Confirm NDJSON is being written:
  - `ls -lh sessions/current/state_stream.ndjson`
  - `tail -n 3 sessions/current/state_stream.ndjson`

- Confirm mode file is being written:
  - `cat sessions/current/adaptive_mode.json`

- Confirm baseline exists:
  - `ls -lh baselines/baseline_global_muse_v1_revised.json`

---

# Notes on Grok Suggestions (Accounted For)

- Threaded engine separation ✅
- Keep NDJSON as state bus ✅
- SQLite persistence layer ✅
- Canonical Q metric selection ✅ (but default is Q_vibe_focus, Option-E is experimental)
- UI dark neon theme ✅
- Global hotkey bookmarks planned ✅

