# ALMA OS (Unified Local App) — Complete Updated Plan (v0.1)

This is the **single source of truth** for building the unified local app that contains:

1) **Daily Readiness Map** (state heatmap)
2) **Life OS Scheduler** (suggest + explain)
3) **State‑Indexed Memory** (“when did I feel this state last?”)
4) **Environment OS** (Neuroadaptive Rooms — Turrell Room first)
5) **State Recipes** (repeatable protocols → target bands)

Built on your existing stack:
- **Cortex Console v4.2.1** (Dash + state engine + NDJSON emitter)
- **Turrell Room 2D v5.2** (Pygame subscriber reading NDJSON + mode file)

---

## 0) Non‑negotiables / invariants

### 0.1 Instrument truthfulness
- **Reliability stays internal**: Only shown on **Home / Neurometrics**.
- Products (Readiness/Scheduler/Memory/Recipes/Environment) should consume **masked metrics + aggregates**.

### 0.2 Preserve baseline engine behaviors
- **Never hard‑invalidate due to channel agreement** (channel agreement remains diagnostic only).
- Keep the **Q_raw last‑420s** graph in Neurometrics.
- Do not regress NDJSON schema and downstream compatibility.

### 0.3 Canonical Q (product‑facing)
- **Default canonical Q for the product UI: `Q_vibe_focus`**.
- In **Neurometrics only**, allow a dropdown:
  - `vibe_focus` *(default)*
  - `vibe` *(more awe/immersion)*
  - `abs` *(fundamental qualia×X axis)*
  - `vibe_focus_e` *(experimental; requires per‑user baseline, optional)*

### 0.4 Muse streaming constraint
- You currently start LSL streaming via:
  - `muselsl stream --address <MUSE_ADDR>`
- Unified app must support **either**:
  - **Manual**: you run the shell loop in a terminal.
  - **Integrated**: the unified app launches/stops the `muselsl` loop itself.

---

## 1) System architecture (practical, fast, low‑risk)

### 1.1 High‑level
ALMA OS is one local program that provides:
- a **State Engine** (pulls LSL → computes X/Q → emits NDJSON → writes SQLite)
- a **Dashboard UI** (Dash multipage)
- optional **Actuators** spawned/managed by ALMA OS (**Turrell visuals only** in this MVP)

### 1.2 Processes
**Process A — ALMA OS (Dash + Engine thread)**
- Runs Dash UI.
- Runs a **background StateEngine thread**.

**Process B — Turrell Room (Pygame)**
- Spawned by ALMA OS on demand.
- Reads the same NDJSON file ALMA writes.
- Reads the same mode file ALMA writes.

*(No other actuators in this MVP. Spotify is **read-only tracking** only — no playlist adaptation.)*

### 1.3 Core “bus” contracts
**NDJSON State Stream** (contract for subscribers)
- ALMA app writes `state_stream.ndjson`.
- Turrell follows the file tail.
- Any future actuator (music/lighting/VR) also subscribes.

**Mode File** (simple actuator policy switch)
- ALMA app writes `adaptive_mode.json` (or equivalent path).
- Turrell reads it and adapts visuals.
- Spotify controller reads it and adapts playlists.

### 1.4 Why NDJSON stays (for now)
- Debuggable.
- Stable “State Layer API” contract.
- Great for handoff to Connor because it’s explicit and testable.

---

## 2) Internal modules (what to build)

### 2.1 `StateEngine` (threaded)
Responsibilities:
- Resolve/attach to LSL EEG stream.
- Maintain rolling buffers (per channel + timestamps).
- Every tick:
  - Extract last **WINDOW_SEC=5** seconds.
  - Compute metrics (X/Q/drivers/reliability) using your existing code.
  - Append to a **ring history** (last N minutes for live plots).
  - Emit NDJSON (if enabled).
  - Persist to SQLite (decimated samples + bucket aggregates).

Key APIs:
- `get_latest()` → latest metrics snapshot for UI.
- `get_history(last_seconds)` → for Neurometrics plots.
- `mark_event(label, note, tags)` → stores manual bookmark.

### 2.2 `StreamManager` (muselsl lifecycle)
Responsibilities:
- Detect whether LSL EEG stream exists.
- Provide UI controls:
  - “Start Muse Stream” → spawns `muselsl stream --address ...` **with auto‑restart**.
  - “Stop Muse Stream” → terminates subprocess.

Implementation note:
- Use `subprocess.Popen(["bash","-lc", <while-loop>])` so your exact restart loop works.

### 2.3 `NDJSONWriter`
Responsibilities:
- Append one JSON object per line.
- Ensure directories exist.
- Never crash the engine if file write fails.

### 2.4 `Storage` (SQLite)
Responsibilities:
- Store:
  - session metadata
  - decimated samples
  - bucket aggregates
  - events/bookmarks
  - recipes
  - schedule blocks
  - (later) outcome scoring

### 2.5 `TurrellRunner`
Responsibilities:
- Launch and stop Turrell v5.2.
- Pass args (must satisfy your requirements):
  - `--ndjson <path>`
  - `--mode-file <path>`
  - `--no-freeze-on-invalid` *(always)*
  - `--quality 4k` *(default; recommended always)*
  - `--display {0|1}` *(user-selectable: laptop vs external)*
  - `--fullscreen` *(toggle)*
  - `--q-metric vibe_focus` *(default)*

**Visual adaptation rule (your requirement):**
- Use **OFF-mode palette** for **all** modes.
- Modes only adjust *style parameters* (silky / velvety-balanced / crisp-glow / sensual) via:
  - bloom strength / bloom threshold
  - exposure
  - drift speed
  - (optionally) Q visual smoothing
- Implementation: render using OFF palette/geometry, then apply mode-specific multipliers to the post/tempo params.

### 2.6 `SpotifyNowPlayingLogger` (read-only)
Responsibilities:
- Poll Spotify “currently playing” state (read-only) and timestamp it.
- Write to:
  - `sessions/current/spotify_playback.ndjson`
  - SQLite table `spotify_playback` (see Storage section)
- Expose “Now Playing” to the UI (Home + optionally other pages).
- **Never** issue playback commands (no skip/seek/playlist adaptation in MVP).

---

## 3) Data model (SQLite MVP)

### 3.1 Tables

#### `sessions`
- `id` (TEXT, primary key)
- `started_at` (REAL unix)
- `ended_at` (REAL unix nullable)
- `profile_id` (TEXT)
- `baseline_version` (TEXT)
- `code_version` (TEXT)
- `ndjson_path` (TEXT)

#### `samples` (optional decimation: 1 Hz or 0.2 Hz)
- `ts` (REAL unix)
- `session_id` (TEXT)
- `t_session` (REAL seconds since start)
- `valid` (INTEGER 0/1)
- `quality_conf` (REAL)
- `X` (REAL)
- `Q_canon` (REAL)  *(default = Q_vibe_focus masked)*
- `Q_abs` (REAL)
- `Q_vibe` (REAL)
- `Q_vibe_focus` (REAL)
- `Q_vibe_focus_e` (REAL nullable)
- `state_label` (TEXT nullable)

#### `buckets` (5–15 min)
- `bucket_start_ts` (REAL unix)
- `bucket_end_ts` (REAL unix)
- `session_id` (TEXT)
- `mean_X` (REAL)
- `mean_Q` (REAL) *(canon)*
- `std_Q` (REAL)
- `Q_slope` (REAL)
- `valid_fraction` (REAL)
- `label` (TEXT)  *(Deep Work / Ideation / Recovery / Social etc)*

#### `events`
- `ts` (REAL unix)
- `session_id` (TEXT)
- `kind` (TEXT)  *(manual / auto_peak / auto_streak / recipe_start / recipe_end)*
- `label` (TEXT)
- `note` (TEXT)
- `tags_json` (TEXT)
- `context_json` (TEXT) *(optional: app/tab/media)*

#### `recipes`
- `id` (TEXT)
- `name` (TEXT)
- `target_json` (TEXT) *(target bands/thresholds)*
- `steps_md` (TEXT)
- `mode` (TEXT) *(OFF/MAX_Q/MAX_FOCUS/STABILIZE/custom)*
- `stats_json` (TEXT) *(success rate, best time windows, etc.)*

#### `schedule_blocks`
- `id` (TEXT)
- `date` (TEXT YYYY-MM-DD)
- `title` (TEXT)
- `type` (TEXT) *(writing / focused / research / explore / creative)*
- `duration_min` (INTEGER)
- `flexible` (INTEGER 0/1)
- `planned_start_ts` (REAL nullable)
- `suggested_start_ts` (REAL nullable)
- `notes` (TEXT)

#### `spotify_playback` (read-only tracking)
- `ts` (REAL unix)
- `session_id` (TEXT)
- `is_playing` (INTEGER 0/1)
- `track_id` (TEXT)
- `track_name` (TEXT)
- `artists` (TEXT)
- `album` (TEXT)
- `context_uri` (TEXT nullable)
- `device_name` (TEXT nullable)
- `progress_ms` (INTEGER nullable)
- `duration_ms` (INTEGER nullable)
- `mode` (TEXT nullable) *(copied from adaptive_mode.json for context)*

---

## 4) Readiness feature vector (shared across products)

Use one shared vector for:
- Readiness labels
- Scheduler suggestions
- Memory similarity search
- Recipe success scoring

### 4.1 Bucket vector
`V_bucket = [mean_X, mean_Q, std_Q, Q_slope, valid_fraction]`

### 4.2 Starter label thresholds (tunable)
*(These are placeholders for MVP; you will tune with your personal data.)*
- **Deep Work**: `mean_X high`, `std_Q low`, `valid_fraction high`
- **Ideation**: `std_Q high` and `Q_slope positive`
- **Recovery**: `mean_X low` and `std_Q low/moderate`
- **Social/Engagement**: `mean_Q elevated` with moderate `std_Q`

Important: products should not show “invalid reasons”; they should show **human explanations**.

---

## 5) UI structure (ALMA OS aesthetic + pages)

### 5.1 Aesthetic direction
- Dark, minimal, neon cyan → indigo → magenta gradients.
- Soft glow borders, circuit/fractal motifs.
- Plotly dark theme + custom CSS.

### 5.2 Navigation (sidebar)
- **Home** (Dashboard)
- **Neurometrics** (full Cortex diagnostics)
- **Readiness Map**
- **Scheduler**
- **Memory**
- **Environment OS**
- **State Recipes**
- **Settings**

### 5.3 Page specs

#### Home (Dashboard)
Must show:
- Live connection dot (LSL alive).
- **Reliability (rolling): Last 600 samples: XX%** *(only here + Neurometrics).*
- Current `X` and **canonical Q** (default `Q_vibe_focus`).
- Current state label (Deep Work/Ideation/etc).
- Quick actions:
  - Start/Stop Muse Stream
  - Start/Stop Engine
  - Toggle NDJSON emission
  - Launch/Stop Turrell
  - Mark Event (manual bookmark)

Also include:
- “Daily summary stripe” mini heatmap + today’s top peaks/streaks.

#### Neurometrics
Purpose: Your full lab console inside ALMA.
- Live graphs:
  - Q_raw last 420s (Q_abs_raw, Q_vibe_raw, Q_vibe_focus_raw)
  - X graph
  - Q family graph
  - Drivers / bands
  - Reliability box
- **Dropdown for Q lens (Neurometrics only):** vibe_focus (default), vibe, abs, vibe_focus_e.
- Advanced controls (only here): quality threshold, drift tuning, validation mode, Option‑E profile + τ (after baseline exists).

#### Readiness Map
- Day heatmap view (bucketed).
- Hover/click a bucket → “why panel” with:
  - mean_X, mean_Q, std_Q, slope
  - friendly explanation
  - any events in that window

#### Scheduler (Life OS)
MVP:
- User defines 3–5 flexible block types.
- App suggests time windows based on readiness buckets.
- “Suggest swap” button generates proposals with explanations.
- Advice‑only initially (no auto scheduling required).

#### Memory (State‑Indexed)
MVP:
- Search bar + filters (date range, label, “high Q”, “stable focus”).
- “Find similar” uses cosine similarity over `V_bucket`.
- Result cards show:
  - timestamp range
  - label
  - similarity score
  - linked events / recipe usage

#### Environment OS
- Mode selector writes mode file.
- Turrell controls:
  - Launch fullscreen
  - Stop
  - Preset selector (later)
  - Q metric for Turrell (default vibe_focus; keep advanced toggles here optional)

#### State Recipes
- Recipe list + create/edit.
- Each recipe has:
  - target band definitions
  - steps
  - “Apply now”:
    - writes mode
    - optionally launches Turrell
    - creates an event marker
- Later:
  - auto scoring per recipe based on pre/post buckets.

#### Settings
- Profile selection (local JSON)
- Muse address configuration
- File paths (NDJSON path, DB path, mode file path)
- Decimation + bucket size

---

## 6) Repo structure (Cursor-friendly)

Create a **new repo**; keep your current scripts untouched.

```
alma_os/
  README.md
  pyproject.toml  (or requirements.txt)
  .env.example

  app.py                 # dash entrypoint

  alma/
    __init__.py

    engine/
      state_engine.py
      stream_manager.py
      cortex_adapter.py   # imports or lifts compute from your current file
      ndjson_writer.py
      storage.py
      models.py           # dataclasses for typed state snapshots

    ui/
      layout.py
      theme.css
      pages/
        home.py
        neurometrics.py
        readiness.py
        scheduler.py
        memory.py
        environment.py
        recipes.py
        settings.py

    integrations/
  turrell_runner.py
  spotify_now_playing_logger.py  # read-only tracking

  profiles/
    default.json

  data/
    alma.db

  external/
    app_v4_2_1_cortex_console_drift_ui_stream_ndjson_reliability_v4_2_no_ch_valid_qraw420.py
    xq_turrell_room_2d_v5_2.py
```

**Note:** You can either:
- keep your existing scripts in `external/` and import/borrow logic, OR
- gradually migrate the compute core into `alma/engine/cortex_adapter.py`.

---

## 7) Build plan (phased, ship-first)

### Phase 0 — Project bootstrap (same day)
Goal: one command runs the app and shows pages.
- Create repo + venv.
- Dash multipage app + sidebar.
- Placeholder pages for all 7 tabs.

**Exit criteria:** `python app.py` opens ALMA OS UI.

### Phase 1 — State Engine integration (core week)
Goal: ALMA can show live X/Q on Home and full plots in Neurometrics.
- Implement `StateEngine` loop in a daemon thread.
- Connect to LSL (resolve stream).
- Maintain ring history.
- Implement `get_latest()` and `get_history()`.
- Implement rolling reliability display (last 600 samples).
- Implement Neurometrics plots (including Q_raw 420s).

**Exit criteria:** You can run ALMA, see live dot + live X/Q, and plots update smoothly.

### Phase 2 — NDJSON + Turrell integration (fast win)
Goal: “Environment OS” works.
- NDJSON toggle in UI.
- Implement mode file writing.
- Implement TurrellRunner Launch/Stop.
- **Turrell style rule implementation:**
  - Use OFF palette for all modes.
  - Modes only change style params (bloom/exposure/drift/smoothing).
  - Implement by either:
    - (Preferred) create a new file `xq_turrell_room_2d_v5_3_style_modes.py` copied from v5.2, and in the main loop force `mode_for_render = OFF` while applying mode-based multipliers to exposure/bloom/drift.

**Exit criteria:** button launches Turrell (4K quality), mode switch changes style (not palette), visuals never freeze on invalid, display 0/1 selectable.

### Phase 3 — Storage + Buckets (enables 3 products)
Goal: ALMA writes SQLite and can build day heatmaps.
- Create schema.
- Store decimated samples.
- Implement bucket aggregation job (e.g., every minute compute last bucket).

**Exit criteria:** Readiness Map shows today with data.

### Phase 4 — Memory MVP
Goal: “when did I feel this” works.
- Similarity search over buckets.
- Manual bookmarks (button now, hotkey later).

**Exit criteria:** search returns similar windows with events.

### Phase 5 — Scheduler MVP
Goal: suggest + explain.
- Create simple schedule block editor.
- Suggest times from readiness buckets.

**Exit criteria:** scheduler produces plausible swaps.

### Phase 6 — Recipes MVP
Goal: protocols loop closed.
- Recipe CRUD.
- Apply now writes mode + creates event + optional Turrell.

**Exit criteria:** recipes can be executed and logged.

---

## 8) Cursor “Super Prompts” (copy/paste)

### Prompt A — Bootstrap repo + multipage Dash shell
“Create a new Python project called `alma_os` with a Dash multipage app using a left sidebar navigation. Pages: Home, Neurometrics, Readiness Map, Scheduler, Memory, Environment OS, State Recipes, Settings. Use a dark neon theme (cyan/indigo/magenta accents) via custom CSS. Provide a runnable `app.py` and the full directory structure. Keep all pages as placeholders but wired.”

### Prompt B — Implement StateEngine (threaded) with LSL attach
“Implement `alma/engine/state_engine.py` as a background thread that resolves an LSL EEG stream (Muse), pulls samples, maintains rolling buffers, computes metrics once per second (window=5s step=1s). Expose `get_latest()` and `get_history(last_seconds)`. For now, mock compute with placeholders returning X/Q until we wire the real compute. Ensure the UI never blocks.”

### Prompt C — Wire real compute by adapting existing Cortex file
“Adapt the compute logic from `external/app_v4_2_1_cortex_console_drift_ui_stream_ndjson_reliability_v4_2_no_ch_valid_qraw420.py` into `alma/engine/cortex_adapter.py` so StateEngine produces the same metric keys (X, Q_abs, Q_vibe, Q_vibe_focus, raw variants, reliability.valid, reliability.quality_conf, reason_codes). Preserve behaviors: channel agreement diagnostic-only; do not hard invalidate on channel agreement. Do not change formulas; only refactor.”

### Prompt D — Home page live widgets + rolling reliability
“Implement Home page callbacks that show: live connection dot (LSL alive if last sample <2.5s), Reliability (rolling last 600 samples valid%), current X and canonical Q (default Q_vibe_focus), and quick buttons: Start/Stop Muse Stream, Start/Stop Engine, Toggle NDJSON Emit, Launch/Stop Turrell, Mark Event. Reliability must only appear on Home and Neurometrics.”

### Prompt E — Neurometrics page plots (incl Q_raw 420s)
“Implement Neurometrics page with Plotly graphs including a Q_raw rolling 420-second plot showing Q_abs_raw, Q_vibe_raw, Q_vibe_focus_raw, plus the existing X/Q/Drivers plots. Add a dropdown for Q lens: vibe_focus (default), vibe, abs, vibe_focus_e. This dropdown affects which Q is displayed in Neurometrics only (not product pages).”

### Prompt F — NDJSON writer + mode file + Turrell runner
“Implement NDJSON emission to a configurable path (default `state_stream.ndjson`), appending one packet per compute step. Implement mode file writer (default `spotify_proto/adaptive_mode.json`). Implement TurrellRunner that launches `external/xq_turrell_room_2d_v5_3_style_modes.py` (create this by copying v5.2) with:
- `--ndjson <path>`
- `--mode-file <path>`
- `--no-freeze-on-invalid`
- `--quality 4k`
- `--display` selectable (0 or 1)
- `--fullscreen` toggle
- `--q-metric vibe_focus`

Implement style-modes behavior in the v5.3 file: always use OFF palette, while modes only modify exposure/bloom/drift/smoothing (silky / velvety-balanced / crisp-glow / sensual). Add Environment OS page UI for: mode selector, display selector (0/1), fullscreen toggle, launch/stop controls.”

### Prompt F2 — Spotify Now Playing Logger (read-only)
“Implement `alma/integrations/spotify_now_playing_logger.py` that logs what’s playing on Spotify (read-only). Requirements:
- Poll every 1–2 seconds.
- Capture: ts, is_playing, track_id, track_name, artists, album, progress_ms, duration_ms, device_name, context_uri.
- Also read `adaptive_mode.json` and copy current `mode` into each record (for analysis context).
- Write each poll result to:
  - SQLite table `spotify_playback`
  - `sessions/current/spotify_playback.ndjson` (append-only)
- Expose `get_latest()` to show “Now Playing” on Home.
- Do NOT control playback (no skip/seek/playlist edits).

Implementation options:
- Use Spotipy with OAuth scope `user-read-playback-state`.
- Persist token locally.
- Fail gracefully when Spotify isn’t running or auth missing.”

### Prompt G — SQLite schema + bucket aggregation + Readiness heatmap
“Implement SQLite storage with tables: sessions, samples (decimated), buckets (5–15 min), events, recipes, schedule_blocks, spotify_playback. Add an aggregation job that computes buckets and writes label + vector fields. Implement Readiness Map page that shows today’s buckets as a heatmap with hover ‘why panel’ (mean_X, mean_Q, std_Q, slope, valid_fraction, label). Do not show reliability details here.””

### Prompt H — Memory MVP
“Implement Memory page: query buckets + events, provide a search UI and a ‘find similar’ function using cosine similarity over [mean_X, mean_Q, std_Q, Q_slope, valid_fraction]. Return results as cards with similarity score, time range, label, and linked events.”

### Prompt I — Scheduler MVP
“Implement Scheduler page: user defines 3–5 flexible blocks. Provide ‘Suggest Swap’ that chooses best upcoming buckets based on block type (e.g., deep work → high X + low std_Q). Advice-only with explanations; no calendar integrations needed.”

### Prompt J — Recipes MVP
“Implement Recipes CRUD and ‘Apply Now’ button that writes mode, logs an event, and optionally launches Turrell. Add later scoring placeholder to compute success rate from pre/post buckets.”

---

## 9) Runbook (how you’ll use it)

### 9.1 Manual stream mode (today)
1) In terminal, run your existing muselsl loop.
2) `python app.py`
3) In Home: Start Engine → Toggle NDJSON → Launch Turrell

### 9.2 Integrated stream mode (after StreamManager)
1) `python app.py`
2) In Home: Start Muse Stream → Start Engine

---

## 10) Option‑E stance (for now)
- Treat Option‑E as **experimental** until you have a per‑user baseline.
- ALMA OS should carry it as optional fields; no product logic should depend on it.
- When ready: add baseline capture + profile stats to enable `vibe_focus_e`.

---

## 11) Success criteria for “MVP ALMA OS”
- One command runs ALMA OS.
- Home shows live dot + rolling reliability + current X/Q.
- Neurometrics provides full diagnostic plots including Q_raw 420s.
- Environment OS launches Turrell and mode switching works.
- Readiness Map shows a day heatmap.
- Memory can retrieve similar states.
- Scheduler suggests swaps.
- Recipes can be applied and logged.

---

## 12) Next improvements (after MVP)
- Hotkey bookmarks (pynput; requires macOS accessibility permission).
- Spotify now-playing logger enhancements (more robust polling, better metadata capture, session linking).
- Per-day circadian pattern models + readiness forecasting.
- Recipe outcome scoring (A/B testing).
- Exportable “state story” summaries (daily/weekly narrative over buckets + events + media).

