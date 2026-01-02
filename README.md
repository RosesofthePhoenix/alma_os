# ALMA OS

A local neurotech dashboard for live EEG/NDJSON streams, HCE/Q/X analytics, bookmarking/capture, scheduling, recipes, and immersive Turrell visuals.

## Quick start
1) Create/activate venv (recommended)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2) Run Dash app (defaults to http://127.0.0.1:8050)
```bash
python app.py
```
If 8050 is busy, stop other processes or set `PORT=8051 python app.py`.

## Key features (Phase 2/3/4)
- Live metrics + HCE: v4.2.1 metrics, EMA for X, HCE (super-linear, scaled) carried through snapshots/NDJSON/buckets. Top bar shows NDJSON status and live `X/Q/HCE` (3 decimals).
- Baseline robustness: loads `baselines/baseline_global_muse_v1_revised.json` with fallbacks; status logged on engine init.
- Persistence (SQLite `data/alma.db`):
  - Samples → buckets (mean_X/mean_Q/mean_HCE/std_Q/Q_slope/valid_fraction)
  - Events/bookmarks (with snapshot_json, context, quick captures)
  - Recipes (description, targets, steps, efficacy_score)
  - Schedule blocks
  - Spotify track_sessions (mean_HCE/Q/X, durations)
- Quick Capture & bookmarks: global panel to save note, window (1–30 min), media/person; bookmark modal captures social/activity/mood/environment/intention; surfaced in Memory.
- Pages:
  - Home: status, HCE/optimal windows, peak track HCE, NDJSON toggle, bookmark modal.
  - Neurometrics: HCE raw + rolling z-score.
  - Readiness: HCE-aware labels (TRANSCENDENT_SYNTHESIS, CONTEMPLATIVE_FLOW, etc.), timeline/stripe, “Why” panel.
  - Scheduler: draggable blocks + HCE-driven suggestions.
  - Recipes: CRUD, apply writes `sessions/current/adaptive_mode.json` (optionally launch Turrell).
  - Memory: similarity search with HCE weighting; quick captures list; TRANSCENDENT filter.
  - Spotify Resonance: visuals first (HCE histogram, top artists by HCE, bar/scatter, timeline, correlation heatmap, findings), Top 20 table at bottom; backfills missing means.
- Turrell Room:
  - Runner launches `external/xq_turrell_room_2d_v6_hce.py` (HCE-enhanced); HUD/text fallback (SysFont→freetype→pixel), hotkeys logged (H, 0–3, F, ESC double-press), display retry, NDJSON follower tolerant to truncation.
  - v5_3 style script kept for reference; v6 is the active target.

## Paths and runtime files
- NDJSON stream: `sessions/current/state_stream.ndjson`
- Adaptive mode: `sessions/current/adaptive_mode.json`
- Baseline: `baselines/baseline_global_muse_v1_revised.json`
- Database: `data/alma.db`
- Profile: `profiles/default.json`
- External analysis exports: `external_analysis/` (state_stream/spotify_playback in NDJSON/JSON)

## Spotify logging
Uses cached token under project root (`.cache`). If needed:
```
export SPOTIPY_CACHE="/Users/raycraigantelo/Documents/alma_os/.cache"
```

## Turrell notes
Start from Environment page (NDJSON must be ON). Runner passes ndjson/mode paths, HUD on, quality preset, display, q-metric. Hotkeys: 0–3 modes, H HUD, F fullscreen, ESC quit (double press), V/M cycle Q source/metric where applicable.

## Troubleshooting
- Port in use: stop other `python app.py` or run with a different PORT.
- Font errors in Turrell: v6 falls back to freetype/pixel text; HUD remains visible even without pygame.font.
- NDJSON empty: ensure engine running and `sessions/current/state_stream.ndjson` being written. ***!
