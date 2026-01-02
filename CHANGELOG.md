# Changelog

## v0.4 — Turrell hardening + Phase 4 UI/Spotify upgrades
- Turrell Room v6/v7: runner targets v6; NDJSON follower reads from start and survives truncation; HUD/text is bulletproof (SysFont→freetype→pixel fallback), HUD forced ON with banner; hotkeys logged (H, 0–3 modes, F fullscreen, ESC double-press); display set_mode retry; stays up even without pygame.font.
- Global UI: persistent live metrics bar (X/Q/HCE) plus Quick Capture panel (notes, window length, media/person) stored to events; NDJSON status indicator shared; quick captures surface in Memory page.
- Contextual logging: bookmark modal enriched (social context, activity, mood/energy, environment, intention); events store snapshot_json with HCE.
- Spotify analytics: track session logging/backfill for mean_HCE/Q/X; “Top tracks”/“Peak Track HCE” surfaced on Home; Resonance page reorganized—visuals first (HCE histogram, top artists by HCE, bar/scatter/timeline/corr, findings) with Top 20 table moved to bottom.
- Readiness/Scheduler/Recipes/Memory continuity: HCE-driven labels and suggestions retained; recipe/schedule schemas persisted (mean_HCE, efficacy, blocks); Memory uses quick-capture trigger and TRANSCENDENT filter.

## v0.3 — Scheduler + Recipes MVP, HCE-driven suggestions, fixes
- Scheduler: day/week timeline with draggable blocks; suggestions rank historical high-HCE hours (mean_HCE>2, valid_fraction, low std_Q); drag persists to SQLite schedule_blocks; Plotly layout fix removes invalid editable flag.
- Recipes: schema now stores description, steps_json, efficacy_score; CRUD modal with JSON targets/steps; Apply writes adaptive_mode.json (target metrics, steps, recipe id) and can launch Turrell; efficacy updates from bucket HCE vs mean_HCE_min; delete/edit via cards or modal.
- Memory: fixed Dash callback to use ctx.triggered_id for selection.
- Stability: initial recipe list renders from storage; newline steps accepted when JSON parsing fails; DB backfill columns for existing recipes.

## v0.1 — Baseline robustness, v4.2.1 metrics, HCE, Turrell args, UI safety
- Baseline: deterministic loading (absolute/ROOT/cwd), gamma_ref/bg_ref captured; Start Engine reloads baseline; status exposed to snapshots.
- Metrics: integrated v4.2.1 compute; X now uses EMA; meta includes metrics_version; raw/masked Qs preserved.
- New metric: HCE = (Q_vibe_focus / X_ema) * log1p(Q_vibe_focus), with HCE_raw; exported in snapshot and NDJSON.
- UI: Home shows HCE; Neurometrics adds HCE chart; Home callback hardened and output order fixed.
- Turrell: runner passes ndjson/mode, --hud, --quality 4k, display selection, q-metric, no-freeze; safer stop.
- Environment: warns when NDJSON emit is off; “Enable NDJSON” button retained.
- Paths: ensure baselines/ directory is created; profile default baseline_path set.
- Engine: baseline status logged on init; state history carries HCE/HCE_raw; NDJSON remains schema-compatible.

## v0.2 — Readiness + HCE + NDJSON persistence + bookmarking
- Readiness Map: HCE-aware labels (ELEVATION), opacity by mean_HCE, hover “Why” details, bucket count indicator, wall-clock bucket timestamps.
- Buckets: mean_HCE persisted; aggregation uses masked metrics; readiness now populates correctly.
- NDJSON toggle: session-scoped store shared across pages with global indicator; Environment sync fixed.
- Neurometrics: adds HCE z-score trace alongside raw.
- Bookmarks: “Bookmark Now” modal with label/note; snapshot_json persisted (includes HCE/X/Q); Recent Events shows bookmarks.
- Stability/cleanup: removed bucket debug noise; readiness click-data mapping fixed; Spotify logging confirmed; persistence via SQLite intact.

## v0.2.1 — HCE super-linear scaling + readiness heatmap rendering
- HCE: switched to super-linear formula ( (q/denom)**1.5 * q ) with X-ema denom and 10k scaling for visibility; raw/masked and NDJSON/buckets carry scaled values.
- Readiness: heatmap rendering fixed; click/hover populates “Why” with HCE/Q/X details; wall-clock timestamps; bucket count shown.
- Neurometrics: HCE chart labeled as “raw (scaled)” with normalized trace intact.

