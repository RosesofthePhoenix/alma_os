# Changelog

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

