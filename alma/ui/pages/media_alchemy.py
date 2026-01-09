import datetime as dt
import json
import sqlite3
import time
from typing import List, Dict, Tuple, Optional

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from alma import config
from alma.engine import storage
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth

PROFILES_DIR = config.ROOT_DIR / "profiles"
PERSONAL_RESONANCE_CACHE = config.SESSIONS_CURRENT_DIR / "personal_resonance_cache.json"


def _fetch_track_sections(track_id: str) -> pd.DataFrame:
    try:
        rows = storage.list_track_sections(track_id, limit_sessions=1)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["section_rel_start"] = df["section_start_ts"] - df["start_ts"]
        df["section_rel_end"] = df["section_end_ts"] - df["start_ts"]
        return df
    except Exception:
        return pd.DataFrame()


_ANALYSIS_FAIL_CACHE: Dict[str, float] = {}


def _spotify_client():
    """Advanced Spotify endpoints disabled; return None to force local estimation."""
    return None


def _load_resonance_cache() -> Dict[str, dict]:
    try:
        if PERSONAL_RESONANCE_CACHE.exists():
            return json.loads(PERSONAL_RESONANCE_CACHE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_resonance_cache(cache: Dict[str, dict]) -> None:
    try:
        PERSONAL_RESONANCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PERSONAL_RESONANCE_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


def _real_sections_from_spotify(track_id: str, duration: float) -> Tuple[List[Tuple[int, str, float, float]], str]:
    # Spotify advanced endpoints restricted — using local personal resonance estimation.
    return _local_resonance_sections(track_id, duration)


def _local_resonance_sections(track_id: str, duration: float) -> Tuple[List[Tuple[int, str, float, float]], str]:
    """Derive pseudo-sections from historical buckets and track_sessions."""
    total = max(duration, 1.0)
    # Try to see historical lifts
    try:
        rows = storage.list_track_sections(track_id, limit_sessions=None)
    except Exception:
        rows = []
    if rows:
        # Use quartiles of session to define phases
        rel_starts = [float(r.get("section_start_ts", 0) - r.get("start_ts", 0)) for r in rows if r.get("start_ts") is not None]
        rel_ends = [float(r.get("section_end_ts", 0) - r.get("start_ts", 0)) for r in rows if r.get("start_ts") is not None]
        bounds = sorted({0.0} | set(rel_starts) | set(rel_ends) | {total})
    else:
        bounds = list(np.linspace(0, total, num=5))
    sections = []
    for i in range(len(bounds) - 1):
        start = bounds[i]
        end = bounds[i + 1]
        label = "Flow"
        # simple heuristic: earlier parts uplift, later introspective
        if i == 0:
            label = "Uplift"
        elif i == len(bounds) - 2:
            label = "Introspective"
        sections.append((i, label, start, end))
    return sections, "local resonance (historical)"


def _aggregate_sections(track_id: str) -> Tuple[pd.DataFrame, str]:
    """Aggregate per-section metrics across all sessions for the track."""
    rows = storage.list_track_sections(track_id, limit_sessions=None)
    track_means, duration, title, artist = _latest_track_stats(track_id)
    if not rows:
        # Build uniform pseudo sections using track means
        df_uniform = _build_uniform_sections(duration, track_means)
        df_uniform["section_start_ts"] = np.nan
        df_uniform["section_end_ts"] = np.nan
        df_uniform["session_id"] = None
        df_uniform["track_session_id"] = None
        df_uniform["start_ts"] = 0.0
        df_uniform["end_ts"] = duration
        df_uniform["source"] = "Uniform (track avg)"
        return df_uniform, "uniform"

    df = pd.DataFrame(rows)
    df["section_rel_start"] = df["section_start_ts"] - df["start_ts"]
    df["section_rel_end"] = df["section_end_ts"] - df["start_ts"]

    # Sessions table (unique)
    sess_df = (
        df.groupby("track_session_id")
        .agg(
            start_ts=("start_ts", "min"),
            end_ts=("end_ts", "max"),
            session_id=("session_id", "first"),
        )
        .reset_index()
    )

    # Always use estimated pseudo sections (audio analysis disabled)
    section_source_note = "Estimated sections — resonance building live"
    total = max(duration, 1.0)
    n_parts = 5
    step = total / float(n_parts)
    base_sections = [(i, f"Part {i+1} (est.)", i * step, (i + 1) * step) for i in range(n_parts)]

    agg_rows = []
    for idx, label, rel_start, rel_end in base_sections:
        means_list = []
        bucket_counts = []
        for _, sr in sess_df.iterrows():
            abs_start = (sr["start_ts"] or 0) + rel_start
            abs_end = (sr["start_ts"] or 0) + rel_end
            if not pd.isna(sr["end_ts"]):
                abs_end = min(abs_end, sr["end_ts"])
            sess_id = sr.get("session_id")
            buckets = storage.get_buckets_between(abs_start, abs_end, session_id=sess_id)
            bucket_counts.append(len(buckets))
            if buckets:
                means_list.append(storage._compute_means_for_window(abs_start, abs_end, session_id=sess_id))

        if means_list:
            hce_vals = [m.get("mean_HCE", 0.0) for m in means_list]
            q_vals = [m.get("mean_Q", 0.0) for m in means_list]
            x_vals = [m.get("mean_X", 0.0) for m in means_list]
            mean_HCE = float(np.nanmean(hce_vals))
            mean_Q = float(np.nanmean(q_vals))
            mean_X = float(np.nanmean(x_vals))
            source = "multi-session buckets"
            total_buckets = sum(bucket_counts)
            if total_buckets > 0 and total_buckets < 3:
                mean_HCE = 0.7 * mean_HCE + 0.3 * track_means.get("mean_HCE", 0.0)
                mean_Q = 0.7 * mean_Q + 0.3 * track_means.get("mean_Q", 0.0)
                mean_X = 0.7 * mean_X + 0.3 * track_means.get("mean_X", 0.0)
                source = "multi-session blended"
        else:
            mean_HCE = float(track_means.get("mean_HCE", 0.0))
            mean_Q = float(track_means.get("mean_Q", 0.0))
            mean_X = float(track_means.get("mean_X", 0.0))
            source = "Uniform (track avg)"
            bucket_counts.append(0)

        agg_rows.append(
            {
                "section_index": idx,
                "section_label": label,
                "section_rel_start": rel_start,
                "section_rel_end": rel_end,
                "mean_HCE": mean_HCE,
                "mean_Q": mean_Q,
                "mean_X": mean_X,
                "source": source,
                "bucket_count": sum(bucket_counts),
            }
        )

    agg_df = pd.DataFrame(agg_rows).sort_values("section_index")
    return agg_df, section_source_note


def _track_options() -> List[Dict[str, str]]:
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            df_top = pd.read_sql_query(
                """
                SELECT track_id, title, artist, AVG(mean_HCE) AS avg_hce
                FROM track_sessions
                WHERE track_id IS NOT NULL AND mean_HCE IS NOT NULL
                GROUP BY track_id, title, artist
                ORDER BY avg_hce DESC
                LIMIT 50
                """,
                conn,
            )
            df_recent = pd.read_sql_query(
                """
                SELECT track_id, title, artist, MAX(start_ts) AS last_play
                FROM track_sessions
                WHERE track_id IS NOT NULL
                GROUP BY track_id, title, artist
            ORDER BY last_play DESC
            LIMIT 100
                """,
                conn,
            )
        df_all = pd.read_sql_query(
            """
            SELECT DISTINCT track_id, title, artist
            FROM track_sessions
            WHERE track_id IS NOT NULL
            """,
            conn,
        )
        df = pd.concat([df_top, df_recent, df_all], ignore_index=True)
        df = df.drop_duplicates(subset=["track_id"], keep="first")
        return [{"label": f"{r['title']} — {r['artist']}", "value": r["track_id"]} for _, r in df.iterrows()] if not df.empty else []
    except Exception:
        return []


def _search_tracks(query: str, limit: int = 50) -> List[Dict[str, str]]:
    if not query:
        return []
    q = f"%{query.lower()}%"
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            df = pd.read_sql_query(
                """
                SELECT DISTINCT track_id, title, artist, MAX(start_ts) AS last_play
                FROM track_sessions
                WHERE track_id IS NOT NULL
                  AND (lower(title) LIKE ? OR lower(artist) LIKE ?)
                GROUP BY track_id, title, artist
                ORDER BY last_play DESC
                LIMIT ?
                """,
                conn,
                params=(q, q, limit),
            )
        return [{"label": f"{r['title']} — {r['artist']}", "value": r["track_id"]} for _, r in df.iterrows()] if not df.empty else []
    except Exception:
        return []


def _latest_track_stats(track_id: str) -> Tuple[Dict[str, float], float, str, str]:
    """Return mean metrics, duration seconds, title, artist for latest session."""
    row = storage.get_latest_track_session(track_id)
    if not row:
        return {"mean_HCE": 0.0, "mean_Q": 0.0, "mean_X": 0.0}, 0.0, "", ""
    start_ts = float(row.get("start_ts") or 0.0)
    end_ts = float(row.get("end_ts") or start_ts)
    duration = max(end_ts - start_ts, 0.0)
    means = {
        "mean_HCE": float(row.get("mean_HCE") or 0.0),
        "mean_Q": float(row.get("mean_Q") or 0.0),
        "mean_X": float(row.get("mean_X") or 0.0),
    }
    return means, duration, row.get("title") or "", row.get("artist") or ""


def _build_uniform_sections(duration: float, means: Dict[str, float]) -> pd.DataFrame:
    total = max(duration, 1.0)
    n_parts = 5
    step = total / float(n_parts)
    rows = []
    for i in range(n_parts):
        rel_start = i * step
        rel_end = (i + 1) * step
        rows.append(
            {
                "section_index": i,
                "section_label": f"Part {i+1} (est.)",
                "section_rel_start": rel_start,
                "section_rel_end": rel_end,
                "mean_HCE": float(means.get("mean_HCE", 0.0)),
                "mean_Q": float(means.get("mean_Q", 0.0)),
                "mean_X": float(means.get("mean_X", 0.0)),
                "source": "Uniform (track avg)",
            }
        )
    return pd.DataFrame(rows)


def _smooth_series(series: pd.Series, window: int = 5) -> pd.Series:
    if series.empty:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _historical_hce_series(track_id: str, bin_sec: float = 1.0, max_sessions: int = 80) -> Tuple[pd.DataFrame, pd.DataFrame, float, Optional[float]]:
    """Aggregate historical per-second metrics; returns (Q_smooth_df, HCE_df, max_dur, hce_p95)."""
    # Prefer stored per-second waveform points
    try:
        wf_points = storage.list_track_waveform_points(track_id, limit_sessions=3, limit_points=8000)
        if wf_points:
            df = pd.DataFrame(wf_points)
            agg_q = df.groupby("rel_sec")["q"].mean().reset_index().rename(columns={"rel_sec": "t", "q": "Q"})
            agg_h = df.groupby("rel_sec")["hce"].mean().reset_index().rename(columns={"rel_sec": "t", "hce": "HCE"})
            agg_q["Q"] = _smooth_series(agg_q["Q"], window=5)
            agg_h["HCE_LOG"] = np.log10(agg_h["HCE"].clip(lower=0) + 1.0)
            hce_p95 = float(np.nanpercentile(agg_h["HCE"], 95)) if not agg_h.empty else None
            max_dur = float(agg_q["t"].max() if not agg_q.empty else agg_h["t"].max() if not agg_h.empty else 0.0)
            return agg_q, agg_h, max_dur, hce_p95
    except Exception:
        pass

    # Fallback from buckets if no stored points
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            sess_df = pd.read_sql_query(
                """
                SELECT ts.session_id, ts.start_ts, ts.end_ts, sp.track_id as uri
                FROM track_sessions ts
                LEFT JOIN spotify_playback sp
                  ON sp.session_id = ts.session_id
                 AND sp.track_id = ts.track_id
                WHERE ts.track_id = ?
                ORDER BY ts.start_ts DESC
                LIMIT ?
                """,
                conn,
                params=(track_id, max_sessions),
            )
    except Exception:
        return pd.DataFrame(columns=["t", "Q"]), pd.DataFrame(columns=["t", "HCE"]), 0.0, None
    if sess_df.empty:
        return pd.DataFrame(columns=["t", "Q"]), pd.DataFrame(columns=["t", "HCE"]), 0.0, None

    bins_q: Dict[int, list] = {}
    bins_h: Dict[int, list] = {}
    max_duration = 0.0
    for _, row in sess_df.iterrows():
        s0 = float(row.get("start_ts") or 0.0)
        s1 = float(row.get("end_ts") or s0)
        max_duration = max(max_duration, s1 - s0)
        sess_id = row.get("session_id")
        buckets = storage.get_buckets_between(s0, s1, session_id=sess_id)
        for b in buckets:
            rel = float(b.get("bucket_start_ts", 0.0) - s0)
            idx = int(rel // bin_sec)
            bins_q.setdefault(idx, []).append(b.get("mean_Q", 0.0) or 0.0)
            bins_h.setdefault(idx, []).append(b.get("mean_HCE", 0.0) or 0.0)

    if not bins_q and not bins_h:
        return pd.DataFrame(columns=["t", "Q"]), pd.DataFrame(columns=["t", "HCE"]), max_duration, None

    max_idx = int(max_duration // bin_sec) + 1
    ts = np.arange(0, max_idx * bin_sec, bin_sec)
    q_vals = np.full_like(ts, np.nan, dtype=float)
    h_vals = np.full_like(ts, np.nan, dtype=float)
    for idx, arr in bins_q.items():
        if 0 <= idx < len(q_vals):
            q_vals[idx] = float(np.nanmean(arr))
    for idx, arr in bins_h.items():
        if 0 <= idx < len(h_vals):
            h_vals[idx] = float(np.nanmean(arr))
    q_series = pd.Series(q_vals, index=ts).interpolate(limit_direction="both")
    h_series = pd.Series(h_vals, index=ts).interpolate(limit_direction="both")
    q_df = pd.DataFrame({"t": q_series.index, "Q": _smooth_series(q_series, window=5)})
    h_df = pd.DataFrame({"t": h_series.index, "HCE": h_series.values})
    h_df["HCE_LOG"] = np.log10(h_df["HCE"].clip(lower=0) + 1.0)
    hce_p95 = float(np.nanpercentile(h_df["HCE"], 95)) if not h_df.empty else None
    return q_df, h_df, max_duration, hce_p95


def _live_hce_series(track_id: str, duration_hint: float, bin_sec: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[float], Optional[str], int]:
    """Live per-second smoothed Q/X/HCE for the current playing session."""
    try:
        latest = storage.get_latest_spotify(session_id=None)
    except Exception:
        latest = None
    if not latest or not latest.get("is_playing") or latest.get("track_id") != track_id:
        return (
            pd.DataFrame(columns=["t", "Q"]),
            pd.DataFrame(columns=["t", "HCE"]),
            pd.DataFrame(columns=["t", "X"]),
            None,
            None,
            0,
        )

    progress_ms = latest.get("progress_ms") or 0
    duration_ms = latest.get("duration_ms") or 1
    progress_sec = max(0.0, min(float(progress_ms) / 1000.0, float(duration_ms) / 1000.0))
    duration_cap = max(duration_hint, float(duration_ms) / 1000.0, progress_sec)

    session_row = storage.get_latest_track_session(track_id)
    if not session_row:
        return (
            pd.DataFrame(columns=["t", "Q"]),
            pd.DataFrame(columns=["t", "HCE"]),
            pd.DataFrame(columns=["t", "X"]),
            progress_sec,
            None,
            0,
        )
    start_ts = float(session_row.get("start_ts") or 0.0)
    # pull buckets in window, any session, and filter by track_uri match when available
    buckets = storage.get_buckets_between(start_ts - 30.0, start_ts + duration_cap + bin_sec + 30.0, session_id=None)

    # Build dense per-second bins up to current progress for Q and HCE
    max_idx = int(np.ceil(max(progress_sec, bin_sec) / bin_sec)) + 1
    ts = np.arange(0, max_idx * bin_sec, bin_sec)
    q_vals = np.full(len(ts), np.nan, dtype=float)
    h_vals = np.full(len(ts), np.nan, dtype=float)
    x_vals = np.full(len(ts), np.nan, dtype=float)
    live_bucket_count = 0
    for b in buckets:
        if b.get("track_uri") and b.get("track_uri") != track_id:
            continue
        rel_start = b.get("relative_seconds")
        if rel_start is None:
            rel_start = float(b.get("bucket_start_ts", 0.0) - start_ts)
        rel_start = float(rel_start)
        rel_end = b.get("bucket_end_ts")
        if rel_end is not None:
            rel_end = float(rel_end - start_ts)
        else:
            rel_end = rel_start + bin_sec
        if rel_end < -10.0 or rel_start > duration_cap + 30.0:
            continue
        idx_start = int(np.floor(rel_start / bin_sec))
        idx_end = int(np.floor(rel_end / bin_sec))
        idx_start = max(idx_start, 0)
        idx_end = min(idx_end, len(ts) - 1)
        q_val = float(b.get("mean_Q", 0.0) or 0.0)
        h_val = float(b.get("mean_HCE", 0.0) or 0.0)
        x_val = float(b.get("mean_X", 0.0) or 0.0)
        for idx in range(idx_start, idx_end + 1):
            q_vals[idx] = q_val
            h_vals[idx] = h_val
            x_vals[idx] = x_val
        live_bucket_count += 1
    q_series = pd.Series(q_vals, index=ts).ffill().bfill()
    h_series = pd.Series(h_vals, index=ts).ffill().bfill()
    x_series = pd.Series(x_vals, index=ts).ffill().bfill()
    q_series = _smooth_series(q_series, window=5)
    x_series = _smooth_series(x_series, window=5)
    h_series = h_series.interpolate(limit_direction="both")
    if h_series.isna().any():
        h_series = h_series.fillna(0.0)
    if q_series.isna().any():
        q_series = q_series.fillna(0.0)
    if x_series.isna().any():
        x_series = x_series.fillna(0.0)
    live_q = pd.DataFrame({"t": ts, "Q": q_series.values})
    live_h = pd.DataFrame({"t": ts, "HCE": h_series.values})
    live_x = pd.DataFrame({"t": ts, "X": x_series.values})

    live_section = None
    try:
        secs = storage.list_track_sections(track_id, limit_sessions=1)
        if secs:
            for s in secs:
                rel_start = (s.get("section_start_ts", 0) - s.get("start_ts", 0)) if s.get("start_ts") is not None else 0
                rel_end = (s.get("section_end_ts", 0) - s.get("start_ts", 0)) if s.get("start_ts") is not None else 0
                if progress_sec >= rel_start and progress_sec <= rel_end:
                    live_section = s.get("section_label")
                    break
    except Exception:
        pass

    return live_q, live_h, live_x, progress_sec, live_section, live_bucket_count


layout = dbc.Container(
    [
        dbc.Card(
            [
                dbc.CardHeader("Media Alchemy — Intra-Track Resonance"),
                dbc.CardBody(
                    [
                        dcc.Dropdown(id="ma-track-select", placeholder="Select a track", className="mb-2"),
                        dbc.Input(id="ma-search-text", placeholder="Search all tracks (title/artist)", className="mb-2", debounce=True),
                        dcc.Dropdown(id="ma-search-results", placeholder="Search results", className="mb-3"),
                        dbc.Checklist(
                            id="ma-metric-toggle",
                            options=[
                                {"label": "Rolling Q (5s)", "value": "q"},
                                {"label": "Rolling X (5s)", "value": "x"},
                                {"label": "Rolling HCE (5s)", "value": "hce"},
                            ],
                            value=["q"],
                            inline=True,
                            switch=True,
                            className="mb-2",
                        ),
                        dbc.Checklist(
                            id="ma-highlight-trans",
                            options=[{"label": "Highlight Transcendence", "value": "on"}],
                            value=["on"],
                            switch=True,
                            className="mb-2",
                        ),
                        dcc.Graph(id="ma-track-plot"),
                        html.Div(id="ma-track-table", className="mt-2"),
                        html.Div(id="ma-oracle-mini", className="mt-2 text-info"),
                        html.Div(id="ma-status", className="small text-muted mt-1"),
                        dcc.Interval(id="ma-interval", interval=1500, n_intervals=0),
                    ]
                ),
            ],
            className="page-card",
        )
    ],
    fluid=True,
    className="page-container",
)


@callback(
    Output("ma-track-select", "options"),
    Output("ma-track-select", "value"),
    Input("ma-interval", "n_intervals"),
    Input("ma-search-results", "value"),
    State("ma-track-select", "value"),
)
def _load_options(_n, search_choice, current_selection):
    opts = _track_options()
    # If user picked from search dropdown, honor it
    if search_choice and any(o["value"] == search_choice for o in opts):
        return opts, search_choice
    # Preserve user choice if present
    if current_selection and any(o["value"] == current_selection for o in opts):
        return opts, current_selection
    # auto-select currently playing if available
    current_val: Optional[str] = None
    try:
        latest = storage.get_latest_spotify(session_id=None)
        if latest and latest.get("is_playing") and latest.get("track_id"):
            current_val = latest.get("track_id")
            if current_val and not any(o["value"] == current_val for o in opts):
                opts = [{"label": f"{latest.get('track_name','?')} — {latest.get('artists','?')}", "value": current_val}] + opts
    except Exception:
        pass
    if not current_val and opts:
        current_val = opts[0]["value"]
    return opts, current_val


@callback(
    Output("ma-search-results", "options"),
    Output("ma-search-results", "value"),
    Input("ma-search-text", "value"),
)
def _search_options(query):
    opts = _search_tracks(query or "")
    return opts, None


@callback(
    Output("ma-track-plot", "figure"),
    Output("ma-track-table", "children"),
    Output("ma-oracle-mini", "children"),
    Output("ma-status", "children"),
    Input("ma-track-select", "value"),
    Input("ma-track-plot", "clickData"),
    Input("ma-interval", "n_intervals"),
    Input("ma-metric-toggle", "value"),
    Input("ma-highlight-trans", "value"),
)
def _render_track(track_id, click_data, _n, metrics_selected, highlight_trans):
    if not track_id:
        return go.Figure(), html.Div("Select a track to view."), "", ""
    track_means, duration, title, artist = _latest_track_stats(track_id)
    df, section_source_note = _aggregate_sections(track_id)
    if df.empty:
        df = _build_uniform_sections(duration, track_means)
    track_mean = track_means.get("mean_HCE", 0.0) or (df["mean_HCE"].mean() if not df.empty else 0.0)
    avg_hce = df["mean_HCE"].mean() if not df.empty else 0.0
    # Live per-second lines only (pure live hero)
    live_q, live_h, live_x, progress_rel, live_section_label, live_bucket_count = _live_hce_series(track_id, duration_hint=duration, bin_sec=1.0)
    duration_final = max(duration, (progress_rel or 0.0), 1.0)

    # Color scale for sections
    hce_min = df["mean_HCE"].min()
    hce_max = df["mean_HCE"].max()
    def _color_for(val):
        if hce_max == hce_min:
            return "purple"
        t = (val - hce_min) / (hce_max - hce_min + 1e-9)
        # blend purple->gold
        return f"rgba({int(120+135*t)}, {int(60+120*t)}, {int(200*t)}, 0.18)"

    fig = go.Figure()
    # Placeholder / backbone over full duration to avoid blanks
    base_x = [0, duration_final]
    fig.add_trace(
        go.Scatter(
            x=base_x,
            y=[0, 0],
            mode="lines",
            line=dict(color="rgba(200,200,200,0.12)", dash="dot"),
            name="baseline",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Placeholder waveform-style backdrop
    try:
        max_rel = df["section_rel_end"].max()
    except Exception:
        max_rel = 0
    t_back = np.linspace(0, max(max_rel, duration_final, 1.0), num=300)
    waveform = 0.6 * np.sin(2 * np.pi * t_back / max(max_rel, 1.0) * 3) + 0.25 * np.sin(2 * np.pi * t_back / max(max_rel, 1.0) * 7)
    fig.add_trace(
        go.Scatter(
            x=t_back,
            y=waveform,
            mode="lines",
            line=dict(color="rgba(120,120,120,0.12)"),
            name="waveform",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Live rolling lines (Q/X/HCE) only
    metrics_selected = metrics_selected or ["q"]
    color_map = {"q": "#d7b34d", "x": "#ff7f7f", "hce": "#9bd19d"}
    label_map = {"q": "Rolling Q (5s)", "x": "Rolling X (5s)", "hce": "Rolling HCE (5s)"}

    def _dot_color(val: float) -> str:
        if val is None or not np.isfinite(val):
            return "#d7b34d"
        if val < 0.3:
            return "#4da6ff"
        if val < 0.8:
            return "#d7b34d"
        return "#ff7ad1"

    for metric in ["q", "x", "hce"]:
        if metric not in metrics_selected:
            continue
        df_live = {"q": live_q, "x": live_x, "hce": live_h}[metric]
        col = {"q": "Q", "x": "X", "hce": "HCE"}[metric]
        if df_live.empty:
            continue
        last_val = float(df_live[col].iloc[-1]) if not df_live.empty else None
        fig.add_trace(
            go.Scatter(
                x=df_live["t"],
                y=df_live[col],
                mode="lines+markers",
                line=dict(color=color_map[metric], width=4),
                marker=dict(color=_dot_color(last_val), size=7, symbol="circle"),
                name=f"{label_map[metric]} (live)",
                hovertemplate="t=%{x:.1f}s<br>value=%{y:.3f}<extra></extra>",
            )
        )

    if all(df.empty for df in [live_q, live_x, live_h]):
        fig.add_annotation(
            x=duration_final * 0.5,
            y=0,
            text="Resonance awakening — playback to map live",
            showarrow=False,
            font=dict(color="#aaaaaa"),
        )

    # Transcendence highlight (live only)
    highlight_on = highlight_trans and ("on" in highlight_trans)
    trans_thresh = None
    if not live_h.empty:
        try:
            trans_thresh = float(np.nanpercentile(live_h["HCE"], 90))
        except Exception:
            trans_thresh = None
    if highlight_on and trans_thresh and not live_h.empty:
        live_trans = live_h[live_h["HCE"] >= trans_thresh]
        if not live_trans.empty:
            fig.add_trace(
                go.Scatter(
                    x=live_trans["t"],
                    y=live_trans["HCE"],
                    mode="markers",
                    marker=dict(color="#ff9ad5", size=8, symbol="circle", line=dict(color="#ff6fb8", width=1)),
                    name="Transcendence (top live)",
                    hovertemplate="HCE=%{y:.2f} at t=%{x:.1f}s<extra></extra>",
                )
            )
    for _, r in df.iterrows():
        fig.add_vrect(
            x0=r["section_rel_start"],
            x1=r["section_rel_end"],
            fillcolor=_color_for(r.get("mean_HCE", 0.0)),
            opacity=0.18,
            line_width=0,
            annotation_text=r.get("section_label", ""),
            annotation_position="top left",
        )
    fig.add_trace(
        go.Scatter(
            x=df["section_rel_start"],
            y=df["mean_HCE"],
            mode="lines+markers",
            line=dict(color="#d7b34d", width=3),
            marker=dict(size=10, color=df["mean_Q"], colorscale="Plasma", showscale=True, colorbar=dict(title="Q")),
            name="HCE",
            hovertext=[
                f"{lbl}: HCE {h:.2f}, lift {((h - track_mean)/track_mean*100) if track_mean else 0:+.1f}%, X {x:.3f}, Q {q:.3f}, src {s}"
                for lbl, h, x, q, s in zip(
                    df["section_label"],
                    df["mean_HCE"],
                    df["mean_X"],
                    df["mean_Q"],
                    df["source"] if "source" in df.columns else [""] * len(df),
                )
            ],
        )
    )
    # Approximate per-section Q/X as step lines to keep waveform non-blank
    fig.add_trace(
        go.Scatter(
            x=df["section_rel_start"],
            y=df["mean_Q"],
            mode="lines",
            line=dict(color="rgba(80,180,255,0.35)"),
            fill="tozeroy",
            name="Q (section avg)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["section_rel_start"],
            y=df["mean_X"],
            mode="lines",
            line=dict(color="rgba(200,80,80,0.35)", dash="dash"),
            name="X (section avg)",
        )
    )
    # Peak annotations for HCE > 1.0 (dots, not stars) — live only
    if not live_h.empty:
        peaks = live_h[live_h["HCE"] > 1.0]
        if not peaks.empty:
            fig.add_trace(
                go.Scatter(
                    x=peaks["t"],
                    y=peaks["HCE"],
                    mode="markers",
                    marker=dict(symbol="circle", color="#ffd700", size=8, line=dict(color="#d7b34d", width=1)),
                    name="Fractal Portal (HCE>1.0)",
                    hovertemplate="Fractal Portal: HCE=%{y:.2f}<br>t=%{x:.1f}s<extra></extra>",
                )
            )
    fig.update_layout(
        template="plotly_dark",
        title="Live Resonance — rolling Q/X/HCE (pure live)",
        xaxis_title="Seconds (relative)",
        yaxis_title="Value",
        height=480,
        legend=dict(x=1.02, y=1, bgcolor="rgba(0,0,0,0)", orientation="v"),
    )
    if progress_rel is not None:
        fig.add_vline(x=progress_rel, line=dict(color="#ffbf00", width=2, dash="dash"), name="Now")
        fig.update_layout(title=f"Sections — live at {progress_rel:.1f}s ({live_section_label or 'current'})")
    oracle_mini = ""
    if click_data and click_data.get("points"):
        pt = click_data["points"][0]
        lbl = pt.get("text") or ""
        y = pt.get("y")
        lift_pct = ((y - track_mean) / track_mean * 100) if track_mean else 0.0
        oracle_mini = html.Div(f"{lbl}: resonant lift {lift_pct:+.1f}%", className="text-warning")

    source_used = ""
    if "source" in df.columns and not df["source"].empty:
        mode_vals = df["source"].mode()
        if not mode_vals.empty:
            source_used = mode_vals.iloc[0]

    def _lift_class(pct: float) -> str:
        if pct > 10:
            return "text-success fw-bold"
        if pct < -10:
            return "text-secondary"
        return "text-info"

    def _lift_class(pct: float, src: str) -> str:
        base = "fw-bold" if pct != 0 else ""
        if pct > 10:
            return f"text-success {base}"
        if pct < -10:
            return f"text-purple {base}"
        # neutral / subtle gold
        return f"text-warning {base}" if src == "live" else base

    # Recompute per-section means from binned HCE (live preferred)
    mapped_points = (
        int(live_q["Q"].count()) + int(live_x["X"].count()) + int(live_h["HCE"].count())
        if not all(df.empty for df in [live_q, live_x, live_h])
        else 0
    )
    total_bins = int(np.ceil(duration_final)) if duration_final else 0
    bins_edges = np.linspace(0, max(duration_final, 1.0), 6)
    per_section_rows = []
    running_mean_candidates = []
    if not live_q.empty:
        running_mean_candidates.append(float(live_q["Q"].mean()))
    if not live_h.empty:
        running_mean_candidates.append(float(live_h["HCE"].mean()))
    if not live_x.empty:
        running_mean_candidates.append(float(live_x["X"].mean()))
    running_mean = float(np.nanmean(running_mean_candidates)) if running_mean_candidates else track_mean
    running_mean = running_mean or track_mean
    personal_low = min([v for v in [running_mean, track_mean] if v]) if (running_mean or track_mean) else None
    portal_count = 0
    for i in range(5):
        start = bins_edges[i]
        end = bins_edges[i + 1]
        mask_live = (not live_h.empty) and (live_h["t"] >= start) & (live_h["t"] < end)
        vals_live = live_h.loc[mask_live, "HCE"].dropna() if isinstance(mask_live, pd.Series) else pd.Series([], dtype=float)
        if len(vals_live) > 0:
            hce_mean = float(vals_live.mean())
            gran_source = "live"
        else:
            hce_mean = running_mean if running_mean is not None else 0.0
            gran_source = "track avg"
        q_mean = float(df.iloc[i].get("mean_Q", 0.0)) if i < len(df) else 0.0
        x_mean = float(df.iloc[i].get("mean_X", 0.0)) if i < len(df) else 0.0
        lift_session = ((hce_mean - running_mean) / running_mean * 100) if (running_mean and not np.isnan(hce_mean)) else None
        lift_track = ((hce_mean - track_mean) / track_mean * 100) if (track_mean and not np.isnan(hce_mean)) else None
        lift_low = ((hce_mean - personal_low) / personal_low * 100) if (personal_low and not np.isnan(hce_mean)) else None
        lifts = [v for v in [lift_session, lift_track, lift_low] if v is not None]
        lift_best = max(lifts) if lifts else None
        portal_flag = (lift_best is not None and lift_best > 100) or (hce_mean is not None and hce_mean > 1.5)
        if portal_flag:
            portal_count += 1
        per_section_rows.append(
            {
                "label": df.iloc[i].get("section_label", f"Part {i+1}") if i < len(df) else f"Part {i+1}",
                "start": start,
                "end": end,
                "hce": hce_mean,
                "q": q_mean,
                "x": x_mean,
                "lift_best": lift_best,
                "lift_session": lift_session,
                "lift_track": lift_track,
                "lift_low": lift_low,
                "portal": portal_flag,
                "src": gran_source,
            }
        )

    table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Section"),
                        html.Th("Start (s)"),
                        html.Th("End (s)"),
                        html.Th("HCE"),
                        html.Th("Lift vs track %"),
                        html.Th("Q"),
                        html.Th("X"),
                        html.Th("Source"),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(r.get("label")),
                            html.Td(f"{r.get('start',0):.1f}"),
                            html.Td(f"{r.get('end',0):.1f}"),
                            html.Td("-" if np.isnan(r.get("hce", np.nan)) else f"{r.get('hce',0):.2f}"),
                            html.Td("-" if np.isnan(r.get("q", np.nan)) else f"{r.get('q',0):.3f}"),
                            html.Td("-" if np.isnan(r.get("x", np.nan)) else f"{r.get('x',0):.3f}"),
                            html.Td(
                                "-" if r.get("lift_best") is None else f"{r.get('lift_best',0):+.1f}%",
                                className=(
                                    "text-warning fw-bold"
                                    if r.get("lift_best") is not None and r.get("lift_best") > 0
                                    else "text-secondary"
                                ),
                            ),
                            html.Td("⭐⭐⭐" if r.get("portal") else ""),
                            html.Td("-" if np.isnan(r.get("q", np.nan)) else f"{r.get('q',0):.3f}"),
                            html.Td(r.get("src", "")),
                        ]
                    )
                    for r in per_section_rows
                ]
            ),
        ],
        striped=True,
        bordered=False,
        hover=True,
        size="sm",
        className="mt-2",
    )
    live_bins = mapped_points
    total_bins = int(np.ceil(duration_final)) if duration_final else 0
    granularity = f"Granularity: live seconds mapped {live_bins}/{total_bins} (building resonance)"
    early_note = ""
    if total_bins and live_bins / max(total_bins, 1) < 0.2:
        early_note = "Resonance awakening — play longer for depth."
    footer = html.Div(
        [
            html.Div("Resonance building — live rolling metrics only (pure nowcast).", className="text-muted small"),
            html.Div(granularity, className="text-muted small"),
            (html.Div(early_note, className="text-warning small") if early_note else None),
            html.Div(
                f"{portal_count} Fractal Resonance Portals detected (consistent elevation windows)",
                className="text-info small",
            ),
        ]
    )

    session_count = 0
    variation = (df["mean_HCE"].max() - df["mean_HCE"].min()) if not df.empty else 0.0
    top_lift = None
    # Debug logging
    try:
        rows = storage.list_track_sections(track_id, limit_sessions=None)
        session_count = len({r.get("track_session_id") for r in rows}) if rows else 0
        if not df.empty and track_mean:
            top_row = df.iloc[df["mean_HCE"].idxmax()]
            top_lift = ((top_row["mean_HCE"] - track_mean) / track_mean * 100) if track_mean else 0.0
        print(
            f"[media_alchemy] track={track_id} title={title} artist={artist} sessions={session_count} "
            f"track_mean={track_means.get('mean_HCE',0):.2f} variation={variation:.2f} "
            f"section_source={section_source_note} top_lift={top_lift:+.1f}% "
            f"live_points={mapped_points} "
            f"sections={[(r.get('label'), r.get('hce'), r.get('src')) for r in per_section_rows]}",
            flush=True,
        )
        print(
            f"[media_alchemy] Table updated: {mapped_points} points across {len(per_section_rows)} sections; live buckets={live_bucket_count}",
            flush=True,
        )
    except Exception:
        pass

    # Radar chart for personal resonance (historical lift proxy)
    radar_fig = None
    try:
        lib_means = storage.get_track_library_avg(track_id)
        if lib_means:
            radar_features = {
                "hce": lib_means.get("mean_HCE", 0),
                "q": lib_means.get("mean_Q", 0),
                "x": lib_means.get("mean_X", 0),
                "richness": variation,
                "expected": top_lift if top_lift is not None else 0.0,
            }
            # Normalize to 0..1 roughly
            radar_norm = {k: max(min(float(v) / 100.0, 1.0), 0.0) for k, v in radar_features.items()}
            radar_categories = list(radar_norm.keys())
            radar_values = list(radar_norm.values())
            radar_fig = go.Figure()
            radar_fig.add_trace(
                go.Scatterpolar(
                    r=radar_values + [radar_values[0]],
                    theta=radar_categories + [radar_categories[0]],
                    fill="toself",
                    name="Personal resonance",
                    line=dict(color="#FFD700"),
                )
            )
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                template="plotly_dark",
                height=280,
                margin=dict(l=20, r=20, t=20, b=20),
            )
    except Exception:
        pass

    status_txt = f"{section_source_note}; live_section={live_section_label or 'n/a'}; metrics={metrics_selected}"
    children = [table, footer]
    if radar_fig:
        children.insert(0, dcc.Graph(figure=radar_fig, style={"height": "300px"}))
    return fig, html.Div(children), oracle_mini, status_txt

