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


def _historical_hce_series(track_id: str, bin_sec: float = 1.0, max_sessions: int = 80) -> Tuple[pd.DataFrame, float]:
    """Aggregate historical HCE across sessions into per-second bins; returns df and max duration."""
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
        return pd.DataFrame(columns=["t", "HCE"]), 0.0
    if sess_df.empty:
        return pd.DataFrame(columns=["t", "HCE"]), 0.0

    bins: Dict[int, list] = {}
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
            bins.setdefault(idx, []).append(b.get("mean_HCE", 0.0) or 0.0)

    if not bins:
        return pd.DataFrame(columns=["t", "HCE"]), max_duration
    max_idx = int(max_duration // bin_sec) + 1
    ts = np.arange(0, max_idx * bin_sec, bin_sec)
    vals = np.full_like(ts, np.nan, dtype=float)
    for idx, arr in bins.items():
        if idx < 0 or idx >= len(vals):
            continue
        vals[idx] = float(np.nanmean(arr))
    series = pd.Series(vals, index=ts)
    series_interp = series.interpolate(limit_direction="both")
    return pd.DataFrame({"t": series_interp.index, "HCE": series_interp.values}), max_duration


def _live_hce_series(track_id: str, duration_hint: float, bin_sec: float = 1.0) -> Tuple[pd.DataFrame, Optional[float], Optional[str], int]:
    """Live per-second HCE for the current playing session (time-windowed by playback duration)."""
    try:
        latest = storage.get_latest_spotify(session_id=None)
    except Exception:
        latest = None
    if not latest or not latest.get("is_playing") or latest.get("track_id") != track_id:
        return pd.DataFrame(columns=["t", "HCE"]), None, None, 0

    progress_ms = latest.get("progress_ms") or 0
    duration_ms = latest.get("duration_ms") or 1
    progress_sec = max(0.0, min(float(progress_ms) / 1000.0, float(duration_ms) / 1000.0))
    duration_cap = max(duration_hint, float(duration_ms) / 1000.0, progress_sec)

    session_row = storage.get_latest_track_session(track_id)
    if not session_row:
        return pd.DataFrame(columns=["t", "HCE"]), progress_sec, None, 0
    start_ts = float(session_row.get("start_ts") or 0.0)
    # pull buckets in window, any session, and filter by track_uri match when available
    buckets = storage.get_buckets_between(start_ts - 30.0, start_ts + duration_cap + bin_sec + 30.0, session_id=None)

    bins: Dict[int, list] = {}
    live_bucket_count = 0
    for b in buckets:
        if b.get("track_uri") and b.get("track_uri") != track_id:
            continue
        # prefer stored relative_seconds if present
        rel = b.get("relative_seconds")
        if rel is None:
            rel = float(b.get("bucket_start_ts", 0.0) - start_ts)
        rel = float(rel)
        if rel < -10.0 or rel > duration_cap + 30.0:
            continue
        idx = int(rel // bin_sec)
        bins.setdefault(idx, []).append(b.get("mean_HCE", 0.0) or 0.0)
        live_bucket_count += 1
    max_idx = int(progress_sec // bin_sec) + 1
    ts = np.arange(0, max_idx * bin_sec, bin_sec)
    vals = np.full_like(ts, np.nan, dtype=float)
    for idx, arr in bins.items():
        if idx < len(vals):
            vals[idx] = float(np.nanmean(arr))
    series = pd.Series(vals, index=ts)
    series_interp = series.interpolate(limit_direction="forward")
    live_df = pd.DataFrame({"t": series_interp.index, "HCE": series_interp.values})

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

    return live_df, progress_sec, live_section, live_bucket_count


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
                            id="ma-show-all-hist",
                            options=[{"label": "Show all historical listens", "value": "all"}],
                            value=[],
                            inline=True,
                            switch=True,
                            className="mb-2",
                        ),
                        dcc.Graph(id="ma-track-plot"),
                        html.Div(id="ma-track-table", className="mt-2"),
                        html.Div(id="ma-oracle-mini", className="mt-2 text-info"),
                        html.Div(id="ma-status", className="small text-muted mt-1"),
                        dcc.Interval(id="ma-interval", interval=8000, n_intervals=0),
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
    Input("ma-show-all-hist", "value"),
)
def _render_track(track_id, click_data, _n, show_hist):
    if not track_id:
        return go.Figure(), html.Div("Select a track to view."), "", ""
    track_means, duration, title, artist = _latest_track_stats(track_id)
    df, section_source_note = _aggregate_sections(track_id)
    if df.empty:
        df = _build_uniform_sections(duration, track_means)
    track_mean = track_means.get("mean_HCE", 0.0) or (df["mean_HCE"].mean() if not df.empty else 0.0)
    avg_hce = df["mean_HCE"].mean() if not df.empty else 0.0
    # Per-second HCE lines
    hist_series, hist_max_dur = _historical_hce_series(track_id, bin_sec=1.0)
    hist_waveforms = []
    if show_hist and "all" in show_hist:
        try:
            wave_rows = storage.list_track_waveforms(track_id, limit=20)
            for row in wave_rows:
                wf = row.get("waveform") or []
                if not wf:
                    continue
                dur = row.get("duration") or len(wf)
                ts = np.arange(0, len(wf))
                hist_waveforms.append((ts, wf))
        except Exception:
            pass
    live_series, progress_rel, live_section_label, live_bucket_count = _live_hce_series(track_id, duration_hint=duration, bin_sec=1.0)
    duration_final = max(duration, hist_max_dur, (progress_rel or 0.0), 1.0)

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
            line=dict(color="rgba(200,200,200,0.1)", dash="dot"),
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
    t = np.linspace(0, max(max_rel, duration_final, 1.0), num=300)
    waveform = 0.6 * np.sin(2 * np.pi * t / max(max_rel, 1.0) * 3) + 0.25 * np.sin(2 * np.pi * t / max(max_rel, 1.0) * 7)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=waveform,
            mode="lines",
            line=dict(color="rgba(120,120,120,0.12)"),
            name="waveform",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Historical HCE line (faint)
    if not hist_series.empty:
        fig.add_trace(
            go.Scatter(
                x=hist_series["t"],
                y=hist_series["HCE"],
                mode="lines",
                line=dict(color="rgba(215,179,77,0.30)", width=3, dash="dash"),
                name="HCE (historical)",
                hovertemplate="t=%{x:.1f}s<br>HCE=%{y:.2f}<extra></extra>",
            )
        )
    if hist_waveforms:
        for ts_arr, wf in hist_waveforms:
            fig.add_trace(
                go.Scatter(
                    x=ts_arr,
                    y=wf,
                    mode="lines",
                    line=dict(color="rgba(215,179,77,0.12)", width=2),
                    name="HCE (past listen)",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        # projection to full duration
        last_val = hist_series["HCE"].iloc[-1] if not hist_series.empty else track_mean
        if duration_final > hist_series["t"].max():
            fig.add_trace(
                go.Scatter(
                    x=[hist_series["t"].max(), duration_final],
                    y=[last_val, last_val],
                    mode="lines",
                    line=dict(color="rgba(215,179,77,0.15)", width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    elif track_mean > 0:
        # fallback flat historical line from library avg
        fig.add_trace(
            go.Scatter(
                x=[0, duration_final],
                y=[track_mean, track_mean],
                mode="lines",
                line=dict(color="rgba(215,179,77,0.15)", width=2, dash="dash"),
                name="HCE (library avg)",
                hoverinfo="skip",
            )
        )
    # Live HCE line (bright) or placeholder
    if not live_series.empty:
        fig.add_trace(
            go.Scatter(
                x=live_series["t"],
                y=live_series["HCE"],
                mode="lines+markers",
                line=dict(color="#d7b34d", width=4),
                marker=dict(color="#ffd15c", size=6),
                name="HCE (live)",
                hovertemplate="t=%{x:.1f}s<br>HCE=%{y:.2f}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=base_x,
                y=[track_mean or 0, track_mean or 0],
                mode="lines",
                line=dict(color="rgba(215,179,77,0.18)", width=2),
                name="HCE (placeholder)",
                hoverinfo="skip",
                showlegend=False,
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
    # Peak annotations for HCE > 1.0
    combined = pd.concat([hist_series, live_series], ignore_index=True) if not hist_series.empty or not live_series.empty else pd.DataFrame(columns=["t", "HCE"])
    if not combined.empty:
        peaks = combined[combined["HCE"] > 1.0]
        if not peaks.empty:
            fig.add_trace(
                go.Scatter(
                    x=peaks["t"],
                    y=peaks["HCE"],
                    mode="markers",
                    marker=dict(symbol="star", color="#ffd700", size=10, line=dict(color="#d7b34d", width=1)),
                    name="Fractal Portal (HCE>1.0)",
                    hovertemplate="Fractal Portal: HCE=%{y:.2f}<br>t=%{x:.1f}s<extra></extra>",
                )
            )
    fig.update_layout(
        template="plotly_dark",
        title="Live HCE waveform — your personal resonance trace",
        xaxis_title="Seconds (relative)",
        yaxis_title="mean_HCE",
        height=480,
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
    mapped_points = int(live_series["HCE"].count()) if not live_series.empty else int(hist_series["HCE"].count()) if not hist_series.empty else 0
    total_bins = int(np.ceil(duration_final)) if duration_final else 0
    bins_edges = np.linspace(0, max(duration_final, 1.0), 6)
    per_section_rows = []
    running_mean = (
        float(live_series["HCE"].mean())
        if not live_series.empty
        else float(hist_series["HCE"].mean()) if not hist_series.empty else track_mean
    )
    running_mean = running_mean or track_mean
    personal_low = min([v for v in [running_mean, track_mean] if v]) if (running_mean or track_mean) else None
    portal_count = 0
    for i in range(5):
        start = bins_edges[i]
        end = bins_edges[i + 1]
        mask_live = (not live_series.empty) and (live_series["t"] >= start) & (live_series["t"] < end)
        mask_hist = (not hist_series.empty) and (hist_series["t"] >= start) & (hist_series["t"] < end)
        vals_live = live_series.loc[mask_live, "HCE"].dropna() if isinstance(mask_live, pd.Series) else pd.Series([], dtype=float)
        vals_hist = hist_series.loc[mask_hist, "HCE"].dropna() if isinstance(mask_hist, pd.Series) else pd.Series([], dtype=float)
        if len(vals_live) > 0:
            hce_mean = float(vals_live.mean())
            gran_source = "live"
        elif len(vals_hist) > 0:
            hce_mean = float(vals_hist.mean())
            gran_source = "historical"
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
    live_bins = int(live_series["HCE"].count()) if not live_series.empty else 0
    total_bins = int(np.ceil(duration_final)) if duration_final else 0
    granularity = f"Granularity: live seconds mapped {live_bins}/{total_bins} (building resonance)"
    early_note = ""
    if total_bins and live_bins / max(total_bins, 1) < 0.2:
        early_note = "Resonance awakening — play longer for depth."
    footer = html.Div(
        [
            html.Div(
                "Resonance building — gold line grows as your live HCE arrives. Estimated sections shown when Spotify analysis is restricted.",
                className="text-muted small",
            ),
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
            f"hist_points={len(hist_series)} live_points={len(live_series)} mapped_points={mapped_points} "
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

    status_txt = f"{section_source_note}; sessions={session_count}; live={live_section_label or 'n/a'}"
    children = [table, footer]
    if radar_fig:
        children.insert(0, dcc.Graph(figure=radar_fig, style={"height": "300px"}))
    return fig, html.Div(children), oracle_mini, status_txt

