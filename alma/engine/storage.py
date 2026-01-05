"""Lightweight SQLite storage for sessions, buckets, events, and Spotify playback."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from alma import config


def _connect() -> sqlite3.Connection:
    config.ensure_required_paths()
    db_path = Path(config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db() -> None:
    """Create tables if they do not exist."""
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                started_at REAL,
                ended_at REAL,
                profile_path TEXT,
                muse_address TEXT,
                lsl_stream_name TEXT,
                ndjson_path TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS buckets (
                bucket_start_ts REAL,
                bucket_end_ts REAL,
                session_id TEXT,
                mean_X REAL,
                mean_Q REAL,
                mean_HCE REAL,
                std_Q REAL,
                Q_slope REAL,
                valid_fraction REAL,
                label TEXT,
                PRIMARY KEY(bucket_start_ts, session_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schedule_blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                block_type TEXT,
                duration_min INTEGER,
                flexible INTEGER,
                start_ts REAL,
                end_ts REAL,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                ts REAL,
                session_id TEXT,
                kind TEXT,
                label TEXT,
                note TEXT,
                tags_json TEXT,
                context_json TEXT,
                snapshot_json TEXT
            )
            """
        )
        # backfill columns for existing installs
        try:
            cur.execute("ALTER TABLE buckets ADD COLUMN mean_HCE REAL")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE events ADD COLUMN snapshot_json TEXT")
        except Exception:
            pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS spotify_playback (
                ts REAL,
                session_id TEXT,
                is_playing INTEGER,
                track_id TEXT,
                track_name TEXT,
                artists TEXT,
                album TEXT,
                context_uri TEXT,
                device_name TEXT,
                progress_ms INTEGER,
                duration_ms INTEGER,
                mode TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS track_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                track_id TEXT,
                title TEXT,
                artist TEXT,
                album TEXT,
                start_ts REAL,
                end_ts REAL,
                mean_HCE REAL,
                mean_Q REAL,
                mean_X REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS track_sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_session_id INTEGER,
                track_id TEXT,
                title TEXT,
                artist TEXT,
                section_index INTEGER,
                section_label TEXT,
                start_ts REAL,
                end_ts REAL,
                mean_HCE REAL,
                mean_Q REAL,
                mean_X REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                description TEXT,
                mode TEXT,
                target_json TEXT,
                steps_json TEXT,
                stats_json TEXT,
                efficacy_score REAL
            )
            """
        )
        # Backfill stats_json column if table exists without it
        try:
            cur.execute("ALTER TABLE recipes ADD COLUMN stats_json TEXT")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE recipes ADD COLUMN description TEXT")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE recipes ADD COLUMN steps_json TEXT")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE recipes ADD COLUMN efficacy_score REAL")
        except Exception:
            pass
        conn.commit()


def add_schedule_block(title: str, block_type: str, duration_min: int, flexible: bool, start_ts: float, end_ts: float) -> None:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO schedule_blocks (title, block_type, duration_min, flexible, start_ts, end_ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (title, block_type, int(duration_min), 1 if flexible else 0, start_ts, end_ts),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_schedule_blocks_for_date(date_str: str) -> List[Dict[str, object]]:
    ts0, ts1 = _ts_bounds_for_date(date_str)
    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT id, title, block_type, duration_min, flexible, start_ts, end_ts
            FROM schedule_blocks
            WHERE start_ts >= ? AND start_ts < ?
            ORDER BY start_ts ASC
            """,
            (ts0, ts1),
        )
        rows = _rows_to_dicts(cur)
        for r in rows:
            r["flexible"] = bool(r.get("flexible"))
        return rows


def update_schedule_block(block_id: int, start_ts: float, end_ts: float, duration_min: Optional[int] = None, title: Optional[str] = None) -> None:
    """Update timing (and optionally duration/title) for a scheduled block."""
    if block_id is None:
        return
    fields = ["start_ts = ?", "end_ts = ?"]
    params: List[object] = [start_ts, end_ts]
    if duration_min is not None:
        fields.append("duration_min = ?")
        params.append(int(duration_min))
    if title is not None:
        fields.append("title = ?")
        params.append(title)
    params.append(block_id)
    with _connect() as conn:
        conn.execute(f"UPDATE schedule_blocks SET {', '.join(fields)} WHERE id = ?", params)
        conn.commit()


# Recipes
def _compute_efficacy_from_stats(stats: Dict[str, object]) -> float:
    runs = int(stats.get("runs", 0) or 0)
    successes = int(stats.get("successes", 0) or 0)
    if runs <= 0:
        return 0.0
    return round(successes / runs, 3)


def list_recipes() -> List[Dict[str, object]]:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT id, name, description, mode, target_json, steps_json, stats_json, efficacy_score FROM recipes ORDER BY id DESC"
        )
        rows = _rows_to_dicts(cur)
        for r in rows:
            try:
                r["target_json"] = json.loads(r.get("target_json") or "{}")
            except Exception:
                r["target_json"] = {}
            try:
                r["steps_json"] = json.loads(r.get("steps_json") or "[]")
            except Exception:
                r["steps_json"] = []
            try:
                r["stats_json"] = json.loads(r.get("stats_json") or "{}")
            except Exception:
                r["stats_json"] = {}
            # Ensure efficacy_score present
            if r.get("efficacy_score") is None:
                r["efficacy_score"] = _compute_efficacy_from_stats(r.get("stats_json") or {})
        return rows


def upsert_recipe(
    recipe_id: Optional[int],
    name: str,
    mode: str,
    target_json: dict,
    steps_json: List[str],
    description: str = "",
) -> int:
    steps_payload = json.dumps(steps_json or [])
    with _connect() as conn:
        if recipe_id:
            conn.execute(
                "UPDATE recipes SET name=?, description=?, mode=?, target_json=?, steps_json=? WHERE id=?",
                (name, description, mode, json.dumps(target_json or {}), steps_payload, recipe_id),
            )
            conn.commit()
            return recipe_id
        cur = conn.execute(
            """
            INSERT INTO recipes (name, description, mode, target_json, steps_json, stats_json, efficacy_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (name, description, mode, json.dumps(target_json or {}), steps_payload, json.dumps({}), None),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_recipe_by_id(recipe_id: int) -> Optional[Dict[str, object]]:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT id, name, description, mode, target_json, steps_json, stats_json, efficacy_score FROM recipes WHERE id=?",
            (recipe_id,),
        )
        rows = _rows_to_dicts(cur)
        if not rows:
            return None
        r = rows[0]
        try:
            r["target_json"] = json.loads(r.get("target_json") or "{}")
        except Exception:
            r["target_json"] = {}
        try:
            r["steps_json"] = json.loads(r.get("steps_json") or "[]")
        except Exception:
            r["steps_json"] = []
        try:
            r["stats_json"] = json.loads(r.get("stats_json") or "{}")
        except Exception:
            r["stats_json"] = {}
        if r.get("efficacy_score") is None:
            r["efficacy_score"] = _compute_efficacy_from_stats(r.get("stats_json") or {})
        return r


def delete_recipe(recipe_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM recipes WHERE id=?", (recipe_id,))
        conn.commit()


def start_session(profile_path: str, muse_address: str, lsl_stream_name: Optional[str], ndjson_path: str) -> str:
    init_db()
    session_id = str(uuid.uuid4())
    started_at = time.time()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions (id, started_at, ended_at, profile_path, muse_address, lsl_stream_name, ndjson_path)
            VALUES (?, ?, NULL, ?, ?, ?, ?)
            """,
            (session_id, started_at, profile_path, muse_address, lsl_stream_name, ndjson_path),
        )
        conn.commit()
    return session_id


def end_session(session_id: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE sessions SET ended_at=? WHERE id=?", (time.time(), session_id))
        conn.commit()


def upsert_bucket(
    bucket_start_ts: float,
    bucket_end_ts: float,
    session_id: str,
    mean_X: float,
    mean_Q: float,
    mean_HCE: float,
    std_Q: float,
    Q_slope: float,
    valid_fraction: float,
    label: str,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO buckets
            (bucket_start_ts, bucket_end_ts, session_id, mean_X, mean_Q, mean_HCE, std_Q, Q_slope, valid_fraction, label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bucket_start_ts,
                bucket_end_ts,
                session_id,
                mean_X,
                mean_Q,
                mean_HCE,
                std_Q,
                Q_slope,
                valid_fraction,
                label,
            ),
        )
        conn.commit()


def insert_event(
    ts: float,
    session_id: str,
    kind: str,
    label: str = "",
    note: str = "",
    tags_json: Optional[dict] = None,
    context_json: Optional[dict] = None,
    snapshot_json: Optional[dict] = None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO events (ts, session_id, kind, label, note, tags_json, context_json, snapshot_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                session_id,
                kind,
                label,
                note,
                json.dumps(tags_json or {}),
                json.dumps(context_json or {}),
                json.dumps(snapshot_json or {}),
            ),
        )
        conn.commit()


def insert_spotify_row(
    ts: float,
    session_id: str,
    is_playing: bool,
    track_id: str = "",
    track_name: str = "",
    artists: str = "",
    album: str = "",
    context_uri: str = "",
    device_name: str = "",
    progress_ms: Optional[int] = None,
    duration_ms: Optional[int] = None,
    mode: str = "",
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO spotify_playback
            (ts, session_id, is_playing, track_id, track_name, artists, album, context_uri, device_name, progress_ms, duration_ms, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                session_id,
                1 if is_playing else 0,
                track_id,
                track_name,
                artists,
                album,
                context_uri,
                device_name,
                progress_ms,
                duration_ms,
                mode,
            ),
        )
        conn.commit()


def _rows_to_dicts(cursor) -> List[Dict[str, object]]:
    cols = [c[0] for c in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _ts_bounds_for_date(date_str: str, tz_local: bool = True) -> tuple[float, float]:
    if tz_local:
        start = datetime.fromisoformat(date_str)
    else:
        start = datetime.fromisoformat(date_str)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


def get_buckets_for_date(date_str: str, tz_local: bool = True) -> List[Dict[str, object]]:
    ts0, ts1 = _ts_bounds_for_date(date_str, tz_local=tz_local)
    return get_buckets_between(ts0, ts1, session_id=None)


def get_buckets_between(ts0: float, ts1: float, session_id: Optional[str] = None) -> List[Dict[str, object]]:
    query = """
        SELECT bucket_start_ts, bucket_end_ts, session_id, mean_X, mean_Q, mean_HCE, std_Q, Q_slope, valid_fraction, label
        FROM buckets
        WHERE bucket_start_ts >= ? AND bucket_end_ts <= ?
    """
    params: List[object] = [ts0, ts1]
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    query += " ORDER BY bucket_start_ts ASC"
    with _connect() as conn:
        cur = conn.execute(query, params)
        return _rows_to_dicts(cur)


def get_bucket_by_start(bucket_start_ts: float, session_id: str) -> Optional[Dict[str, object]]:
    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT bucket_start_ts, bucket_end_ts, session_id, mean_X, mean_Q, mean_HCE, std_Q, Q_slope, valid_fraction, label
            FROM buckets
            WHERE bucket_start_ts = ? AND session_id = ?
            """,
            (bucket_start_ts, session_id),
        )
        rows = _rows_to_dicts(cur)
        return rows[0] if rows else None


def get_events_between(ts0: float, ts1: float, session_id: Optional[str] = None) -> List[Dict[str, object]]:
    query = """
        SELECT ts, session_id, kind, label, note, tags_json, context_json
        FROM events
        WHERE ts >= ? AND ts <= ?
    """
    params: List[object] = [ts0, ts1]
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    query += " ORDER BY ts ASC"
    with _connect() as conn:
        cur = conn.execute(query, params)
        rows = _rows_to_dicts(cur)
        for r in rows:
            r["tags_json"] = json.loads(r.get("tags_json") or "{}")
            r["context_json"] = json.loads(r.get("context_json") or "{}")
        return rows


def get_spotify_between(ts0: float, ts1: float, session_id: Optional[str] = None) -> List[Dict[str, object]]:
    query = """
        SELECT ts, session_id, is_playing, track_id, track_name, artists, album, context_uri, device_name, progress_ms, duration_ms, mode
        FROM spotify_playback
        WHERE ts >= ? AND ts <= ?
    """
    params: List[object] = [ts0, ts1]
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    query += " ORDER BY ts ASC"
    with _connect() as conn:
        cur = conn.execute(query, params)
        return _rows_to_dicts(cur)


def get_latest_spotify(session_id: Optional[str] = None) -> Optional[Dict[str, object]]:
    query = """
        SELECT ts, session_id, is_playing, track_id, track_name, artists, album, context_uri, device_name, progress_ms, duration_ms, mode
        FROM spotify_playback
    """
    params: List[object] = []
    if session_id:
        query += " WHERE session_id = ?"
        params.append(session_id)
    query += " ORDER BY ts DESC LIMIT 1"
    with _connect() as conn:
        cur = conn.execute(query, params)
        rows = _rows_to_dicts(cur)
        return rows[0] if rows else None


# Track sessions (Spotify)
def start_track_session(
    session_id: Optional[str],
    track_id: str,
    title: str,
    artist: str,
    album: str,
    start_ts: float,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO track_sessions (session_id, track_id, title, artist, album, start_ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, track_id, title, artist, album, start_ts),
        )
        conn.commit()
        return int(cur.lastrowid)


def _compute_means_for_window(ts0: float, ts1: float, session_id: Optional[str]) -> Dict[str, float]:
    buckets = get_buckets_between(ts0, ts1, session_id=session_id)
    if not buckets:
        return {"mean_HCE": 0.0, "mean_Q": 0.0, "mean_X": 0.0}
    mean_HCE = float(np.nanmean([b.get("mean_HCE", 0.0) for b in buckets]))
    mean_Q = float(np.nanmean([b.get("mean_Q", 0.0) for b in buckets]))
    mean_X = float(np.nanmean([b.get("mean_X", 0.0) for b in buckets]))
    return {"mean_HCE": mean_HCE, "mean_Q": mean_Q, "mean_X": mean_X}


def update_track_session_metrics(track_session_id: int, end_ts: float) -> None:
    with _connect() as conn:
        cur = conn.execute("SELECT start_ts, session_id FROM track_sessions WHERE id=?", (track_session_id,))
        row = cur.fetchone()
        if not row:
            return
        start_ts, session_id = row
        means = _compute_means_for_window(start_ts, end_ts, session_id=session_id)
        conn.execute(
            """
            UPDATE track_sessions
            SET end_ts=?, mean_HCE=?, mean_Q=?, mean_X=?
            WHERE id=?
            """,
            (end_ts, means["mean_HCE"], means["mean_Q"], means["mean_X"], track_session_id),
        )
        conn.commit()


def upsert_track_sections(track_session_id: int, track_id: str, title: str, artist: str, sections: List[Dict[str, float]]) -> None:
    """Persist per-section metrics for a track session."""
    if not sections:
        return
    with _connect() as conn:
        conn.execute("DELETE FROM track_sections WHERE track_session_id=?", (track_session_id,))
        conn.executemany(
            """
            INSERT INTO track_sections (track_session_id, track_id, title, artist, section_index, section_label, start_ts, end_ts, mean_HCE, mean_Q, mean_X)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    track_session_id,
                    track_id,
                    title,
                    artist,
                    int(s.get("idx", 0)),
                    s.get("label") or f"Section {s.get('idx', 0)}",
                    float(s.get("start_ts", 0.0)),
                    float(s.get("end_ts", 0.0)),
                    float(s.get("mean_HCE", 0.0)),
                    float(s.get("mean_Q", 0.0)),
                    float(s.get("mean_X", 0.0)),
                )
                for s in sections
            ],
        )
        conn.commit()


def list_track_sections(track_id: str, limit_sessions: int = 1) -> List[Dict[str, object]]:
    """Fetch latest sections for a track (most recent sessions)."""
    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT ts.id AS track_session_id, ts.title, ts.artist, ts.start_ts, ts.end_ts, sec.section_index,
                   sec.section_label, sec.start_ts AS section_start_ts, sec.end_ts AS section_end_ts,
                   sec.mean_HCE, sec.mean_Q, sec.mean_X
            FROM track_sessions ts
            JOIN track_sections sec ON ts.id = sec.track_session_id
            WHERE ts.track_id = ?
            ORDER BY ts.start_ts DESC, sec.section_index ASC
            """,
            (track_id,),
        )
        rows = _rows_to_dicts(cur)
        if not rows:
            return []
        # limit sessions by unique track_session_id
        seen = set()
        limited = []
        for r in rows:
            tsid = r.get("track_session_id")
            if len(seen) >= limit_sessions and tsid not in seen:
                continue
            seen.add(tsid)
            limited.append(r)
        return limited


def list_top_sections(limit: int = 10) -> List[Dict[str, object]]:
    """Aggregate top-performing sections across history."""
    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT section_label,
                   AVG(mean_HCE) AS avg_hce,
                   AVG(mean_Q) AS avg_q,
                   AVG(mean_X) AS avg_x,
                   COUNT(*) AS plays
            FROM track_sections
            GROUP BY section_label
            ORDER BY avg_hce DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = _rows_to_dicts(cur)
        for r in rows:
            r["avg_hce"] = float(r.get("avg_hce") or 0.0)
            r["avg_q"] = float(r.get("avg_q") or 0.0)
            r["avg_x"] = float(r.get("avg_x") or 0.0)
            r["plays"] = int(r.get("plays") or 0)
        return rows


def list_top_tracks(limit: int = 10, offset: int = 0) -> List[Dict[str, object]]:
    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT track_id, title, artist, album,
                   AVG(mean_HCE) AS avg_hce,
                   COUNT(*) AS play_count
            FROM track_sessions
            WHERE mean_HCE IS NOT NULL
            GROUP BY track_id, title, artist, album
            ORDER BY avg_hce DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = _rows_to_dicts(cur)
        for r in rows:
            r["avg_hce"] = float(r.get("avg_hce") or 0.0)
            r["play_count"] = int(r.get("play_count") or 0)
        return rows


def list_recent_events(limit: int = 20, session_id: Optional[str] = None) -> List[Dict[str, object]]:
    query = """
        SELECT ts, session_id, kind, label, note, tags_json, context_json
        FROM events
    """
    params: List[object] = []
    if session_id:
        query += " WHERE session_id = ?"
        params.append(session_id)
    query += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    with _connect() as conn:
        cur = conn.execute(query, params)
        rows = _rows_to_dicts(cur)
        for r in rows:
            r["tags_json"] = json.loads(r.get("tags_json") or "{}")
            r["context_json"] = json.loads(r.get("context_json") or "{}")
        return rows


def get_context_for_window(ts0: float, ts1: float, session_id: Optional[str] = None) -> Dict[str, object]:
    buckets = get_buckets_between(ts0, ts1, session_id=session_id)
    events = get_events_between(ts0, ts1, session_id=session_id)
    spotify_rows = get_spotify_between(ts0, ts1, session_id=session_id)

    top_tracks = []
    if spotify_rows:
        key_counts = Counter()
        last_ts: Dict[str, float] = {}
        for row in spotify_rows:
            key = (row.get("track_name") or "", row.get("artists") or "")
            key_counts[key] += 1
            last_ts[key] = row.get("ts", 0)
        top = key_counts.most_common(5)
        top_tracks = [
            {"track_name": k[0], "artists": k[1], "count": cnt, "last_ts": last_ts.get(k)}
            for (k, cnt) in top
        ]

    bucket_summary = None
    if buckets:
        last = buckets[-1]
        bucket_summary = {
            "label": last.get("label"),
            "mean_Q": last.get("mean_Q"),
            "mean_X": last.get("mean_X"),
            "std_Q": last.get("std_Q"),
            "Q_slope": last.get("Q_slope"),
            "bucket_start_ts": last.get("bucket_start_ts"),
            "bucket_end_ts": last.get("bucket_end_ts"),
        }

    return {
        "top_tracks": top_tracks,
        "events": events,
        "bucket_summary": bucket_summary,
    }


def _aggregate_buckets(buckets: List[Dict[str, object]]) -> Optional[Dict[str, float]]:
    if not buckets:
        return None
    mean_X = float(np.nanmean([b.get("mean_X", 0.0) for b in buckets]))
    mean_Q = float(np.nanmean([b.get("mean_Q", 0.0) for b in buckets]))
    mean_HCE = float(np.nanmean([b.get("mean_HCE", 0.0) for b in buckets]))
    std_Q = float(np.nanmean([b.get("std_Q", 0.0) for b in buckets]))
    slope = float(np.nanmean([b.get("Q_slope", 0.0) for b in buckets]))
    return {"mean_X": mean_X, "mean_Q": mean_Q, "mean_HCE": mean_HCE, "std_Q": std_Q, "slope": slope}


def score_recipe_run(recipe_id: int, end_ts: float) -> None:
    """Update recipe stats_json with run result using buckets around the run."""
    recipe = get_recipe_by_id(recipe_id)
    if not recipe:
        return
    target = recipe.get("target_json") or {}
    defaults = {
        "mean_HCE_min": 0.0,
    }
    thresholds = {**defaults, **target}

    # Find the most recent recipe_start for this recipe before end_ts
    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT ts FROM events
            WHERE kind='recipe_start'
              AND tags_json LIKE ?
              AND ts <= ?
            ORDER BY ts DESC
            LIMIT 1
            """,
            (f'%\"recipe_id\": {recipe_id}%', end_ts),
        )
        row = cur.fetchone()
        if not row:
            return
        start_ts = row[0]

    pre_start = start_ts - 600  # 10 minutes before
    pre_end = start_ts
    during_start = start_ts
    during_end = end_ts

    pre_buckets = get_buckets_between(pre_start, pre_end, session_id=None)
    during_buckets = get_buckets_between(during_start, during_end, session_id=None)

    pre_feat = _aggregate_buckets(pre_buckets) or {}
    during_feat = _aggregate_buckets(during_buckets) or {}

    success = False
    if during_feat:
        success = during_feat.get("mean_HCE", 0.0) >= thresholds.get("mean_HCE_min", 0.0)
    delta_hce = 0.0
    if during_feat:
        delta_hce = float(during_feat.get("mean_HCE", 0.0) - thresholds.get("mean_HCE_min", 0.0))

    stats = recipe.get("stats_json") or {}
    runs = int(stats.get("runs", 0)) + 1
    successes = int(stats.get("successes", 0)) + (1 if success else 0)
    stats["runs"] = runs
    stats["successes"] = successes
    stats["last_run_summary"] = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "pre": pre_feat,
        "during": during_feat,
        "success": success,
        "delta_HCE": delta_hce,
    }
    efficacy_score = _compute_efficacy_from_stats(stats)

    with _connect() as conn:
        conn.execute(
            "UPDATE recipes SET stats_json=?, efficacy_score=? WHERE id=?",
            (json.dumps(stats), efficacy_score, recipe_id),
        )
        conn.commit()


__all__ = [
    "init_db",
    "start_session",
    "end_session",
    "upsert_bucket",
    "insert_event",
    "insert_spotify_row",
    "get_buckets_for_date",
    "get_buckets_between",
    "get_bucket_by_start",
    "get_events_between",
    "get_spotify_between",
    "get_latest_spotify",
    "list_recent_events",
    "get_context_for_window",
    "add_schedule_block",
    "update_schedule_block",
    "list_schedule_blocks_for_date",
    "start_track_session",
    "update_track_session_metrics",
    "list_top_tracks",
    "list_recipes",
    "upsert_recipe",
    "delete_recipe",
    "get_recipe_by_id",
]

