"""Lightweight LSL helpers for resolving EEG streams."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pylsl import StreamInlet, resolve_byprop, resolve_streams


def list_streams(timeout: float = 1.0) -> List[Dict[str, object]]:
    """Return a list of available LSL streams with basic metadata."""
    streams = []
    try:
        for info in resolve_streams(wait_time=timeout):
            streams.append(
                {
                    "name": info.name() if hasattr(info, "name") else None,
                    "type": info.type() if hasattr(info, "type") else None,
                    "channel_count": info.channel_count() if hasattr(info, "channel_count") else None,
                    "nominal_srate": info.nominal_srate() if hasattr(info, "nominal_srate") else None,
                    "source_id": info.source_id() if hasattr(info, "source_id") else None,
                }
            )
    except Exception:
        return []
    return streams


def resolve_preferred_stream(
    preferred_name: Optional[str] = None, timeout: float = 2.0
) -> Tuple[Optional[StreamInlet], Optional[object]]:
    """Resolve an EEG stream, honoring a preferred name when provided."""
    try_order = []
    if preferred_name:
        try_order.append(lambda: resolve_byprop("name", preferred_name, timeout=timeout))
    try_order.extend(
        [
            lambda: resolve_byprop("type", "EEG", timeout=timeout),
            lambda: resolve_streams(wait_time=timeout),
        ]
    )

    results = None
    chosen = None
    for resolver in try_order:
        try:
            results = resolver()
        except Exception:
            results = None
        if results:
            if resolver == try_order[-1]:
                chosen = _choose_best_stream(results)
            else:
                chosen = results[0]
            if chosen:
                return StreamInlet(chosen), chosen
    return None, None


def _choose_best_stream(stream_infos) -> Optional[object]:
    """Pick the most EEG-like stream."""
    if not stream_infos:
        return None

    def score(info) -> int:
        s = 0
        try:
            if hasattr(info, "type") and info.type() == "EEG":
                s += 3
            if hasattr(info, "name") and "Muse" in (info.name() or ""):
                s += 2
            if hasattr(info, "channel_count") and (info.channel_count() or 0) >= 4:
                s += 1
        except Exception:
            pass
        return s

    best = max(stream_infos, key=score)
    return best


__all__ = ["resolve_preferred_stream", "list_streams"]

