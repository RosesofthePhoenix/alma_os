"""Manage Turrell Room runner subprocess."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

from alma import config


class TurrellRunner:
    def __init__(self, log_lines: int = 50) -> None:
        self._proc: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()
        self._log: Deque[str] = deque(maxlen=log_lines)
        self._last_error: Optional[str] = None
        self._stopping = False

    def start(self, display: int = 0, fullscreen: bool = False, ndjson_path: Optional[str] = None) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return
            self._last_error = None
            self._stopping = False

        ndjson = ndjson_path or str(config.STATE_STREAM_PATH)
        mode_file = str(config.MODE_FILE_PATH)
        script_path = Path(__file__).resolve().parents[2] / "external" / "xq_turrell_room_2d_v5_3_style_modes.py"

        cmd_parts = [
            "python",
            shlex.quote(str(script_path)),
            "--ndjson",
            shlex.quote(ndjson),
            "--mode-file",
            shlex.quote(mode_file),
            "--no-freeze-on-invalid",
            "--hud",
            "--quality",
            "4k",
            "--display",
            str(int(display)),
        ]
        if fullscreen:
            cmd_parts.append("--fullscreen")
        cmd = " ".join(cmd_parts)

        try:
            proc = subprocess.Popen(
                ["bash", "-lc", cmd],
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:  # pragma: no cover
            with self._lock:
                self._proc = None
                self._last_error = f"Failed to start Turrell: {exc}"
            return

        with self._lock:
            self._proc = proc

        for stream, prefix in ((proc.stdout, "STDOUT"), (proc.stderr, "STDERR")):
            if stream:
                threading.Thread(
                    target=self._consume_stream, args=(stream, prefix), daemon=True
                ).start()
        threading.Thread(target=self._monitor_process, args=(proc,), daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            if not self._proc or self._proc.poll() is not None:
                self._proc = None
                return
            self._stopping = True
            try:
                os.killpg(self._proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self._proc = None

    def status(self) -> Dict[str, Optional[object]]:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            pid = self._proc.pid if running and self._proc else None
            return {
                "running": running,
                "pid": pid,
                "last_error": self._last_error,
                "log_tail": list(self._log),
            }

    def _consume_stream(self, stream, prefix: str) -> None:
        for line in stream:
            with self._lock:
                self._log.append(f"[{prefix}] {line.rstrip()}")
        stream.close()

    def _monitor_process(self, proc: subprocess.Popen[str]) -> None:
        return_code = proc.wait()
        with self._lock:
            if self._stopping:
                self._stopping = False
                return
            if return_code not in (0, None):
                self._last_error = f"Turrell exited with code {return_code}"


__all__ = ["TurrellRunner"]

