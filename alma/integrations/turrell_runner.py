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

    def start(
        self,
        display: int = 0,
        fullscreen: bool = False,
        ndjson_path: Optional[str] = None,
        q_metric: str = config.CANONICAL_Q,
    ) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return
            self._last_error = None
            self._stopping = False

        ndjson_input = ndjson_path if ndjson_path else str(config.STATE_STREAM_PATH)
        ndjson_resolved = Path(ndjson_input).expanduser()
        if not ndjson_resolved.is_absolute():
            ndjson_resolved = (config.ROOT_DIR / ndjson_resolved).resolve()
        mode_file = Path(config.MODE_FILE_PATH).expanduser()
        script_path = Path(__file__).resolve().parents[2] / "external" / "xq_turrell_room_2d_v6_hce.py"

        cmd_parts = [
            "python",
            shlex.quote(str(script_path)),
            "--ndjson",
            shlex.quote(str(ndjson_resolved)),
            "--mode-file",
            shlex.quote(str(mode_file)),
            "--no-freeze-on-invalid",
            "--hud",
            "--quality",
            "4k",
            "--display",
            str(int(display)),
            "--q-metric",
            shlex.quote(q_metric or config.CANONICAL_Q),
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
        proc: Optional[subprocess.Popen[str]] = None
        with self._lock:
            if not self._proc or self._proc.poll() is not None:
                self._proc = None
                return
            self._stopping = True
            proc = self._proc
        if proc:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=2.0)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
        with self._lock:
            self._proc = None
            self._stopping = False

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

