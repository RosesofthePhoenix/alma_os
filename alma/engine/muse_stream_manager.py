"""Manage muselsl streaming in a supervised subprocess loop."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
from collections import deque
from typing import Deque, Dict, List, Optional


class MuseStreamManager:
    """Start/stop muselsl stream loop and track lightweight logs."""

    def __init__(self, log_lines: int = 50) -> None:
        self._proc: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()
        self._log: Deque[str] = deque(maxlen=log_lines)
        self._last_error: Optional[str] = None
        self._stopping = False

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
                self._last_error = f"muselsl exited with code {return_code}"

    def start(self, address: str) -> None:
        loop = (
            "while true; do\n"
            f"  muselsl stream --address {shlex.quote(address)}\n"
            '  echo "[muselsl exited] restarting in 2s..."\n'
            "  sleep 2\n"
            "done"
        )
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return
            self._last_error = None
            self._stopping = False
            try:
                proc = subprocess.Popen(
                    ["bash", "-lc", loop],
                    preexec_fn=os.setsid,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._proc = None
                self._last_error = f"Failed to start muselsl: {exc}"
                return

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
            log_tail: List[str] = list(self._log)
            return {
                "running": running,
                "pid": pid,
                "last_error": self._last_error,
                "log_tail": log_tail,
            }


__all__ = ["MuseStreamManager"]

