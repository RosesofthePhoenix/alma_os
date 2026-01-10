import threading
import time
from typing import Optional

try:
    import Quartz
    import AppKit
except Exception:  # pragma: no cover
    Quartz = None
    AppKit = None

from alma.engine import storage


_logger_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_running = False


def _is_fullscreen(win_info) -> bool:
    try:
        return bool(win_info.get("kCGWindowLayer") == 0 and win_info.get("kCGWindowBounds", {}).get("Height", 0) >= 900)
    except Exception:
        return False


def _poll_loop(interval: float = 10.0) -> None:
    global _running
    while not _stop_event.is_set():
        ts = time.time()
        idle = False
        try:
            if Quartz:
                idle_secs = Quartz.CGEventSourceSecondsSinceLastEventType(
                    Quartz.kCGEventSourceStateCombinedSessionState, Quartz.kCGAnyInputEventType
                )
                idle = idle_secs >= interval
                window_list = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
                if window_list:
                    for win in window_list:
                        if _is_fullscreen(win):
                            continue
                        app_name = win.get("kCGWindowOwnerName", "")
                        title = win.get("kCGWindowName", "")
                        storage.insert_ambient_log(ts=ts, app_name=app_name, window_title=title, duration=interval, idle_flag=idle)
        except Exception:
            pass
        time.sleep(interval)
    _running = False


def start(interval: float = 10.0) -> None:
    global _logger_thread, _running
    if _running:
        return
    if Quartz is None:
        _running = False
        return
    _stop_event.clear()
    _logger_thread = threading.Thread(target=_poll_loop, args=(interval,), daemon=True)
    _logger_thread.start()
    _running = True


def stop() -> None:
    global _running
    _stop_event.set()
    _running = False


def status() -> bool:
    return _running
