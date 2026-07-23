"""Single-keystroke operator panel for the actor terminal.

No Enter needed — designed to be used with one hand while the other holds
the master arm:

    space  pause / resume   (servo holds position instantly)
    t      takeover on/off  (master arm drives; policy ignored while on —
                             holding the master still holds the arm still)
    s      end episode as SUCCESS (reward 1)
    f      end episode as FAILURE (reward 0)
    ?      print this help

Reads the controlling terminal (/dev/tty) in cbreak mode from a daemon
thread. If there is no tty (e.g. running under a supervisor), the panel
disables itself and PiperEnv falls back to the line-based
success_detector_manual().
"""

from __future__ import annotations

import logging
import select
import threading

logger = logging.getLogger(__name__)

HELP = "[panel] space=pause/resume  t=takeover on/off  s=success  f=failure  ?=help"


class OperatorPanel:
    def __init__(self, key_source=None):
        """key_source: iterator of chars (tests); None = read /dev/tty."""
        self._lock = threading.Lock()
        self._paused = False
        self._takeover = False
        self._label: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._key_source = key_source
        self._tty = None

    # ---------------- lifecycle ----------------

    def start(self) -> bool:
        """Returns False (and stays inert) when no tty is available."""
        if self._key_source is None:
            try:
                import termios  # noqa: F401  (fail early on non-unix)

                self._tty = open("/dev/tty", "rb", buffering=0)
            except Exception as e:
                logger.warning("operator panel disabled (no tty: %s)", e)
                return False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="operator-panel")
        self._thread.start()
        print(HELP, flush=True)
        return True

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._tty is not None:
            try:
                self._tty.close()
            except Exception:
                pass
            self._tty = None

    # ---------------- state consumed by the env ----------------

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def takeover(self) -> bool:
        with self._lock:
            return self._takeover

    def consume_label(self) -> str | None:
        """Return and clear 'success' / 'reset', or None."""
        with self._lock:
            label, self._label = self._label, None
            return label

    def reset_episode(self):
        """Clear pause/takeover + pending label at episode boundaries."""
        with self._lock:
            self._paused = False
            self._takeover = False
            self._label = None

    # ---------------- key handling ----------------

    def handle_key(self, ch: str):
        if ch == " ":
            with self._lock:
                self._paused = not self._paused
                paused = self._paused
            print(f"[panel] {'PAUSED — arm holding position' if paused else 'resumed'}", flush=True)
        elif ch in ("t", "T"):
            with self._lock:
                self._takeover = not self._takeover
                takeover = self._takeover
            print(f"[panel] {'TAKEOVER — master arm in control' if takeover else 'takeover off — policy resumes'}",
                  flush=True)
        elif ch in ("s", "S"):
            with self._lock:
                self._label = "success"
                self._paused = False
            print("[panel] episode -> SUCCESS", flush=True)
        elif ch in ("f", "F"):
            with self._lock:
                self._label = "reset"
                self._paused = False
            print("[panel] episode -> FAILURE (reset)", flush=True)
        elif ch == "?":
            print(HELP, flush=True)

    def _loop(self):
        if self._key_source is not None:
            for ch in self._key_source:
                if self._stop.is_set():
                    return
                self.handle_key(ch)
            return

        import termios
        import tty as tty_mod

        fd = self._tty.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty_mod.setcbreak(fd)
            while not self._stop.is_set():
                ready, _, _ = select.select([fd], [], [], 0.2)
                if not ready:
                    continue
                data = self._tty.read(1)
                if data:
                    self.handle_key(data.decode(errors="ignore"))
        except Exception:
            logger.exception("operator panel reader died")
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass
