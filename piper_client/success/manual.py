"""Manual reward labeling from the terminal (non-blocking).

success button -> done=True, success=True, reward=1
reset button   -> done=True, success=False, reward=0
"""

from __future__ import annotations

import select
import sys


def success_detector_manual() -> str:
    """Non-blocking manual episode override via stdin (when input is pending).

    Returns one of:
    - "keep_going": no input queued, or unrecognized input
    - "success": user entered 1
    - "reset": user entered 2 (end episode without success)

    Uses /dev/tty when available so it works even if stdin is redirected.
    """

    def _readline(prompt: str) -> str:
        try:
            sys.stdout.write(prompt + "\n")
            sys.stdout.flush()
        except Exception:
            pass
        try:
            with open("/dev/tty", "r") as tty:
                return tty.readline()
        except Exception:
            return sys.stdin.readline()

    try:
        pending, _, _ = select.select([sys.stdin], [], [], 0)
    except Exception:
        return "keep_going"
    if not pending:
        return "keep_going"

    # Consume the queued line, then confirm intent.
    first = sys.stdin.readline().strip()
    choice = first if first in ("1", "2") else _readline(
        "[manual] 1=success  2=reset (end episode, no success)  3=keep going"
    ).strip()
    if choice == "1":
        return "success"
    if choice == "2":
        return "reset"
    return "keep_going"
