import os
import select
import sys
import termios
import time
import tty


class TerminalKeyboard:
    """Read Enter, arrows and Escape from a Linux/macOS terminal."""

    _SEQUENCES = {
        b"\r": "ENTER",
        b"\n": "ENTER",
        b"\x1b[C": "RIGHT",
        b"\x1b[D": "LEFT",
        b"\x1bOC": "RIGHT",
        b"\x1bOD": "LEFT",
        b"s": "RIGHT",
        b"S": "RIGHT",
        b"\x1b": "ESC",
    }

    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        if not self.stream.isatty():
            raise RuntimeError("episode keyboard requires an interactive terminal")
        self.fd = self.stream.fileno()
        self._saved = None
        self._pending = b""
        self._escape_started = None

    def _parse_pending(self):
        while self._pending:
            if not self._pending.startswith(b"\x1b"):
                first, self._pending = self._pending[:1], self._pending[1:]
                action = self._SEQUENCES.get(first)
                if action:
                    return action
                continue
            if self._escape_started is None:
                self._escape_started = time.monotonic()
            if len(self._pending) >= 2 and self._pending[1:2] not in (b"[", b"O"):
                self._pending = self._pending[1:]
                self._escape_started = None
                return "ESC"
            for index in range(2, len(self._pending)):
                if 0x40 <= self._pending[index] <= 0x7e:
                    sequence, self._pending = self._pending[:index + 1], self._pending[index + 1:]
                    self._escape_started = None
                    action = self._SEQUENCES.get(sequence)
                    if action:
                        return action
                    break
            else:
                if time.monotonic() - self._escape_started < 0.15:
                    return None
                standalone_escape = self._pending == b"\x1b"
                self._pending = b""
                self._escape_started = None
                # A truncated arrow sequence is never treated as an exit.
                return "ESC" if standalone_escape else None
        return None

    def __enter__(self):
        self._saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def poll(self, timeout_s: float):
        action = self._parse_pending()
        if action:
            return action
        timeout_s = max(0.0, float(timeout_s))
        if self._escape_started is not None:
            timeout_s = min(timeout_s, max(0.0, 0.15 - (time.monotonic() - self._escape_started)))
        readable, _, _ = select.select([self.fd], [], [], timeout_s)
        if not readable:
            return self._parse_pending()
        self._pending += os.read(self.fd, 64)
        return self._parse_pending()

    def __exit__(self, *_args):
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            self._saved = None
