"""Out-of-process supervision for the physical dual-RM65 collector."""

from __future__ import annotations

import fcntl
import os
import selectors
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path


SUPERVISOR_FD_ENV = "XR_RM65_SUPERVISOR_FD"


class SupervisorNotifier:
    """Send small liveness events to the parent without SDK interaction."""

    def __init__(self, fd=None, *, heartbeat_interval_s: float = 1.0):
        self.fd = None if fd is None else int(fd)
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self._last_heartbeat = None

    @classmethod
    def from_environment(cls):
        value = os.environ.get(SUPERVISOR_FD_ENV)
        return cls(None if value is None else int(value))

    def _send(self, event: str) -> None:
        if self.fd is None:
            return
        try:
            os.write(self.fd, f"{event}\n".encode("ascii"))
        except OSError:
            self.fd = None

    def ready(self) -> None:
        self._send("READY")

    @contextmanager
    def saving(self):
        """Allow a bounded disk/video save without relaxing normal heartbeats."""
        self._send("SAVE_BEGIN")
        try:
            yield
        finally:
            self._send("SAVE_END")

    def heartbeat(self, now=None) -> None:
        now = time.monotonic() if now is None else float(now)
        if (
            self._last_heartbeat is None
            or now - self._last_heartbeat >= self.heartbeat_interval_s
        ):
            self._send("HEARTBEAT")
            self._last_heartbeat = now


@contextmanager
def _exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another dual-RM65 collection is already active ({path})"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _stop_worker(process, *, interrupt_signal, grace_s: float) -> None:
    if process.poll() is not None:
        return
    process.send_signal(interrupt_signal)
    try:
        process.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    process.terminate()
    try:
        process.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace_s)


def run_supervised(
    command,
    *,
    startup_timeout_s: float = 120.0,
    heartbeat_timeout_s: float = 15.0,
    save_timeout_s: float = 300.0,
    terminate_grace_s: float = 3.0,
    lock_path="/tmp/xr-rm65-collection.lock",
) -> int:
    """Run the collector while retaining control outside native SDK calls."""
    if not command:
        raise ValueError("worker command may not be empty")
    read_fd, write_fd = os.pipe()
    env = os.environ.copy()
    env[SUPERVISOR_FD_ENV] = str(write_fd)
    lock = nullcontext() if lock_path is None else _exclusive_lock(lock_path)
    with lock:
        process = subprocess.Popen(
            list(command),
            env=env,
            pass_fds=(write_fd,),
        )
        os.close(write_fd)
        selector = selectors.DefaultSelector()
        selector.register(read_fd, selectors.EVENT_READ)
        started = time.monotonic()
        ready = False
        last_heartbeat = None
        save_started = None
        requested_signal = None
        buffer = b""
        previous_handlers = {}

        def request_stop(signum, _frame):
            nonlocal requested_signal
            requested_signal = signum

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, request_stop)
            while True:
                status = process.poll()
                if status is not None:
                    return int(status)
                if requested_signal is not None:
                    print(
                        "Collection supervisor stopping worker...",
                        file=sys.stderr,
                        flush=True,
                    )
                    _stop_worker(
                        process,
                        interrupt_signal=requested_signal,
                        grace_s=terminate_grace_s,
                    )
                    return 128 + int(requested_signal)

                now = time.monotonic()
                if not ready and now - started > startup_timeout_s:
                    print(
                        "Collection startup timed out; terminating a worker "
                        "blocked in hardware/SDK initialization.",
                        file=sys.stderr,
                        flush=True,
                    )
                    _stop_worker(
                        process,
                        interrupt_signal=signal.SIGINT,
                        grace_s=terminate_grace_s,
                    )
                    return 124
                if (
                    ready
                    and last_heartbeat is not None
                    and (
                        (save_started is None and now - last_heartbeat > heartbeat_timeout_s)
                        or (save_started is not None and now - save_started > save_timeout_s)
                    )
                ):
                    print(
                        ("Collection save timed out; terminating the "
                         if save_started is not None else
                         "Collection heartbeat timed out; terminating the ") +
                        "unresponsive worker.",
                        file=sys.stderr,
                        flush=True,
                    )
                    _stop_worker(
                        process,
                        interrupt_signal=signal.SIGINT,
                        grace_s=terminate_grace_s,
                    )
                    return 125

                for key, _ in selector.select(timeout=0.1):
                    chunk = os.read(key.fd, 4096)
                    if not chunk:
                        selector.unregister(key.fd)
                        continue
                    buffer += chunk
                    while b"\n" in buffer:
                        raw, buffer = buffer.split(b"\n", 1)
                        event = raw.decode("ascii", "replace")
                        event_time = time.monotonic()
                        if event == "READY":
                            ready = True
                            last_heartbeat = event_time
                        elif event == "HEARTBEAT":
                            last_heartbeat = event_time
                        elif event == "SAVE_BEGIN" and ready and save_started is None:
                            save_started = event_time
                        elif event == "SAVE_END":
                            save_started = None
                            last_heartbeat = event_time
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            selector.close()
            os.close(read_fd)
            if process.poll() is None:
                _stop_worker(
                    process,
                    interrupt_signal=signal.SIGINT,
                    grace_s=terminate_grace_s,
                )
