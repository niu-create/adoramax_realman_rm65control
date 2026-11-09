import threading
import time
from typing import Dict, Mapping, Optional


class GripperFeedbackSampler:
    """Poll gripper feedback independently of dataset and arm rates."""

    def __init__(
        self,
        grippers: Mapping[str, object],
        rate_hz: float = 10.0,
        error_backoff_s: float = 1.0,
    ):
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        if error_backoff_s < 0.0:
            raise ValueError("error_backoff_s may not be negative")
        self.grippers = dict(grippers)
        self.rate_hz = float(rate_hz)
        self.error_backoff_s = float(error_backoff_s)
        self._errors: Dict[str, tuple] = {}
        self._retry_after: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def sample_once(self, now=None) -> None:
        now = time.monotonic() if now is None else float(now)
        for name, gripper in self.grippers.items():
            with self._lock:
                retry_after = self._retry_after.get(name, float("-inf"))
            if now < retry_after:
                continue
            try:
                gripper.get_motion_status()
            except Exception as exc:
                with self._lock:
                    self._errors[name] = (str(exc), now)
                    self._retry_after[name] = now + self.error_backoff_s
            else:
                with self._lock:
                    self._errors.pop(name, None)
                    self._retry_after.pop(name, None)

    def snapshots(self) -> Dict[str, dict]:
        result = {}
        with self._lock:
            errors = dict(self._errors)
        for name, gripper in self.grippers.items():
            snapshot = dict(gripper.get_collection_snapshot())
            native_open = float(getattr(gripper, "POSITION_OPEN", 100))
            if native_open <= 0.0:
                raise ValueError(f"{name} POSITION_OPEN must be positive")
            native_position = snapshot.get("position")
            native_target = snapshot.get("target_position")
            snapshot["native_position"] = native_position
            snapshot["native_target_position"] = native_target
            snapshot["position"] = (
                None
                if native_position is None
                else float(native_position) * 100.0 / native_open
            )
            snapshot["target_position"] = (
                None
                if native_target is None
                else float(native_target) * 100.0 / native_open
            )
            error = errors.get(name)
            snapshot["sampler_error"] = None if error is None else error[0]
            snapshot["sampler_error_time"] = None if error is None else error[1]
            result[name] = snapshot
        return result

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        while not self._stop.is_set():
            started = time.monotonic()
            self.sample_once(now=started)
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="gripper-feedback",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
