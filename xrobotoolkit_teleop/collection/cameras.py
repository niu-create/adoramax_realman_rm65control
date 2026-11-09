import threading
import time
from typing import Optional

import numpy as np

from .models import CameraSample


class CameraReadError(RuntimeError):
    pass


class OpenCVCameraReader:
    """Keep only the newest RGB frame from one OpenCV/UVC camera."""

    def __init__(
        self,
        role: str,
        device,
        *,
        width: int,
        height: int,
        fps: int,
        cv_module=None,
        clock=time.monotonic,
        reconnect_failure_threshold: int = 3,
        reconnect_backoff_s: float = 0.25,
    ):
        if role not in ("top", "left_wrist", "right_wrist"):
            raise ValueError(f"unsupported camera role: {role}")
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("camera width, height and fps must be positive")
        if cv_module is None:
            import cv2 as cv_module  # type: ignore
        self.role = role
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._clock = clock
        self._cv = cv_module
        self._requested_fourcc = cv_module.VideoWriter_fourcc(*"MJPG")
        self._bayer_fourcc = cv_module.VideoWriter_fourcc(*"BA81")
        self._input_format = "MJPG"
        self._reconnect_failure_threshold = max(
            1, int(reconnect_failure_threshold)
        )
        self._reconnect_backoff_s = max(0.0, float(reconnect_backoff_s))
        self._capture = self._open_capture()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._sample: Optional[CameraSample] = None
        self._error: Optional[BaseException] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _open_capture(self):
        cv_module = self._cv
        capture = cv_module.VideoCapture(self.device, cv_module.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise CameraReadError(
                f"{self.role} camera could not open {self.device}"
            )
        if capture.set(cv_module.CAP_PROP_FOURCC, self._requested_fourcc):
            self._input_format = "MJPG"
        else:
            # Gemini 335 connected through a USB 2.0 path may expose its RGB
            # sensor only as raw Bayer (BA81), while the other cameras retain
            # MJPG.  Accept that native stream and demosaic it in read_once.
            capture.set(cv_module.CAP_PROP_FOURCC, self._bayer_fourcc)
            self._input_format = "BA81"
        settings = (
            (cv_module.CAP_PROP_FRAME_WIDTH, self.width, "width"),
            (cv_module.CAP_PROP_FRAME_HEIGHT, self.height, "height"),
            (cv_module.CAP_PROP_FPS, self.fps, "fps"),
            (cv_module.CAP_PROP_BUFFERSIZE, 1, "buffer size"),
        )
        for prop, value, label in settings:
            if not capture.set(prop, value):
                capture.release()
                raise CameraReadError(
                    f"{self.role} camera rejected {label}={value}"
                )
        actual = {
            "fourcc": int(capture.get(cv_module.CAP_PROP_FOURCC)),
            "width": int(capture.get(cv_module.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv_module.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv_module.CAP_PROP_FPS)),
        }
        if (
            actual["fourcc"] not in (self._requested_fourcc, self._bayer_fourcc)
            or actual["width"] != self.width
            or actual["height"] != self.height
            or abs(actual["fps"] - self.fps) > 0.5
        ):
            capture.release()
            raise CameraReadError(
                f"{self.role} camera format mismatch: requested=MJPG-or-BA81/"
                f"{self.width}x{self.height}@{self.fps}, actual={actual}"
            )
        return capture

    def _reopen_capture(self) -> None:
        self._capture.release()
        self._capture = self._open_capture()

    def read_once(self) -> CameraSample:
        ok, frame = self._capture.read()
        if not ok or frame is None or np.asarray(frame).size == 0:
            raise CameraReadError(f"{self.role} camera returned an empty frame")
        frame = np.asarray(frame)
        if frame.shape == (self.height, self.width):
            if self._input_format != "BA81":
                raise CameraReadError(
                    f"{self.role} camera returned unexpected single-channel frame"
                )
            frame = self._cv.cvtColor(frame, self._cv.COLOR_BayerBG2BGR)
        if frame.shape != (self.height, self.width, 3):
            raise CameraReadError(
                f"{self.role} camera resolution changed: "
                f"got={frame.shape}, expected={(self.height, self.width, 3)}"
            )
        if frame.dtype != np.uint8:
            raise CameraReadError(f"{self.role} camera frame is not uint8")
        sample = CameraSample(
            role=self.role,
            image=frame[..., ::-1].copy(),
            monotonic_time=float(self._clock()),
        )
        with self._condition:
            self._sample = sample
            self._error = None
            self._condition.notify_all()
        return sample

    def _run(self) -> None:
        period = 1.0 / self.fps
        consecutive_failures = 0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.read_once()
                consecutive_failures = 0
            except BaseException as exc:
                with self._lock:
                    self._error = exc
                consecutive_failures += 1
                if consecutive_failures >= self._reconnect_failure_threshold:
                    try:
                        self._stop.wait(self._reconnect_backoff_s)
                        if self._stop.is_set():
                            break
                        self._reopen_capture()
                        consecutive_failures = 0
                    except BaseException as reopen_exc:
                        with self._lock:
                            self._error = reopen_exc
                        self._stop.wait(self._reconnect_backoff_s)
                continue
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-{self.role}",
            daemon=True,
        )
        self._thread.start()

    def snapshot(self) -> CameraSample:
        with self._lock:
            if self._sample is None:
                if self._error is not None:
                    raise CameraReadError(str(self._error)) from self._error
                raise CameraReadError(f"{self.role} camera has no frame")
            return CameraSample(
                role=self._sample.role,
                image=self._sample.image.copy(),
                monotonic_time=self._sample.monotonic_time,
            )

    def snapshot_after(
        self,
        monotonic_time: float,
        *,
        timeout_s: float,
    ) -> CameraSample:
        """Return a frame newer than ``monotonic_time`` or fail.

        Dataset sampling and UVC capture both run near 30 Hz.  A plain
        ``snapshot()`` can therefore return the same camera frame on two
        adjacent dataset ticks.  Waiting for the camera timestamp to advance
        prevents that old frame from being encoded twice.
        """
        timeout_s = float(timeout_s)
        if timeout_s < 0.0:
            raise ValueError("timeout_s may not be negative")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while (
                self._sample is None
                or self._sample.monotonic_time <= float(monotonic_time)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    detail = (
                        f"; last_error={self._error}"
                        if self._error is not None
                        else ""
                    )
                    raise CameraReadError(
                        f"{self.role} camera did not produce a fresh frame "
                        f"within {timeout_s * 1000.0:.0f} ms{detail}"
                    )
                self._condition.wait(remaining)
            return CameraSample(
                role=self._sample.role,
                image=self._sample.image.copy(),
                monotonic_time=self._sample.monotonic_time,
            )

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def health_error(self, *, max_age_s: float = 0.5) -> Optional[str]:
        with self._lock:
            if self._sample is None:
                return f"{self.role} camera has no frame"
            age_s = float(self._clock()) - self._sample.monotonic_time
            if age_s > max_age_s:
                detail = f"; last_error={self._error}" if self._error else ""
                return (
                    f"{self.role} camera stale: {age_s * 1000.0:.0f} ms"
                    f"{detail}"
                )
        return None

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._capture.release()
