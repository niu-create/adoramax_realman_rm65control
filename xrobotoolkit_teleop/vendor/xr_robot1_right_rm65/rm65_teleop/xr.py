"""Atomic single/dual-controller XR input with per-hand freshness tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import numpy as np

from .models import XrSample
from .xr_atomic import AtomicXrClient, AtomicXrError


class XrError(RuntimeError):
    pass


class XrSdkClient:
    def __init__(
        self,
        controller_name: str,
        grip_name: str,
        trigger_name: str,
        *,
        atomic_client=None,
    ):
        input_names = (controller_name, grip_name, trigger_name)
        supported_inputs = {
            ("right_controller", "right_grip", "right_trigger"): "right",
            ("left_controller", "left_grip", "left_trigger"): "left",
        }
        if input_names not in supported_inputs:
            raise ValueError(
                "XR input names must be one complete matching left or right "
                "controller/grip/trigger set"
            )
        self._hand = supported_inputs[input_names]
        self._client = AtomicXrClient() if atomic_client is None else atomic_client
        self._client.open()
        self._sequence = 0
        self._closed = False

    def read_sample(self, *, timeout_s: float = 0.25) -> XrSample:
        """Convert one complete vendor callback packet into a live sample."""
        try:
            packet = self._client.wait_next(
                self._sequence,
                timeout_s=float(timeout_s),
            )
        except AtomicXrError as exc:
            raise XrError(str(exc)) from exc
        self._sequence = int(packet.sequence)
        return _sample_from_atomic_packet(packet, self._hand)

    def diagnostics(self) -> dict:
        return self._client.diagnostics()

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True


def _sample_from_atomic_packet(packet, hand: str) -> XrSample:
    if hand == "right":
        pose = packet.pose_xyz_xyzw
        grip = packet.grip
        trigger = packet.trigger
        axis_xy = packet.axis_xy
        axis_click = packet.axis_click
        primary_button = packet.primary_button
        secondary_button = packet.secondary_button
        menu_button = packet.menu_button
    elif hand == "left":
        required = {
            "pose": packet.left_pose_xyz_xyzw,
            "grip": packet.left_grip,
            "trigger": packet.left_trigger,
            "axis_xy": packet.left_axis_xy,
            "axis_click": packet.left_axis_click,
            "primary_button": packet.left_primary_button,
            "secondary_button": packet.left_secondary_button,
            "menu_button": packet.left_menu_button,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise XrError(
                "atomic XR packet is missing left-controller fields: "
                + ", ".join(missing)
            )
        pose = required["pose"]
        grip = required["grip"]
        trigger = required["trigger"]
        axis_xy = required["axis_xy"]
        axis_click = required["axis_click"]
        primary_button = required["primary_button"]
        secondary_button = required["secondary_button"]
        menu_button = required["menu_button"]
    else:
        raise ValueError("hand must be 'left' or 'right'")
    return XrSample(
        np.asarray(pose, dtype=float),
        float(grip),
        float(trigger),
        int(packet.timestamp_ns),
        float(packet.received_monotonic_ns) / 1.0e9,
        axis_xy=tuple(float(value) for value in axis_xy),
        axis_click=bool(axis_click),
        primary_button=bool(primary_button),
        secondary_button=bool(secondary_button),
        menu_button=bool(menu_button),
        head_pose_xyz_xyzw=(
            None
            if packet.head_pose_xyz_xyzw is None
            else np.asarray(packet.head_pose_xyz_xyzw, dtype=float)
        ),
    )


class DualXrSdkClient:
    """Read left and right controller states from one atomic SDK callback."""

    def __init__(self, *, atomic_client=None):
        self._client = AtomicXrClient() if atomic_client is None else atomic_client
        self._client.open()
        self._sequence = 0
        self._closed = False
        self.last_hand_errors: dict[str, XrError] = {}

    def read_samples(self, *, timeout_s: float = 0.25) -> dict[str, XrSample]:
        try:
            packet = self._client.wait_next(
                self._sequence,
                timeout_s=float(timeout_s),
            )
        except AtomicXrError as exc:
            raise XrError(str(exc)) from exc
        self._sequence = int(packet.sequence)
        samples = {}
        errors = {}
        for hand in ("right", "left"):
            try:
                samples[hand] = _sample_from_atomic_packet(packet, hand)
            except XrError as exc:
                errors[hand] = exc
        self.last_hand_errors = errors
        if not samples:
            details = "; ".join(f"{hand}: {error}" for hand, error in errors.items())
            raise XrError(f"atomic XR packet contains no usable controller: {details}")
        return samples

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True


class FreshSampleBuffer:
    """Latest XR sample plus a host-side source-timestamp watchdog."""

    def __init__(self, timeout_s: float, *, required_advances: int = 2):
        self.timeout_s = float(timeout_s)
        self.required_advances = int(required_advances)
        self._lock = threading.Lock()
        self._latest: XrSample | None = None
        self._last_timestamp_ns: int | None = None
        self._last_advance_monotonic: float | None = None
        self._advance_count = 0
        self._ever_fresh = False
        self._last_error: str | None = None
        self._last_update_monotonic: float | None = None
        self._update_count = 0
        self._timestamp_advance_count = 0
        self._duplicate_timestamp_count = 0
        self._read_error_count = 0

    def update(self, sample: XrSample) -> None:
        with self._lock:
            self._update_count += 1
            self._last_update_monotonic = time.monotonic()
            timestamp = int(sample.timestamp_ns)
            if self._last_timestamp_ns is None:
                self._advance_count = 1
                self._timestamp_advance_count += 1
                self._last_advance_monotonic = sample.received_monotonic
            elif timestamp > self._last_timestamp_ns:
                self._advance_count += 1
                self._timestamp_advance_count += 1
                self._last_advance_monotonic = sample.received_monotonic
            elif timestamp < self._last_timestamp_ns:
                # SDK/device restart: establish a new epoch and require another
                # advancing packet before control is considered live.
                self._advance_count = 1
                self._timestamp_advance_count += 1
                self._last_advance_monotonic = sample.received_monotonic
            else:
                self._duplicate_timestamp_count += 1
            self._last_timestamp_ns = timestamp
            self._latest = sample
            self._last_error = None
            if self._is_fresh_locked(sample.received_monotonic):
                self._ever_fresh = True

    def record_error(self, error: BaseException) -> None:
        with self._lock:
            self._read_error_count += 1
            self._last_error = f"{type(error).__name__}: {error}"

    def diagnostics(self, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            last_callback = None if self._latest is None else self._latest.received_monotonic
            return {
                "source_timestamp_ns": self._last_timestamp_ns,
                "received_count": self._update_count,
                "timestamp_advance_count": self._timestamp_advance_count,
                "duplicate_timestamp_count": self._duplicate_timestamp_count,
                "read_error_count": self._read_error_count,
                "last_error": self._last_error,
                "callback_age_s": None if last_callback is None else max(0.0, now - last_callback),
                "buffer_update_age_s": (
                    None if self._last_update_monotonic is None
                    else max(0.0, now - self._last_update_monotonic)
                ),
            }

    def _age_locked(self, now: float) -> float:
        if self._last_advance_monotonic is None:
            return float("inf")
        return max(0.0, now - self._last_advance_monotonic)

    def _is_fresh_locked(self, now: float) -> bool:
        return (
            self._advance_count >= self.required_advances
            and self._age_locked(now) <= self.timeout_s
        )

    def latest(self, now: float | None = None) -> XrSample | None:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            if self._latest is None:
                return None
            fresh = self._is_fresh_locked(now)
            if fresh:
                self._ever_fresh = True
            return replace(self._latest, stream_fresh=fresh)

    def status(self, now: float | None = None) -> tuple[bool, bool, float, str | None]:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            fresh = self._is_fresh_locked(now)
            if fresh:
                self._ever_fresh = True
            return fresh, self._ever_fresh, self._age_locked(now), self._last_error


class XrReaderThread:
    def __init__(self, client: XrSdkClient, buffer: FreshSampleBuffer):
        self.client = client
        self.buffer = buffer
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="rm65-clean-xr-reader",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def diagnostics(self) -> dict:
        return self.client.diagnostics()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.buffer.update(self.client.read_sample())
            except Exception as exc:
                self.buffer.record_error(exc)
                self.stop_event.wait(0.005)
            else:
                self.stop_event.wait(0.001)

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        if self.thread.is_alive():
            raise XrError("XR reader thread did not stop within one second")
        self.client.close()


class DualXrReaderThread:
    """Publish both hands from each atomic packet into independent buffers."""

    def __init__(self, client: DualXrSdkClient, buffers: dict[str, FreshSampleBuffer]):
        if set(buffers) != {"right", "left"}:
            raise ValueError("dual XR buffers must contain exactly left and right")
        self.client = client
        self.buffers = dict(buffers)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="rm65-clean-dual-xr-reader",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                samples = self.client.read_samples()
                for hand, sample in samples.items():
                    self.buffers[hand].update(sample)
                for hand, error in self.client.last_hand_errors.items():
                    self.buffers[hand].record_error(error)
            except Exception as exc:
                for buffer in self.buffers.values():
                    buffer.record_error(exc)
                self.stop_event.wait(0.005)
            else:
                self.stop_event.wait(0.001)

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        if self.thread.is_alive():
            raise XrError("dual XR reader thread did not stop within one second")
        self.client.close()
