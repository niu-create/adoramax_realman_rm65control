"""Atomic, read-only access to XRoboToolkit device-state JSON packets.

The vendor Python binding exposes one getter per field.  Those getters use
different mutexes, so combining them cannot prove that pose, buttons, and the
timestamp came from one packet.  This module consumes the vendor C callback
directly and publishes one immutable object per ``PXREADeviceStateJson``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


PXREA_SERVER_CONNECT = 1 << 2
PXREA_SERVER_DISCONNECT = 1 << 3
PXREA_DEVICE_FIND = 1 << 4
PXREA_DEVICE_MISSING = 1 << 5
PXREA_DEVICE_STATE_JSON = 1 << 25
PXREA_FULL_MASK = 0xFFFFFFFF


class AtomicXrError(RuntimeError):
    """XR callback, protocol, or stream failure."""


class _DeviceStateJson(ctypes.Structure):
    _fields_ = [
        ("devID", ctypes.c_char * 32),
        ("stateJson", ctypes.c_char * 16352),
    ]


_CallbackType = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
)


def _finite_tuple(value: Any, size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        raise AtomicXrError(f"{name} must be a comma-separated string or array")
    if len(parts) != size:
        raise AtomicXrError(f"{name} must contain exactly {size} values")
    try:
        result = tuple(float(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise AtomicXrError(f"{name} contains a non-numeric value") from exc
    if not all(math.isfinite(number) for number in result):
        raise AtomicXrError(f"{name} contains a non-finite value")
    return result


def _finite_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AtomicXrError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise AtomicXrError(f"{name} must be finite")
    return result


def _required_bool(
    mapping: dict[str, Any],
    key: str,
    hand: str = "right",
) -> bool:
    if key not in mapping:
        raise AtomicXrError(f"{hand} controller is missing {key}")
    value = mapping[key]
    if not isinstance(value, bool):
        raise AtomicXrError(f"{hand} controller {key} must be boolean")
    return value


def _parse_controller(mapping: Any, hand: str) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise AtomicXrError(f"{hand} controller must be an object")
    pose = _finite_tuple(mapping.get("pose"), 7, f"{hand} controller pose")
    quaternion_norm = math.sqrt(sum(component * component for component in pose[3:7]))
    if quaternion_norm < 1.0e-9:
        raise AtomicXrError(f"{hand} controller quaternion has zero norm")
    return {
        "pose_xyz_xyzw": pose,
        "grip": _finite_scalar(mapping.get("grip"), f"{hand} grip"),
        "trigger": _finite_scalar(mapping.get("trigger"), f"{hand} trigger"),
        "axis_xy": (
            _finite_scalar(mapping.get("axisX"), f"{hand} axisX"),
            _finite_scalar(mapping.get("axisY"), f"{hand} axisY"),
        ),
        "axis_click": _required_bool(mapping, "axisClick", hand),
        "primary_button": _required_bool(mapping, "primaryButton", hand),
        "secondary_button": _required_bool(mapping, "secondaryButton", hand),
        "menu_button": _required_bool(mapping, "menuButton", hand),
        "raw": dict(mapping),
    }


def _parse_head(mapping: Any) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise AtomicXrError("headset state must be an object")
    pose = _finite_tuple(mapping.get("pose"), 7, "headset pose")
    quaternion_norm = math.sqrt(sum(component * component for component in pose[3:7]))
    if quaternion_norm < 1.0e-9:
        raise AtomicXrError("headset quaternion has zero norm")
    return {"pose_xyz_xyzw": pose, "raw": dict(mapping)}


@dataclass(frozen=True)
class AtomicXrPacket:
    """One atomic vendor packet containing the right and optional left hand."""

    sequence: int
    device_id: str
    pose_xyz_xyzw: tuple[float, ...]
    grip: float
    trigger: float
    axis_xy: tuple[float, ...]
    axis_click: bool
    primary_button: bool
    secondary_button: bool
    menu_button: bool
    app_focused: bool | None
    timestamp_ns: int
    predict_time_us: float | None
    input_device: int | None
    received_monotonic_ns: int
    received_unix_ns: int
    raw_right_controller: dict[str, Any]
    head_pose_xyz_xyzw: tuple[float, ...] | None = None
    raw_head: dict[str, Any] | None = None
    left_pose_xyz_xyzw: tuple[float, ...] | None = None
    left_grip: float | None = None
    left_trigger: float | None = None
    left_axis_xy: tuple[float, ...] | None = None
    left_axis_click: bool | None = None
    left_primary_button: bool | None = None
    left_secondary_button: bool | None = None
    left_menu_button: bool | None = None
    raw_left_controller: dict[str, Any] | None = None

    @property
    def position(self) -> tuple[float, float, float]:
        return self.pose_xyz_xyzw[:3]  # type: ignore[return-value]

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        return self.pose_xyz_xyzw[3:7]  # type: ignore[return-value]

    @property
    def quaternion_norm(self) -> float:
        return math.sqrt(sum(component * component for component in self.quaternion))

    def as_record(self) -> dict[str, Any]:
        """Return a lossless record for the right-controller fields used here."""
        return {
            "sequence": self.sequence,
            "device_id": self.device_id,
            "pose_xyz_xyzw": list(self.pose_xyz_xyzw),
            "grip": self.grip,
            "trigger": self.trigger,
            "axis_xy": list(self.axis_xy),
            "axis_click": self.axis_click,
            "primary_button": self.primary_button,
            "secondary_button": self.secondary_button,
            "menu_button": self.menu_button,
            "source_app_focused": self.app_focused,
            "source_timestamp_ns": self.timestamp_ns,
            "source_predict_time_us": self.predict_time_us,
            "source_input_device": self.input_device,
            "host_received_monotonic_ns": self.received_monotonic_ns,
            "host_received_unix_ns": self.received_unix_ns,
            "raw_right_controller": self.raw_right_controller,
            "head_pose_xyz_xyzw": (
                None
                if self.head_pose_xyz_xyzw is None
                else list(self.head_pose_xyz_xyzw)
            ),
            "raw_head": self.raw_head,
            "left_pose_xyz_xyzw": (
                None
                if self.left_pose_xyz_xyzw is None
                else list(self.left_pose_xyz_xyzw)
            ),
            "left_grip": self.left_grip,
            "left_trigger": self.left_trigger,
            "left_axis_xy": (
                None if self.left_axis_xy is None else list(self.left_axis_xy)
            ),
            "left_axis_click": self.left_axis_click,
            "left_primary_button": self.left_primary_button,
            "left_secondary_button": self.left_secondary_button,
            "left_menu_button": self.left_menu_button,
            "raw_left_controller": self.raw_left_controller,
        }


def parse_device_state_json(
    device_id: str,
    state_json: str | bytes,
    *,
    received_monotonic_ns: int | None = None,
    received_unix_ns: int | None = None,
) -> AtomicXrPacket:
    """Parse one complete ``PXREADevStateJson`` callback payload."""
    try:
        outer = json.loads(state_json)
        value_field = outer["value"]
        value = json.loads(value_field) if isinstance(value_field, str) else value_field
        controllers = value["Controller"]
        right = controllers["right"]
        left = controllers.get("left")
        head = value.get("Head")
        timestamp_ns = int(value["timeStampNs"])
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AtomicXrError(f"invalid XR device-state JSON: {exc}") from exc
    if timestamp_ns <= 0:
        raise AtomicXrError("XR source timestamp must be positive")

    right_state = _parse_controller(right, "right")
    left_state = None if left is None else _parse_controller(left, "left")
    head_state = None if head is None else _parse_head(head)

    predict_raw = value.get("predictTime")
    predict_time_us = None if predict_raw is None else _finite_scalar(predict_raw, "predictTime")
    input_raw = value.get("Input")
    input_device = None if input_raw is None else int(input_raw)
    app_state = value.get("appState")
    focus_raw = app_state.get("focus") if isinstance(app_state, dict) else None
    if focus_raw is not None and not isinstance(focus_raw, bool):
        raise AtomicXrError("appState.focus must be boolean when present")
    received_monotonic_ns = (
        time.monotonic_ns() if received_monotonic_ns is None else int(received_monotonic_ns)
    )
    received_unix_ns = time.time_ns() if received_unix_ns is None else int(received_unix_ns)

    return AtomicXrPacket(
        sequence=0,
        device_id=str(device_id),
        pose_xyz_xyzw=right_state["pose_xyz_xyzw"],
        grip=right_state["grip"],
        trigger=right_state["trigger"],
        axis_xy=right_state["axis_xy"],
        axis_click=right_state["axis_click"],
        primary_button=right_state["primary_button"],
        secondary_button=right_state["secondary_button"],
        menu_button=right_state["menu_button"],
        app_focused=focus_raw,
        timestamp_ns=timestamp_ns,
        predict_time_us=predict_time_us,
        input_device=input_device,
        received_monotonic_ns=received_monotonic_ns,
        received_unix_ns=received_unix_ns,
        raw_right_controller=right_state["raw"],
        head_pose_xyz_xyzw=(
            None if head_state is None else head_state["pose_xyz_xyzw"]
        ),
        raw_head=None if head_state is None else head_state["raw"],
        left_pose_xyz_xyzw=(
            None if left_state is None else left_state["pose_xyz_xyzw"]
        ),
        left_grip=None if left_state is None else left_state["grip"],
        left_trigger=None if left_state is None else left_state["trigger"],
        left_axis_xy=None if left_state is None else left_state["axis_xy"],
        left_axis_click=None if left_state is None else left_state["axis_click"],
        left_primary_button=(
            None if left_state is None else left_state["primary_button"]
        ),
        left_secondary_button=(
            None if left_state is None else left_state["secondary_button"]
        ),
        left_menu_button=None if left_state is None else left_state["menu_button"],
        raw_left_controller=None if left_state is None else left_state["raw"],
    )


def resolve_sdk_library(explicit: str | Path | None = None) -> Path:
    """Resolve the official PC Service client SDK without changing linker state."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    environment_path = os.environ.get("XR_ROBOT_SDK_LIB")
    if environment_path:
        candidates.append(Path(environment_path).expanduser())
    candidates.extend(
        [
            Path("/opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so"),
            Path("/opt/apps/roboticsservice/SDK/arm64/libPXREARobotSDK.so"),
        ]
    )
    discovered = ctypes.util.find_library("PXREARobotSDK")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates) or "(none)"
    raise AtomicXrError(f"cannot find libPXREARobotSDK.so; checked: {rendered}")


class AtomicXrClient:
    """Thread-safe latest-packet client backed by the vendor C callback."""

    def __init__(
        self,
        sdk_library: str | Path | None = None,
        *,
        queue_capacity: int = 4096,
    ):
        if int(queue_capacity) < 2:
            raise ValueError("queue_capacity must be at least 2")
        self.sdk_library = resolve_sdk_library(sdk_library)
        self.queue_capacity = int(queue_capacity)
        self._condition = threading.Condition()
        self._queue: deque[AtomicXrPacket] = deque(maxlen=self.queue_capacity)
        self._latest: AtomicXrPacket | None = None
        self._sequence = 0
        self._parse_error_count = 0
        self._queue_drop_count = 0
        self._last_error: str | None = None
        self._server_connected = False
        self._device_ids: set[str] = set()
        self._library: ctypes.CDLL | None = None
        self._callback: Any = None
        self._opened = False

    def open(self) -> "AtomicXrClient":
        if self._opened:
            raise AtomicXrError("XR client is already open")
        library = ctypes.CDLL(str(self.sdk_library))
        library.PXREAInit.argtypes = [ctypes.c_void_p, _CallbackType, ctypes.c_uint]
        library.PXREAInit.restype = ctypes.c_int
        library.PXREADeinit.argtypes = []
        library.PXREADeinit.restype = ctypes.c_int
        callback = _CallbackType(self._on_callback)
        result = int(library.PXREAInit(None, callback, ctypes.c_uint(PXREA_FULL_MASK)))
        if result != 0:
            raise AtomicXrError(f"PXREAInit failed with code {result}")
        self._library = library
        self._callback = callback
        self._opened = True
        return self

    def __enter__(self) -> "AtomicXrClient":
        return self.open()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @staticmethod
    def _c_string(user_data: int | None) -> str:
        if not user_data:
            return ""
        return ctypes.string_at(user_data).decode("utf-8", "replace")

    def _on_callback(self, context, event_type, status, user_data) -> None:
        del context, status
        try:
            if event_type == PXREA_SERVER_CONNECT:
                with self._condition:
                    self._server_connected = True
                    self._condition.notify_all()
                return
            if event_type == PXREA_SERVER_DISCONNECT:
                with self._condition:
                    self._server_connected = False
                    self._condition.notify_all()
                return
            if event_type in (PXREA_DEVICE_FIND, PXREA_DEVICE_MISSING):
                device_id = self._c_string(user_data)
                with self._condition:
                    if event_type == PXREA_DEVICE_FIND:
                        self._device_ids.add(device_id)
                    else:
                        self._device_ids.discard(device_id)
                    self._condition.notify_all()
                return
            if event_type != PXREA_DEVICE_STATE_JSON or not user_data:
                return

            received_monotonic_ns = time.monotonic_ns()
            received_unix_ns = time.time_ns()
            state = ctypes.cast(user_data, ctypes.POINTER(_DeviceStateJson)).contents
            device_id = bytes(state.devID).split(b"\0", 1)[0].decode("utf-8", "replace")
            raw_json = bytes(state.stateJson).split(b"\0", 1)[0]
            packet = parse_device_state_json(
                device_id,
                raw_json,
                received_monotonic_ns=received_monotonic_ns,
                received_unix_ns=received_unix_ns,
            )
            with self._condition:
                self._sequence += 1
                packet = replace(packet, sequence=self._sequence)
                if len(self._queue) == self.queue_capacity:
                    self._queue_drop_count += 1
                self._queue.append(packet)
                self._latest = packet
                self._last_error = None
                self._condition.notify_all()
        except Exception as exc:  # callbacks must never unwind through C
            with self._condition:
                self._parse_error_count += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._condition.notify_all()

    def wait_next(
        self,
        after_sequence: int = 0,
        *,
        timeout_s: float = 2.0,
    ) -> AtomicXrPacket:
        if not self._opened:
            raise AtomicXrError("XR client is not open")
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while self._latest is None or self._latest.sequence <= int(after_sequence):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    detail = "" if self._last_error is None else f"; last error: {self._last_error}"
                    raise AtomicXrError(f"timed out waiting for a new XR packet{detail}")
                self._condition.wait(remaining)
            return self._latest

    def drain(self) -> list[AtomicXrPacket]:
        with self._condition:
            packets = list(self._queue)
            self._queue.clear()
            return packets

    def diagnostics(self) -> dict[str, Any]:
        with self._condition:
            return {
                "sdk_library": str(self.sdk_library),
                "server_connected": self._server_connected,
                "device_ids": sorted(self._device_ids),
                "received_packet_count": self._sequence,
                "parse_error_count": self._parse_error_count,
                "queue_drop_count": self._queue_drop_count,
                "last_error": self._last_error,
            }

    def close(self) -> None:
        if not self._opened:
            return
        assert self._library is not None
        result = int(self._library.PXREADeinit())
        self._opened = False
        self._callback = None
        self._library = None
        if result != 0:
            raise AtomicXrError(f"PXREADeinit failed with code {result}")
