"""Data objects shared by the clean teleoperation modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .transforms import finite_vector


@dataclass(frozen=True)
class FrameSpec:
    name: str
    pose_m_rad: np.ndarray

    def __post_init__(self):
        object.__setattr__(
            self,
            "pose_m_rad",
            finite_vector(self.pose_m_rad, 6, f"{self.name} pose"),
        )


@dataclass(frozen=True)
class XrSample:
    pose_xyz_xyzw: np.ndarray
    grip: float
    trigger: float
    timestamp_ns: int
    received_monotonic: float
    stream_fresh: bool = True
    axis_xy: tuple[float, float] = (0.0, 0.0)
    axis_click: bool = False
    primary_button: bool = False
    secondary_button: bool = False
    menu_button: bool = False
    head_pose_xyz_xyzw: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "pose_xyz_xyzw",
            finite_vector(self.pose_xyz_xyzw, 7, "XR controller pose"),
        )
        for name, value in (("grip", self.grip), ("trigger", self.trigger)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        axis = finite_vector(self.axis_xy, 2, "XR controller axis")
        object.__setattr__(self, "axis_xy", (float(axis[0]), float(axis[1])))
        for name in (
            "axis_click",
            "primary_button",
            "secondary_button",
            "menu_button",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if self.head_pose_xyz_xyzw is not None:
            object.__setattr__(
                self,
                "head_pose_xyz_xyzw",
                finite_vector(self.head_pose_xyz_xyzw, 7, "XR headset pose"),
            )
        if int(self.timestamp_ns) <= 0:
            raise ValueError("timestamp_ns must be positive")

    @property
    def position(self) -> np.ndarray:
        return self.pose_xyz_xyzw[:3].copy()


@dataclass(frozen=True)
class RobotSnapshot:
    q_rad: np.ndarray
    transform_base_flange: np.ndarray
    transform_work_tcp: np.ndarray

    def __post_init__(self):
        object.__setattr__(
            self, "q_rad", finite_vector(self.q_rad, 6, "joint feedback")
        )


@dataclass(frozen=True)
class SessionSummary:
    session_id: int
    stop_reason: str
    requested_delta_work_m: np.ndarray
    actual_delta_work_m: np.ndarray
    requested_axis: str
    actual_axis: str
    distance_error_ratio: float | None
    cross_axis_limit_m: float | None
    orientation_change_deg: float
    result: str
    checks: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "stop_reason": self.stop_reason,
            "requested_delta_work_m": self.requested_delta_work_m.tolist(),
            "actual_delta_work_m": self.actual_delta_work_m.tolist(),
            "requested_axis": self.requested_axis,
            "actual_axis": self.actual_axis,
            "distance_error_ratio": self.distance_error_ratio,
            "cross_axis_limit_m": self.cross_axis_limit_m,
            "orientation_change_deg": self.orientation_change_deg,
            "result": self.result,
            "checks": self.checks,
        }
