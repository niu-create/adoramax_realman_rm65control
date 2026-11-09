"""Pure planning and monitored execution helpers for phase-one bring-up."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .acceptance import evaluate_session
from .config import TeleopConfig
from .models import FrameSpec, RobotSnapshot, SessionSummary
from .transforms import (
    base_flange_from_work_tcp,
    nearest_equivalent_angles,
    pose6_to_transform,
    rotation_to_euler_xyz,
    rotation_angle_rad,
    work_tcp_from_base_flange,
)


AXIS_VECTORS = {
    "X+": np.array([1.0, 0.0, 0.0]),
    "X-": np.array([-1.0, 0.0, 0.0]),
    "Y+": np.array([0.0, 1.0, 0.0]),
    "Y-": np.array([0.0, -1.0, 0.0]),
    "Z+": np.array([0.0, 0.0, 1.0]),
    "Z-": np.array([0.0, 0.0, -1.0]),
}


@dataclass(frozen=True)
class AxisPlan:
    axis: str
    distance_m: float
    start: RobotSnapshot
    requested_delta_operator_m: np.ndarray
    commanded_delta_work_m: np.ndarray
    rotation_work_from_operator: np.ndarray
    target_transform_work_tcp: np.ndarray
    target_transform_base_flange: np.ndarray
    target_q_rad: np.ndarray
    predicted_transform_work_tcp: np.ndarray

    @property
    def joint_delta_deg(self) -> np.ndarray:
        return np.rad2deg(self.target_q_rad - self.start.q_rad)

    @property
    def predicted_delta_work_m(self) -> np.ndarray:
        return self.predicted_transform_work_tcp[:3, 3] - self.start.transform_work_tcp[:3, 3]

    @property
    def predicted_delta_operator_m(self) -> np.ndarray:
        return self.rotation_work_from_operator.T @ self.predicted_delta_work_m


def _verify_frame(config: TeleopConfig, expected: FrameSpec, actual: FrameSpec, kind: str):
    if actual.name != expected.name:
        raise RuntimeError(f"active {kind} frame {actual.name!r} != expected {expected.name!r}")
    expected_transform = pose6_to_transform(expected.pose_m_rad)
    actual_transform = pose6_to_transform(actual.pose_m_rad)
    position_error_m = float(np.linalg.norm(actual_transform[:3, 3] - expected_transform[:3, 3]))
    rotation_error_deg = float(
        np.degrees(
            rotation_angle_rad(
                expected_transform[:3, :3],
                actual_transform[:3, :3],
            )
        )
    )
    if position_error_m > config.frame_position_tolerance_m:
        raise RuntimeError(f"{kind} frame position mismatch: {position_error_m * 1000.0:.3f} mm")
    if rotation_error_deg > config.frame_rotation_tolerance_deg:
        raise RuntimeError(f"{kind} frame rotation mismatch: {rotation_error_deg:.3f} deg")


@dataclass(frozen=True)
class MotionMonitorResult:
    q_rad: np.ndarray
    max_observed_speed_deg_s: float
    max_cross_feedback_error_deg: float
    peak_joint_current_change: np.ndarray | None


def read_verified_transforms(config: TeleopConfig, arm):
    work_frame, tool_frame = arm.read_frames()
    _verify_frame(config, config.work_frame, work_frame, "work")
    _verify_frame(config, config.tool_frame, tool_frame, "tool")
    return (
        pose6_to_transform(work_frame.pose_m_rad),
        pose6_to_transform(tool_frame.pose_m_rad),
    )


def snapshot(arm, transform_base_work, transform_flange_tcp, q_rad=None) -> RobotSnapshot:
    if q_rad is None:
        q_rad = arm.read_joint_positions()
    q_rad = np.asarray(q_rad, dtype=float)
    transform_base_flange = pose6_to_transform(arm.forward_pose(q_rad))
    transform_work_tcp = work_tcp_from_base_flange(
        transform_base_flange,
        transform_base_work,
        transform_flange_tcp,
    )
    return RobotSnapshot(q_rad, transform_base_flange, transform_work_tcp)


def require_same_a_point(arm, reference_q_rad, *, tolerance_deg=0.05):
    current_q = arm.read_joint_positions()
    error_deg = float(np.max(np.abs(np.rad2deg(current_q - np.asarray(reference_q_rad)))))
    if error_deg > float(tolerance_deg):
        raise RuntimeError(
            "joint feedback changed after A-point capture; "
            f"max_change={error_deg:.3f} deg, limit={tolerance_deg:.3f} deg. "
            "Rerun to capture the new A point"
        )
    return current_q


def plan_work_axis_step(
    config: TeleopConfig,
    arm,
    *,
    axis: str,
    distance_m: float,
    max_joint_delta_deg: float = 15.0,
    max_distance_m: float = 0.03,
) -> tuple[AxisPlan, np.ndarray, np.ndarray]:
    if axis not in AXIS_VECTORS:
        raise ValueError(f"unsupported work axis {axis!r}")
    distance_m = float(distance_m)
    max_distance_m = float(max_distance_m)
    if not np.isfinite(max_distance_m) or max_distance_m <= 0.0:
        raise ValueError("maximum commissioning distance must be positive")
    if not 0.0 < distance_m <= max_distance_m:
        raise ValueError(
            "commissioning distance must be in "
            f"(0, {max_distance_m:.6g}] metres"
        )
    transform_base_work, transform_flange_tcp = read_verified_transforms(config, arm)
    start = snapshot(arm, transform_base_work, transform_flange_tcp)
    requested_delta_operator = AXIS_VECTORS[axis] * distance_m
    commanded_delta_work = config.rotation_work_from_operator @ requested_delta_operator
    target_tcp = start.transform_work_tcp.copy()
    target_tcp[:3, 3] += commanded_delta_work
    target_flange = base_flange_from_work_tcp(
        target_tcp,
        transform_base_work,
        transform_flange_tcp,
    )
    target_pose = np.concatenate(
        (
            target_flange[:3, 3],
            rotation_to_euler_xyz(target_flange[:3, :3]),
        )
    )
    target_q = nearest_equivalent_angles(
        arm.inverse_kinematics(start.q_rad, target_pose),
        start.q_rad,
    )
    if np.any(target_q < config.joint_soft_lower_rad) or np.any(
        target_q > config.joint_soft_upper_rad
    ):
        raise RuntimeError("phase-one IK target exceeds RM65 joint soft limits")
    joint_delta_deg = np.rad2deg(target_q - start.q_rad)
    if float(np.max(np.abs(joint_delta_deg))) > float(max_joint_delta_deg):
        raise RuntimeError(
            "phase-one IK target changes a joint too far: "
            f"delta_deg={joint_delta_deg.tolist()}, limit={max_joint_delta_deg}"
        )
    predicted = snapshot(
        arm,
        transform_base_work,
        transform_flange_tcp,
        target_q,
    )
    predicted_delta = predicted.transform_work_tcp[:3, 3] - start.transform_work_tcp[:3, 3]
    position_error_m = float(np.linalg.norm(predicted_delta - commanded_delta_work))
    orientation_error_deg = float(
        np.degrees(
            rotation_angle_rad(
                start.transform_work_tcp[:3, :3],
                predicted.transform_work_tcp[:3, :3],
            )
        )
    )
    if position_error_m > 0.0002:
        raise RuntimeError(f"SDK IK/FK predicted TCP error is {position_error_m * 1000.0:.3f} mm")
    if orientation_error_deg > 0.1:
        raise RuntimeError(
            f"SDK IK/FK predicted TCP orientation error is {orientation_error_deg:.3f} deg"
        )
    return (
        AxisPlan(
            axis=axis,
            distance_m=distance_m,
            start=start,
            requested_delta_operator_m=requested_delta_operator,
            commanded_delta_work_m=commanded_delta_work,
            rotation_work_from_operator=config.rotation_work_from_operator,
            target_transform_work_tcp=target_tcp,
            target_transform_base_flange=target_flange,
            target_q_rad=target_q,
            predicted_transform_work_tcp=predicted.transform_work_tcp,
        ),
        transform_base_work,
        transform_flange_tcp,
    )


def wait_for_joint_target(
    arm,
    target_q_rad,
    *,
    max_joint_speed_deg_s: float,
    timeout_s: float = 60.0,
    target_tolerance_deg: float = 0.15,
):
    """Monitor a nonblocking MoveJ and stop immediately on measured overspeed."""
    target = np.asarray(target_q_rad, dtype=float)

    def read_feedback():
        if hasattr(arm, "read_commissioning_feedback"):
            feedback = arm.read_commissioning_feedback()
            return (
                np.asarray(feedback["q_rad"], dtype=float),
                float(feedback["cross_feedback_error_deg"]),
                np.asarray(feedback["joint_current"], dtype=float),
            )
        return arm.read_joint_positions(), 0.0, None

    previous, initial_cross_error, initial_current = read_feedback()
    previous_time = time.monotonic()
    started = previous_time
    stable_count = 0
    max_observed_speed = 0.0
    max_cross_error = initial_cross_error
    peak_current_change = np.zeros(6, dtype=float) if initial_current is not None else None
    while True:
        time.sleep(0.05)
        current, cross_error, joint_current = read_feedback()
        now = time.monotonic()
        elapsed = now - previous_time
        if elapsed <= 0.0:
            raise RuntimeError("non-positive feedback interval")
        speeds_deg_s = np.abs(np.rad2deg(current - previous)) / elapsed
        observed = float(np.max(speeds_deg_s))
        max_observed_speed = max(max_observed_speed, observed)
        max_cross_error = max(max_cross_error, cross_error)
        if peak_current_change is not None:
            peak_current_change = np.maximum(
                peak_current_change,
                np.abs(joint_current - initial_current),
            )
        if observed > float(max_joint_speed_deg_s):
            stop_status = arm.slow_stop()
            raise RuntimeError(
                "measured joint speed exceeded commissioning limit: "
                f"observed={observed:.3f} deg/s, "
                f"limit={max_joint_speed_deg_s:.3f} deg/s, "
                f"slow_stop_status={stop_status}"
            )
        error_deg = np.abs(np.rad2deg(target - current))
        if float(np.max(error_deg)) <= float(target_tolerance_deg):
            stable_count += 1
            if stable_count >= 3:
                return MotionMonitorResult(
                    q_rad=current,
                    max_observed_speed_deg_s=max_observed_speed,
                    max_cross_feedback_error_deg=max_cross_error,
                    peak_joint_current_change=peak_current_change,
                )
        else:
            stable_count = 0
        if now - started > float(timeout_s):
            stop_status = arm.slow_stop()
            raise RuntimeError(
                f"planned joint move timed out; slow_stop_status={stop_status}, "
                f"remaining_error_deg={error_deg.tolist()}"
            )
        previous = current
        previous_time = now


def evaluate_endpoint(plan: AxisPlan, actual: RobotSnapshot, reason: str) -> SessionSummary:
    actual_delta_work = actual.transform_work_tcp[:3, 3] - plan.start.transform_work_tcp[:3, 3]
    return evaluate_session(
        session_id=1,
        stop_reason=reason,
        requested_delta_work_m=plan.requested_delta_operator_m,
        actual_delta_work_m=(plan.rotation_work_from_operator.T @ actual_delta_work),
        start_rotation_work_tcp=plan.start.transform_work_tcp[:3, :3],
        actual_rotation_work_tcp=actual.transform_work_tcp[:3, :3],
    )
