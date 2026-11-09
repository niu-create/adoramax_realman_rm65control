"""Requirement-focused PICO-to-RM65 TCP translation controller."""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable

import numpy as np

from .acceptance import evaluate_session
from .auto_wrist import (
    AutoWristSwitch,
    filter_wrist_rotation,
    ray_rotation_components,
    select_wrist_component,
)
from .accessories import GripperChordLatch, GripperTriggerLatch, SafeButtonGesture
from .config import TeleopConfig
from .hardware import ArmError, ArmFeedbackTimeout, ArmProtocol, InverseKinematicsError
from .logging_jsonl import JsonlLogger
from .models import RobotSnapshot, SessionSummary, XrSample
from .pose_follow import PoseOrientationTarget, RayWristIntent
from .transforms import (
    base_flange_from_work_tcp,
    interpolate_transform,
    invert_transform,
    nearest_equivalent_angles,
    pose6_to_transform,
    quaternion_xyzw_to_rotation,
    rotation_angle_rad,
    rotation_exp,
    rotation_log,
    rotation_to_quaternion_xyzw,
    transform_to_pose6,
    work_tcp_from_base_flange,
)


class TeleopSafetyError(RuntimeError):
    pass


class XrTrackingDiscontinuity(TeleopSafetyError):
    """A recoverable XR tracking gap or relocalization discontinuity."""


class PositionTargetUnavailable(TeleopSafetyError):
    """An unreachable/stalled position target, distinct from hardware faults."""


class AbsoluteTargetUnavailable(TeleopSafetyError):
    """An exact absolute target is infeasible; keep its mapping and hold."""


@dataclass
class _GripReference:
    sample: XrSample
    tcp_position_work_m: np.ndarray


@dataclass
class _Session:
    session_id: int
    start_sample: XrSample
    start_snapshot: RobotSnapshot
    last_xr_position: np.ndarray
    last_requested_delta_operator_m: np.ndarray
    last_commanded_delta_work_m: np.ndarray
    requested_transform_work_tcp: np.ndarray
    last_command_q_rad: np.ndarray
    cancel_event: threading.Event
    generation: int
    motion_started: bool = False
    wrist_active: bool = False
    wrist_reference_rotation: np.ndarray | None = None
    wrist_target_rad: float | None = None
    wrist_selected_joint: int | None = None
    soft_limit_recovery_announced: bool = False
    elbow_branch_sign: int = 0
    started_monotonic: float = 0.0
    next_diagnostic_monotonic: float = 0.0
    minimum_joint_alpha: float = 1.0
    maximum_target_gap_m: float = 0.0
    maximum_orientation_change_deg: float = 0.0
    grasp_lift_assist_active: bool = False
    grasp_lift_retract_complete: bool = False
    shoulder_swing_active: bool = False
    last_command_monotonic: float | None = None
    auto_switch: AutoWristSwitch | None = None
    auto_transition_started: float | None = None
    auto_transition_from: str | None = None
    auto_transition_hand_position: np.ndarray | None = None
    auto_settle_anchor_q: np.ndarray | None = None
    auto_settle_since: float | None = None
    auto_wrist_reference_rotation: np.ndarray | None = None
    auto_wrist_target_q: np.ndarray | None = None
    auto_wrist_component: int | None = None
    auto_wrist_component_last_active: float | None = None
    auto_wrist_intent: RayWristIntent | None = None
    auto_wrist_pose_target: np.ndarray | None = None
    auto_wrist_j5_target_q: np.ndarray | None = None
    absolute_hold_q_rad: np.ndarray | None = None
    absolute_hold_tcp_position: np.ndarray | None = None
    absolute_syncing: bool = False
    absolute_last_wrist_command_monotonic: float | None = None
    absolute_last_verified_target: np.ndarray | None = None
    absolute_last_verified_at: float | None = None
    absolute_pose_paused: bool = False
    absolute_next_retry_at: float = 0.0
    absolute_settle_q_rad: np.ndarray | None = None
    absolute_settle_count: int = 0
    sync_target_tcp: np.ndarray | None = None
    sync_hand_position: np.ndarray | None = None
    sync_started_monotonic: float | None = None
    sync_stable_since: float | None = None
    sync_last_tcp_position: np.ndarray | None = None
    sync_progress_distance_m: float = 0.0
    sync_progress_monotonic: float | None = None
    follow_checked_at: float | None = None
    follow_progress_at: float | None = None
    follow_progress_m: float = 0.0
    follow_last_position: np.ndarray | None = None
    follow_last_target: np.ndarray | None = None
    pose_orientation: PoseOrientationTarget | None = None
    pose_progress_at: float | None = None
    pose_progress: float = 0.0
    pose_last_actual: np.ndarray | None = None
    pose_last_target: np.ndarray | None = None


class CleanTeleopController:
    GRIP_ACTIVATE = 0.9
    GRIP_RELEASE = 0.1
    SOFT_LIMIT_RECOVERY_SPEED_DEG_S = 30.0
    SOFT_LIMIT_RECOVERY_CLEARANCE_DEG = 1.0
    ELBOW_BRANCH_HYSTERESIS_DEG = 3.0
    TERMINAL_DIAGNOSTIC_PERIOD_S = 0.2
    CARTESIAN_COMMAND_LEAD_CYCLES = 3.0
    CARTESIAN_PATH_REANCHOR_M = 0.002
    IK_POSITION_TOLERANCE_M = 0.002
    IK_ORIENTATION_TOLERANCE_DEG = 2.0
    AUTO_WRIST_COMPONENT_RELEASE_S = 0.18
    FORWARD_REACH_KNEE_M = 0.0
    GRIP_SYNC_MAX_DISTANCE_M = 0.30
    GRIP_SYNC_POSITION_TOLERANCE_M = 0.005
    GRIP_SYNC_JOINT_TOLERANCE_DEG = 1.5
    GRIP_SYNC_STABLE_S = 0.12
    GRIP_SYNC_STALL_S = 2.0
    GRIP_SYNC_TIMEOUT_S = 15.0
    GRIP_SYNC_SPEED_DEG_S = 60.0
    GRIP_SYNC_MAX_FEEDBACK_LEAD_DEG = 5.0
    POSE_COMMAND_LEAD_MARGIN_DEG = 0.5
    ABSOLUTE_MAX_ORIENTATION_GAP_DEG = 60.0
    ABSOLUTE_POSITION_TOLERANCE_M = 0.001
    ABSOLUTE_ORIENTATION_TOLERANCE_DEG = 1.0
    ABSOLUTE_WRIST_POSITION_ERROR_M = 0.001
    ABSOLUTE_WRIST_UPDATE_PERIOD_S = 1.0 / 30.0

    def __init__(
        self,
        config: TeleopConfig,
        arm: ArmProtocol,
        logger: JsonlLogger,
        *,
        gripper=None,
        gripper_chord: bool = False,
        trigger_handler: Callable[[float, XrSample], None] | None = None,
        print_session_summary: bool = True,
        translation_gain_override: float | None = None,
        forward_reach_gain: float = 1.0,
        right_reach_gain: float = 1.0,
        vertical_reach_gain: float = 1.0,
        grip_sync_speed_deg_s: float = 60.0,
        position_fault_recovery: str = "release",
        translation_response: str = "original",
        wrist_control: str = "buttons",
        ray_j4_control: bool = False,
        ray_j4_sign: float = 1.0,
        tracking_frame: str = "world",
        resume_tracking_faults: bool = False,
        xr_recovery_window_s: float | None = None,
        grip_sync: str = "reanchor",
        absolute_control: str = "off",
        head_camera_calibration: str | None = None,
        allow_uncalibrated_absolute: bool = False,
        grip_sync_orientation_relax_deg: float = 0.0,
        left_reference_work_rotation=None,
    ):
        self.config = config
        # Opt-in installation adapter: keep the right controller's state
        # machine, but express its observed directions in the left work frame.
        self._left_reference_work_rotation = None
        if left_reference_work_rotation is not None:
            from .transforms import validate_rotation
            if config.controller_name != "left_controller" or config.arm_ip != "192.168.1.18":
                raise ValueError("left installation adapter requires left_controller at 192.168.1.18")
            self._left_reference_work_rotation = validate_rotation(left_reference_work_rotation)
        commissioned_hand = (config.controller_name == "right_controller"
                             or self._left_reference_work_rotation is not None)
        if translation_response not in ("original", "faster", "responsive"):
            raise ValueError("translation_response must be original, faster or responsive")
        if translation_response != "original" and (
            not commissioned_hand
            or config.cartesian_path_rate_limit
            or config.feedback_anchored_rate_limit
        ):
            raise ValueError("enhanced response requires the commissioned right command-anchored path")
        self.translation_response = translation_response
        if wrist_control not in ("buttons", "auto", "pose", "auto-ik"):
            raise ValueError("wrist_control must be buttons, auto, pose or auto-ik")
        if wrist_control in ("pose", "auto-ik") and (
            not commissioned_hand
            or absolute_control != "off"
            or grip_sync_orientation_relax_deg != 0.0
        ):
            raise ValueError("pose following requires the right arm, absolute_control=off and no orientation relaxation")
        if wrist_control in ("auto", "auto-ik") and (
            not commissioned_hand
            or not config.wrist_joint_control_enabled
            or config.j5_controller_axis == config.j6_controller_axis
        ):
            raise ValueError("auto wrist control requires the right arm and distinct enabled wrist axes")
        self.wrist_control = wrist_control
        if ray_j4_control and (
            wrist_control not in ("auto", "auto-ik")
            or not commissioned_hand
            or 1 in (config.j5_controller_axis, config.j6_controller_axis)
        ):
            raise ValueError("ray_j4_control requires right-arm auto or auto-ik wrist mode with local Y free")
        if ray_j4_sign not in (-1.0, 1.0):
            raise ValueError("ray_j4_sign must be -1 or +1")
        self.ray_j4_control = bool(ray_j4_control)
        self.ray_j4_sign = float(ray_j4_sign)
        self.auto_wrist_axes = (
            (1, config.j5_controller_axis, config.j6_controller_axis)
            if self.ray_j4_control
            else (config.j5_controller_axis, config.j6_controller_axis)
        )
        self.auto_wrist_indices = np.array(
            (3, 4, 5) if self.ray_j4_control else (4, 5), dtype=int
        )
        self.auto_wrist_labels = (
            ("J4_YAW", "J5_PITCH", "J6_TWIST")
            if self.ray_j4_control else ("J5_PITCH", "J6_TWIST")
        )
        if tracking_frame not in ("world", "head_relative"):
            raise ValueError("tracking_frame must be world or head_relative")
        self.tracking_frame = tracking_frame
        if grip_sync not in ("reanchor", "persistent"):
            raise ValueError("grip_sync must be reanchor or persistent")
        if grip_sync == "persistent" and tracking_frame != "head_relative":
            raise ValueError("persistent Grip sync requires head_relative tracking")
        self.grip_sync = grip_sync
        if absolute_control not in ("off", "pose", "wrist"):
            raise ValueError("absolute_control must be off, pose or wrist")
        if absolute_control != "off" and tracking_frame != "head_relative":
            raise ValueError("absolute control requires headset-relative tracking")
        if absolute_control != "off" and config.controller_name != "right_controller":
            raise ValueError("absolute camera control is commissioned for the right arm only")
        if absolute_control != "off" and not head_camera_calibration and not allow_uncalibrated_absolute:
            raise ValueError("absolute control requires head-camera calibration or explicit uncalibrated test mode")
        if allow_uncalibrated_absolute and absolute_control == "off":
            raise ValueError("uncalibrated absolute test requires pose or wrist control")
        if allow_uncalibrated_absolute and head_camera_calibration:
            raise ValueError("choose calibrated or uncalibrated absolute control")
        self.absolute_control = absolute_control
        self.allow_uncalibrated_absolute = bool(allow_uncalibrated_absolute)
        if not 0.0 <= float(grip_sync_orientation_relax_deg) <= 10.0:
            raise ValueError("grip_sync_orientation_relax_deg must be in [0, 10]")
        if grip_sync_orientation_relax_deg and config.controller_name != "right_controller":
            raise ValueError("Grip sync orientation fallback is commissioned for the right arm only")
        self.grip_sync_orientation_relax_deg = float(grip_sync_orientation_relax_deg)
        self._absolute_calibration = None
        self._absolute_retarget = None
        if absolute_control != "off" and head_camera_calibration:
            from .absolute_retarget import HeadCameraCalibration
            self._absolute_calibration = HeadCameraCalibration.load(head_camera_calibration)
        self.resume_tracking_faults = bool(resume_tracking_faults)
        # Opt-in collection policy; existing right-arm callers keep their behavior.
        if xr_recovery_window_s is not None:
            if not np.isfinite(xr_recovery_window_s) or not 0 < xr_recovery_window_s <= 1.0:
                raise ValueError("XR recovery window must be in (0, 1] seconds")
            if grip_sync != "reanchor" or absolute_control != "off":
                raise ValueError("bounded XR recovery requires relative reanchor control")
        self.xr_recovery_window_s = xr_recovery_window_s
        self._xr_pause_started = None
        self._xr_pause_failures = 0
        self._xr_requires_release = False
        self._feedback_fault_active = False
        self._xr_pause_stopping = 0
        self._xr_fresh_since = None
        self._xr_fresh_timestamp = None
        self._translation_rate_period_s = config.period_s
        translation_gain = (
            config.teleop_position_gain
            if translation_gain_override is None
            else float(translation_gain_override)
        )
        if (
            not np.isfinite(translation_gain)
            or translation_gain <= 0.0
            or translation_gain > config.teleop_position_gain
        ):
            raise ValueError(
                "translation_gain_override must be positive and may not exceed "
                "the configured teleoperation gain"
            )
        self.translation_gain = translation_gain
        if (
            not np.isfinite(forward_reach_gain)
            or not 1.0 <= forward_reach_gain <= 2.0
            or (forward_reach_gain > 1.0 and not commissioned_hand)
        ):
            raise ValueError("forward_reach_gain must be 1.0..2.0 for the right arm")
        self.forward_reach_gain = float(forward_reach_gain)
        for name, gain in (("right_reach_gain", right_reach_gain),
                           ("vertical_reach_gain", vertical_reach_gain)):
            if not np.isfinite(gain) or not 1.0 <= gain <= 1.5 or (
                gain != 1.0 and not commissioned_hand
            ):
                raise ValueError(f"{name} must be 1.0..1.5 for the right arm")
            setattr(self, name, float(gain))
        if not np.isfinite(grip_sync_speed_deg_s) or not 1 <= grip_sync_speed_deg_s <= 180:
            raise ValueError("grip_sync_speed_deg_s must be in [1, 180]")
        self.grip_sync_speed_deg_s = float(grip_sync_speed_deg_s)
        if position_fault_recovery not in ("release", "reanchor"):
            raise ValueError("position_fault_recovery must be release or reanchor")
        self.position_fault_recovery = position_fault_recovery
        self.hand = config.controller_name.removesuffix("_controller")
        self.hand_label = self.hand.upper()
        self.home_button_name = "B" if self.hand == "right" else "Y"
        self.arm = arm
        self.logger = logger
        self.gripper = gripper
        self.gripper_chord = bool(gripper_chord)
        self.trigger_handler = trigger_handler
        self.print_session_summary = bool(print_session_summary)
        self._lock = threading.RLock()
        self._session: _Session | None = None
        self._grip_reference: _GripReference | None = None
        self._release_seen = False
        self._generation = 0
        self._next_session_id = 1
        self._initialized = False
        self._shutdown = False
        self._last_stop_reason: str | None = None
        self._tracking_resume_reason: str | None = None
        self._next_tracking_resume_attempt = 0.0
        self._position_reanchor_pending = False
        self._position_reanchor_epoch = 0
        self._position_reanchor_after = 0.0
        self._position_reanchor_since: float | None = None
        self._position_reanchor_q: np.ndarray | None = None
        self._position_reanchor_hand: np.ndarray | None = None
        self._transform_base_work: np.ndarray | None = None
        self._transform_flange_tcp: np.ndarray | None = None
        self._home_target_q_rad: np.ndarray | None = None
        self._home_active = False
        self._home_deadline = 0.0
        self._home_next_feedback_time = 0.0
        self._selected_wrist_joint = 5
        self._button_gesture = SafeButtonGesture(config.home_button_hold_s)
        self._gripper_latch = (GripperChordLatch if self.gripper_chord else GripperTriggerLatch)(
            debounce_s=config.gripper_button_debounce_s,
        )
        # AG initialization establishes a deterministic closed target.
        self._gripper_target_open = False
        self.last_summary: SessionSummary | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._session is not None

    @property
    def collection_follow_ready(self) -> bool:
        """Read-only gate for starting a dataset after Grip positioning."""
        with self._lock:
            return (
                self._session is not None
                and self._session.sync_target_tcp is None
                and not self._position_reanchor_pending
                and not self._home_active
                and not self._shutdown
            )

    @property
    def release_seen(self) -> bool:
        with self._lock:
            return self._release_seen

    @property
    def home_active(self) -> bool:
        with self._lock:
            return self._home_active

    @property
    def selected_wrist_joint(self) -> int:
        with self._lock:
            return self._selected_wrist_joint

    def initialize(self) -> None:
        """Read and strictly verify the live work/tool coordinate frames."""
        if self._left_reference_work_rotation is not None and not getattr(self.arm, "_normalize_kinematics", False):
            raise ValueError("left installation adapter requires normalized base/flange FK and IK")
        actual_work, actual_tool = self.arm.read_frames()
        self._verify_frame("work", self.config.work_frame, actual_work)
        self._verify_frame("tool", self.config.tool_frame, actual_tool)
        self._transform_base_work = pose6_to_transform(actual_work.pose_m_rad)
        self._transform_flange_tcp = pose6_to_transform(actual_tool.pose_m_rad)
        if self.absolute_control != "off":
            from .absolute_retarget import AbsolutePoseRetarget, CameraAbsolutePoseRetarget
            rotation_work_xr = (
                self.config.rotation_work_from_operator
                @ self._transform_base_work[:3, :3].T
                @ self.config.rotation_base_xr
                @ np.diag(self.config.translation_xr_axis_signs)
            )
            self._absolute_retarget = (
                CameraAbsolutePoseRetarget(self._absolute_calibration)
                if self._absolute_calibration is not None
                else AbsolutePoseRetarget(None, rotation_work_xr)
            )
        self._home_target_q_rad = self.arm.read_joint_positions()
        self._initialized = True
        self.logger.write(
            "preflight_ok",
            arm_ip=self.config.arm_ip,
            arm_port=self.config.arm_port,
            work_frame={
                "name": actual_work.name,
                "pose_m_rad": actual_work.pose_m_rad,
            },
            tool_frame={
                "name": actual_tool.name,
                "pose_m_rad": actual_tool.pose_m_rad,
            },
            rotation_base_xr=self.config.rotation_base_xr,
            rotation_work_from_operator=self.config.rotation_work_from_operator,
            effective_rotation_work_from_operator=self._rotation_work_operator,
            left_reference_work_rotation=self._left_reference_work_rotation,
            translation_xr_axis_signs=self.config.translation_xr_axis_signs,
            position_scale=self.config.position_scale,
            teleop_position_gain=self.config.teleop_position_gain,
            effective_translation_gain=self.translation_gain,
            forward_reach_gain=self.forward_reach_gain,
            forward_reach_knee_m=self.FORWARD_REACH_KNEE_M,
            translation_response=self.translation_response,
            effective_wrist_scale=self.effective_wrist_scale,
            wrist_control=self.wrist_control,
            ray_j4_control=self.ray_j4_control,
            ray_j4_sign=self.ray_j4_sign,
            tracking_frame=self.tracking_frame,
            grip_sync=self.grip_sync,
            absolute_control=self.absolute_control,
            absolute_registration=(
                "first_grip_uncalibrated" if self.allow_uncalibrated_absolute
                else "head_camera_calibrated" if self.absolute_control != "off"
                else None
            ),
            head_camera_calibration=(
                None if self._absolute_calibration is None
                else self._absolute_calibration.as_dict()
            ),
            resume_tracking_faults=self.resume_tracking_faults,
            max_target_position_lag_m=self.config.max_target_position_lag_m,
            feedback_anchored_rate_limit=(
                self.config.feedback_anchored_rate_limit
            ),
            cartesian_path_rate_limit=self.config.cartesian_path_rate_limit,
            position_priority_ik=self.config.position_priority_ik,
            home_target_q_rad=self._home_target_q_rad,
            selected_wrist_joint=self._selected_wrist_joint,
            calibration_rms_error_deg=self.config.calibration_rms_error_deg,
        )

    def _verify_frame(self, kind, expected, actual) -> None:
        if actual.name != expected.name:
            raise TeleopSafetyError(
                f"active {kind} frame {actual.name!r} != expected {expected.name!r}"
            )
        expected_transform = pose6_to_transform(expected.pose_m_rad)
        actual_transform = pose6_to_transform(actual.pose_m_rad)
        position_error = float(
            np.linalg.norm(actual_transform[:3, 3] - expected_transform[:3, 3])
        )
        rotation_error_deg = float(
            np.degrees(
                rotation_angle_rad(
                    expected_transform[:3, :3],
                    actual_transform[:3, :3],
                )
            )
        )
        if position_error > self.config.frame_position_tolerance_m:
            raise TeleopSafetyError(
                f"{kind} frame position differs by {position_error * 1000.0:.3f} mm "
                f"(limit {self.config.frame_position_tolerance_m * 1000.0:.3f} mm)"
            )
        if rotation_error_deg > self.config.frame_rotation_tolerance_deg:
            raise TeleopSafetyError(
                f"{kind} frame rotation differs by {rotation_error_deg:.3f} deg "
                f"(limit {self.config.frame_rotation_tolerance_deg:.3f} deg)"
            )

    def _require_ready(self) -> None:
        if not self._initialized:
            raise RuntimeError("controller.initialize() must succeed before use")
        if self._shutdown:
            raise RuntimeError("controller has already shut down")

    def _snapshot(self, q_rad=None) -> RobotSnapshot:
        if q_rad is None:
            q_rad = self.arm.read_joint_positions()
        q_rad = np.asarray(q_rad, dtype=float)
        transform_base_flange = pose6_to_transform(self.arm.forward_pose(q_rad))
        transform_work_tcp = work_tcp_from_base_flange(
            transform_base_flange,
            self._transform_base_work,
            self._transform_flange_tcp,
        )
        return RobotSnapshot(q_rad, transform_base_flange, transform_work_tcp)

    @staticmethod
    def _pose_is_valid(sample: XrSample) -> bool:
        pose = sample.pose_xyz_xyzw
        quaternion_norm = float(np.linalg.norm(pose[3:7]))
        return (
            np.all(np.isfinite(pose))
            and not np.allclose(pose[:3], 0.0, atol=1.0e-4)
            and 0.5 <= quaternion_norm <= 1.5
        )

    def _head_pose_is_valid(self, sample: XrSample) -> bool:
        pose = sample.head_pose_xyz_xyzw
        if pose is None:
            return False
        quaternion_norm = float(np.linalg.norm(pose[3:7]))
        return (
            np.all(np.isfinite(pose))
            # A valid headset may be exactly at the XR tracking origin.
            and (self.absolute_control != "off" or self.wrist_control in ("pose", "auto-ik")
                 or not np.allclose(pose[:3], 0.0, atol=1.0e-4))
            and 0.5 <= quaternion_norm <= 1.5
        )

    def _control_sample(self, sample: XrSample) -> XrSample:
        """Return the controller pose in the configured operator tracking frame."""
        if self.tracking_frame == "world":
            return sample
        if not self._head_pose_is_valid(sample):
            raise XrTrackingDiscontinuity("headset pose is unavailable or invalid")
        head_pose = sample.head_pose_xyz_xyzw
        controller_rotation = quaternion_xyzw_to_rotation(sample.pose_xyz_xyzw[3:7])
        head_rotation = quaternion_xyzw_to_rotation(head_pose[3:7])
        # Position stays expressed in the calibrated XR axes while common
        # tracking-origin translation is removed.  Orientation is expressed
        # in the headset frame, so ray pitch/roll retains its physical meaning.
        relative_position = sample.position - head_pose[:3]
        relative_rotation = head_rotation.T @ controller_rotation
        relative_pose = np.concatenate(
            (relative_position, rotation_to_quaternion_xyzw(relative_rotation))
        )
        return replace(sample, pose_xyz_xyzw=relative_pose)

    def _mapped_translation(
        self, sample: XrSample, origin: XrSample, *, delta_xr_override=None
    ) -> np.ndarray:
        """Apply the commissioned hand-to-work mapping, including reach gain."""
        delta_xr = (
            sample.position - origin.position
            if delta_xr_override is None else np.asarray(delta_xr_override, dtype=float)
        )
        mapped_delta_xr = self.config.translation_xr_axis_signs * delta_xr
        delta_base = (
            self.config.rotation_base_xr @ mapped_delta_xr * self.config.position_scale
        )
        delta_operator = (
            self._transform_base_work[:3, :3].T @ delta_base * self.translation_gain
        )
        if delta_operator[0] > self.FORWARD_REACH_KNEE_M:
            delta_operator[0] = (
                self.FORWARD_REACH_KNEE_M
                + (delta_operator[0] - self.FORWARD_REACH_KNEE_M)
                * self.forward_reach_gain
            )
        # Commissioned operator axes: +X forward, -Y right, +Z up.
        if delta_operator[1] < 0.0:
            delta_operator[1] *= self.right_reach_gain
        delta_operator[2] *= self.vertical_reach_gain
        deadband = self.config.translation_axis_deadband_m
        if deadband > 0.0:
            delta_operator = np.sign(delta_operator) * np.maximum(
                np.abs(delta_operator) - deadband, 0.0
            )
        return delta_operator

    @property
    def _rotation_work_operator(self):
        rotation = self.config.rotation_work_from_operator
        if self._left_reference_work_rotation is not None:
            # Right auto mode uses legacy work/TCP FK as base/flange. Its
            # physical work displacement therefore includes R_right_work.
            # Left FK/IK is normalized; include that factor explicitly once.
            return self._left_reference_work_rotation @ rotation
        if self.wrist_control in ("pose", "auto-ik"):
            # Legacy FK was interpreted as base/flange even though the SDK
            # returned work/TCP. With normalized FK/IK, preserve its observed
            # translation directions explicitly, without the duplicate tool
            # offset that would move the TCP during an in-place rotation.
            rotation = self._transform_base_work[:3, :3] @ rotation
        return rotation

    def _reset_pose_follow(self, session, sample, snapshot):
        if self.wrist_control != "pose":
            return
        basis = (
            self._rotation_work_operator
            @ self._transform_base_work[:3, :3].T
            @ self.config.rotation_base_xr
            @ np.diag(self.config.translation_xr_axis_signs)
        )
        if self.tracking_frame == "head_relative":
            # _control_sample rotations are head-relative; the position map
            # still uses XR-world axes. Freeze their alignment at takeover.
            basis = basis @ quaternion_xyzw_to_rotation(sample.head_pose_xyz_xyzw[3:7])
        session.pose_orientation = PoseOrientationTarget(
            quaternion_xyzw_to_rotation(sample.pose_xyz_xyzw[3:7]),
            snapshot.transform_work_tcp[:3, :3], basis,
            deadband_deg=self.config.wrist_rotation_deadband_deg,
            max_step_deg=self.config.wrist_max_rotation_step_deg,
        )
        session.pose_progress_at = None
        session.pose_progress = 0.0
        session.pose_last_actual = None
        session.pose_last_target = None

    def _persistent_sync_target(
        self, sample: XrSample, actual: RobotSnapshot,
        *, rotation_work_tcp: np.ndarray | None = None,
    ) -> np.ndarray:
        reference = self._grip_reference
        if reference is None:
            raise TeleopSafetyError("initial Grip reference is unavailable")
        if np.any(actual.q_rad < self.config.joint_soft_lower_rad) or np.any(
            actual.q_rad > self.config.joint_soft_upper_rad
        ):
            raise TeleopSafetyError("Grip sync starts outside joint soft limits")
        delta_operator = self._mapped_translation(sample, reference.sample)
        target = actual.transform_work_tcp.copy()
        if rotation_work_tcp is not None:
            target[:3, :3] = rotation_work_tcp
        target[:3, 3] = (
            reference.tcp_position_work_m
            + self._rotation_work_operator @ delta_operator
        )
        distance = float(np.linalg.norm(
            target[:3, 3] - actual.transform_work_tcp[:3, 3]
        ))
        if distance > self.GRIP_SYNC_MAX_DISTANCE_M:
            raise PositionTargetUnavailable(
                f"Grip sync distance {distance * 1000:.1f} mm exceeds "
                f"{self.GRIP_SYNC_MAX_DISTANCE_M * 1000:.0f} mm"
            )
        offsets = [0.0]
        if self.grip_sync_orientation_relax_deg > 0:
            increments = np.arange(2.0, self.grip_sync_orientation_relax_deg + 0.01, 2.0)
            if not len(increments) or increments[-1] < self.grip_sync_orientation_relax_deg - 0.01:
                increments = np.r_[increments, self.grip_sync_orientation_relax_deg]
            for magnitude in increments:
                offsets.extend((-float(magnitude), float(magnitude)))
        last_error = "no IK candidate"
        for offset_deg in offsets:
            candidate = target.copy()
            if offset_deg:
                # Rotate about the TCP origin, so the requested XYZ stays exact.
                candidate[:3, :3] = (
                    target[:3, :3]
                    @ rotation_exp([0.0, np.deg2rad(offset_deg), 0.0])
                )
            flange = base_flange_from_work_tcp(
                candidate, self._transform_base_work, self._transform_flange_tcp
            )
            try:
                solution = self._solve_verified_ik(
                    actual.q_rad, transform_to_pose6(flange), actual.q_rad
                )
                if np.shape(solution) != (6,) or not np.all(np.isfinite(solution)):
                    raise ValueError("IK returned invalid joints")
                if np.any(solution < self.config.joint_soft_lower_rad) or np.any(
                    solution > self.config.joint_soft_upper_rad
                ):
                    raise ValueError("IK exceeds joint soft limits")
                reached = self._snapshot(solution).transform_work_tcp
                if not np.all(np.isfinite(reached)):
                    raise ValueError("FK returned an invalid pose")
                position_error = float(np.linalg.norm(reached[:3, 3] - target[:3, 3]))
                orientation_error = float(np.degrees(rotation_angle_rad(
                    reached[:3, :3], candidate[:3, :3]
                )))
                if position_error > self.GRIP_SYNC_POSITION_TOLERANCE_M or orientation_error > 5.0:
                    raise ValueError(
                        f"FK residual {position_error * 1000:.1f} mm / "
                        f"{orientation_error:.1f} deg"
                    )
            except ArmFeedbackTimeout:
                raise
            except ArmError as exc:
                if self.wrist_control in ("pose", "auto-ik") and not isinstance(exc, InverseKinematicsError):
                    raise
                last_error = str(exc)
                continue
            except Exception as exc:
                last_error = str(exc)
                continue
            if offset_deg:
                self.logger.write(
                    "grip_sync_orientation_relaxed",
                    offset_tool_y_deg=offset_deg,
                    original_target_tcp_work=target,
                    relaxed_target_tcp_work=candidate,
                    q_solution_rad=solution,
                    fk_position_error_m=position_error,
                    fk_orientation_error_deg=orientation_error,
                )
                print(
                    f"[{self.hand_label} RM65 GRIP SYNC] "
                    f"TCP XYZ preserved; tool orientation adjusted "
                    f"{offset_deg:+.1f} deg for IK.", flush=True,
                )
            return candidate
        if not self.grip_sync_orientation_relax_deg:
            if last_error.startswith("FK residual"):
                raise PositionTargetUnavailable(f"Grip sync target is unreachable: {last_error}")
            raise PositionTargetUnavailable(f"Grip sync target has no IK solution: {last_error}")
        raise PositionTargetUnavailable(f"Grip sync target has no verified IK solution: {last_error}")

    def _absolute_target(self, sample: XrSample, actual: RobotSnapshot):
        target, camera_target = self._absolute_retarget.target(sample)
        gap = float(np.linalg.norm(
            target[:3, 3] - actual.transform_work_tcp[:3, 3]
        ))
        angle_deg = float(np.degrees(rotation_angle_rad(
            actual.transform_work_tcp[:3, :3], target[:3, :3]
        )))
        if self.absolute_control == "wrist":
            gap = 0.0
            # Translation of the hand does not enter the wrist target, including
            # telemetry. Hold the physical TCP captured on this Grip press.
            session = self._session
            target[:3, 3] = (
                session.absolute_hold_tcp_position
                if session is not None else actual.transform_work_tcp[:3, 3]
            )
            if self._absolute_calibration is not None:
                camera_target = invert_transform(
                    self._absolute_calibration.transform_work_camera
                ) @ target
        elif gap > self.GRIP_SYNC_MAX_DISTANCE_M:
            raise TeleopSafetyError(
                f"absolute target is {gap * 1000:.1f} mm from TCP "
                f"(limit {self.GRIP_SYNC_MAX_DISTANCE_M * 1000:.0f} mm)"
            )
        if angle_deg > self.ABSOLUTE_MAX_ORIENTATION_GAP_DEG:
            raise TeleopSafetyError(
                f"absolute orientation gap {angle_deg:.1f} deg exceeds "
                f"{self.ABSOLUTE_MAX_ORIENTATION_GAP_DEG:.0f} deg"
            )
        return target, camera_target, gap, angle_deg

    def _verify_absolute_pose_ik(self, actual: RobotSnapshot, target) -> np.ndarray:
        flange = base_flange_from_work_tcp(
            target, self._transform_base_work, self._transform_flange_tcp
        )
        try:
            solution = self._solve_verified_ik(
                actual.q_rad, transform_to_pose6(flange), actual.q_rad
            )
        except ArmFeedbackTimeout:
            raise
        except InverseKinematicsError as exc:
            raise AbsoluteTargetUnavailable(f"absolute target IK failed: {exc}") from exc
        except Exception as exc:
            raise TeleopSafetyError(f"absolute target IK failed: {exc}") from exc
        if np.shape(solution) != (6,) or not np.all(np.isfinite(solution)):
            raise TeleopSafetyError("absolute IK returned invalid joints")
        if np.any(solution < self.config.joint_soft_lower_rad) or np.any(
            solution > self.config.joint_soft_upper_rad
        ):
            raise TeleopSafetyError("absolute IK exceeds joint soft limits")
        try:
            reached = self._snapshot(solution).transform_work_tcp
        except ArmFeedbackTimeout:
            raise
        except Exception as exc:
            raise TeleopSafetyError(f"absolute IK FK verification failed: {exc}") from exc
        position_error = float(np.linalg.norm(reached[:3, 3] - target[:3, 3]))
        orientation_error = float(np.degrees(rotation_angle_rad(
            reached[:3, :3], target[:3, :3]
        )))
        if (position_error > self.ABSOLUTE_POSITION_TOLERANCE_M
                or orientation_error > self.ABSOLUTE_ORIENTATION_TOLERANCE_DEG):
            raise TeleopSafetyError(
                f"absolute IK FK residual {position_error * 1000:.1f} mm / "
                f"{orientation_error:.1f} deg"
            )
        return solution

    def _toggle_wrist_joint(self) -> None:
        with self._lock:
            self._selected_wrist_joint = 6 if self._selected_wrist_joint == 5 else 5
            selected = self._selected_wrist_joint
        print(
            f"[{self.hand_label} RM65 WRIST MODE] Selected J{selected}.",
            flush=True,
        )
        self.logger.write(
            "wrist_joint_selected",
            button=self.home_button_name,
            joint=selected,
        )

    def _update_gripper_gesture(self, sample):
        if self.gripper_chord:
            return self._gripper_latch.update(
                first=sample.grip, second=sample.trigger,
                enabled=(not self.home_active and not self._position_reanchor_pending
                         and self._tracking_resume_reason is None),
            )
        return self._gripper_latch.update(grip=sample.grip, trigger=sample.trigger)

    def _command_gripper(self, action: str) -> None:
        if action == "TOGGLE":
            action = "CLOSE" if self._gripper_target_open else "OPEN"
        if self.gripper is None:
            print(
                f"[{self.hand_label} GRIPPER] {action} ignored: "
                "configured gripper is not available.",
                flush=True,
            )
            self.logger.write("gripper_unavailable", action=action)
            return
        try:
            result = self.gripper.command_open_state(action == "OPEN")
        except Exception as exc:
            print(
                f"[{self.hand_label} GRIPPER ERROR] {exc}; "
                "gripper control disabled.",
                flush=True,
            )
            self.logger.write(
                "gripper_error",
                action=action,
                error=f"{type(exc).__name__}: {exc}",
            )
            try:
                self.gripper.shutdown()
            except Exception:
                pass
            self.gripper = None
            return
        self._gripper_target_open = action == "OPEN"
        position_max = result.get("position_max", 1000)
        protocol = getattr(
            self.gripper,
            "PROTOCOL_NAME",
            self.config.gripper_protocol.upper(),
        )
        print(
            f"[{self.hand_label} GRIPPER COMMAND SENT] {protocol} {action} "
            f"target={result['target_position']}/{position_max} "
            f"force={result['force_percent']}% "
            f"register=0x{result['target_register']:04X}.",
            flush=True,
        )
        self.logger.write("gripper_command", action=action, result=result)

    def _start_home(self) -> None:
        if self._home_target_q_rad is None:
            raise TeleopSafetyError("startup Home pose is unavailable")
        if np.any(self._home_target_q_rad < self.config.joint_soft_lower_rad) or np.any(
            self._home_target_q_rad > self.config.joint_soft_upper_rad
        ):
            raise TeleopSafetyError(
                "startup Home pose is outside configured joint soft limits"
            )
        status = self.arm.start_home_joint_move(
            self._home_target_q_rad,
            speed_percent=self.config.home_speed_percent,
        )
        if status != 0:
            raise TeleopSafetyError(f"Home MoveJ failed: status={status}")
        with self._lock:
            self._home_active = True
            self._home_deadline = time.monotonic() + self.config.home_timeout_s
            self._home_next_feedback_time = 0.0
            self._release_seen = False
        print(
            f"[{self.hand_label} RM65 HOME START] Returning to the pose captured "
            f"at program startup. Grip, Trigger, {self.home_button_name}, XR loss, "
            "or timeout stops Home.",
            flush=True,
        )
        self.logger.write(
            "home_start",
            target_q_rad=self._home_target_q_rad,
            speed_percent=self.config.home_speed_percent,
        )

    def _cancel_home(self, reason: str, *, require_release: bool) -> None:
        with self._lock:
            if not self._home_active:
                if require_release:
                    self._release_seen = False
                return
            self._home_active = False
            self._release_seen = not require_release
        status = self.arm.slow_stop()
        print(
            f"[{self.hand_label} RM65 HOME STOP] "
            f"reason={reason} status={status}.",
            flush=True,
        )
        self.logger.write("home_stop", reason=reason, slow_stop_status=status)
        if status != 0:
            raise TeleopSafetyError(f"Home slow-stop failed: status={status}")

    def _advance_home(self, sample: XrSample) -> None:
        if (
            sample.grip >= self.GRIP_RELEASE
            or sample.trigger >= 0.1
            or sample.secondary_button
        ):
            self._cancel_home("operator_cancel", require_release=True)
            return
        now = time.monotonic()
        if now >= self._home_deadline:
            self._cancel_home("timeout", require_release=True)
            return
        if now < self._home_next_feedback_time:
            return
        self._home_next_feedback_time = now + 0.1
        q_rad = self.arm.read_joint_positions()
        error_deg = float(np.max(np.abs(np.rad2deg(q_rad - self._home_target_q_rad))))
        if error_deg <= 1.0:
            with self._lock:
                self._home_active = False
                self._release_seen = True
            print(
                f"[{self.hand_label} RM65 HOME COMPLETE] Startup pose reached.",
                flush=True,
            )
            self.logger.write("home_complete", maximum_joint_error_deg=error_deg)

    def step(self, sample: XrSample) -> None:
        """Process one latest XR sample at the configured control rate."""
        self._require_ready()
        control_sample = None
        tracking_error = None
        if self._pose_is_valid(sample):
            try:
                control_sample = self._control_sample(sample)
            except XrTrackingDiscontinuity as exc:
                tracking_error = str(exc)
        else:
            tracking_error = "controller pose is unavailable or invalid"
        self.logger.write(
            "xr_input",
            timestamp_ns=sample.timestamp_ns,
            position=sample.position,
            pose_xyz_xyzw=sample.pose_xyz_xyzw,
            head_pose=sample.head_pose_xyz_xyzw,
            control_position=(
                None if control_sample is None else control_sample.position
            ),
            control_pose_xyz_xyzw=(
                None if control_sample is None else control_sample.pose_xyz_xyzw
            ),
            tracking_frame=self.tracking_frame,
            grip=sample.grip,
            trigger=sample.trigger,
            axis_xy=sample.axis_xy,
            primary_button=sample.primary_button,
            secondary_button=sample.secondary_button,
            stream_fresh=sample.stream_fresh,
        )
        if self.trigger_handler is not None:
            # Trigger is deliberately outside the coordinate/control state
            # machine.  Existing application behavior can be attached here
            # without changing Grip-to-TCP mapping.
            self.trigger_handler(float(sample.trigger), sample)
        if not sample.stream_fresh:
            self.handle_stream_stale("xr_stale")
            return
        if control_sample is None:
            self._handle_tracking_fault("invalid_xr_pose", tracking_error)
            return
        sample = control_sample
        if not self._bounded_xr_resume_ready(sample):
            return

        if self.home_active:
            if self.gripper_chord:
                self._gripper_latch.reset()
            self._advance_home(sample)
            return

        with self._lock:
            session = self._session
        if session is None:
            if self._position_reanchor_pending:
                if self.gripper_chord:
                    self._gripper_latch.reset()
                if sample.grip < self.GRIP_RELEASE:
                    with self._lock:
                        self._position_reanchor_pending = False
                        self._position_reanchor_epoch += 1
                        self._release_seen = True
                    self.logger.write("position_reanchor_cancelled", reason="grip_released")
                elif sample.grip > self.GRIP_ACTIVATE:
                    self._try_position_reanchor(sample)
                return
            gesture = self._button_gesture.update(
                grip=sample.grip,
                trigger=sample.trigger,
                pressed=sample.secondary_button,
            )
            if gesture == "LONG" and self.config.home_enabled:
                if self.gripper_chord:
                    self._gripper_latch.reset()
                self._start_home()
                return
            if gesture == "SHORT" and self.config.wrist_joint_control_enabled and self.wrist_control == "buttons":
                self._toggle_wrist_joint()
                return
            gripper_action = self._update_gripper_gesture(sample)
            if gripper_action is not None:
                self._command_gripper(gripper_action)
                if not self.gripper_chord:
                    return
            if sample.grip < self.GRIP_RELEASE:
                with self._lock:
                    self._release_seen = True
                    self._tracking_resume_reason = None
                    self._next_tracking_resume_attempt = 0.0
                return
            if sample.grip > self.GRIP_ACTIVATE and self.release_seen:
                with self._lock:
                    resume_pending = self._tracking_resume_reason is not None
                    retry_at = self._next_tracking_resume_attempt
                if resume_pending and time.monotonic() < retry_at:
                    return
                try:
                    self._start_session(sample)
                except TeleopSafetyError as exc:
                    if isinstance(exc, PositionTargetUnavailable) and self._position_recovery_enabled:
                        self._queue_position_reanchor(str(exc))
                        return
                    with self._lock:
                        self._release_seen = False
                    print(
                        f"[{self.hand_label} RM65 GRIP SYNC REJECTED] {exc}. "
                        "Release Grip before trying again.",
                        flush=True,
                    )
                    self.logger.write(
                        "grip_sync_rejected",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                except Exception as exc:
                    if isinstance(exc, ArmFeedbackTimeout) and self._left_reference_work_rotation is not None:
                        with self._lock:
                            self._finish_xr_pause(failed=True)
                            self._feedback_fault_active = True
                            self._xr_requires_release = True
                            self._release_seen = False
                            self._tracking_resume_reason = "robot_feedback_timeout"
                        self.logger.write("tracking_resume_blocked", reason="robot_feedback_timeout", error=str(exc))
                        print(f"[{self.hand_label} RM65] 接管前状态读取超时，机械臂保持停止；"
                              "松开再按 Grip 重试，采集程序保留。", flush=True)
                        return
                    if (not isinstance(exc, ArmFeedbackTimeout) or not self.resume_tracking_faults
                            or self.xr_recovery_window_s is not None):
                        raise
                    with self._lock:
                        self._tracking_resume_reason = "robot_feedback_timeout"
                        self._release_seen = True
                        self._next_tracking_resume_attempt = time.monotonic() + 0.25
                    print(
                        f"[{self.hand_label} RM65 XR RESUME WAIT] Robot feedback "
                        f"is not ready ({exc}); retrying while Grip remains held.",
                        flush=True,
                    )
                    self.logger.write(
                        "tracking_resume_retry",
                        error=f"{type(exc).__name__}: {exc}",
                    )
            return

        # Home remains idle-only. The gripper chord can run during following.
        self._button_gesture.update(
            grip=sample.grip,
            trigger=sample.trigger,
            pressed=sample.secondary_button,
        )
        gripper_action = self._update_gripper_gesture(sample)

        if sample.grip < self.GRIP_RELEASE:
            self._stop_session("grip_released", require_release=False)
            with self._lock:
                self._release_seen = True
            if self.gripper_chord and gripper_action is not None:
                self._command_gripper(gripper_action)
            return

        try:
            self._advance_session(session, sample)
        except XrTrackingDiscontinuity as exc:
            self._handle_tracking_fault("xr_tracking_jump", str(exc))
            return
        except ArmFeedbackTimeout as exc:
            self._handle_tracking_fault("robot_feedback_timeout", str(exc))
            return
        except TeleopSafetyError as exc:
            if isinstance(exc, PositionTargetUnavailable) and self._position_recovery_enabled:
                self._queue_position_reanchor(str(exc))
                return
            # Input discontinuities and unreachable Cartesian requests are
            # recoverable operator-session faults. Stop only this arm, require
            # Grip release, and keep the other arm/runtime alive.
            self._stop_session("control_rejected", require_release=True)
            print(
                f"[{self.hand_label} RM65 CONTROL REJECTED] {exc}. "
                "Release Grip before trying again.",
                flush=True,
            )
            self.logger.write(
                "control_rejected",
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        except Exception as exc:
            self._stop_session("control_error", require_release=True)
            raise TeleopSafetyError(f"control cycle failed: {exc}") from exc

        if self.gripper_chord and gripper_action is not None:
            # Update the arm in this cycle before writing the gripper target.
            # A failed/cancelled arm cycle must not dispatch a stale chord.
            with self._lock:
                allowed = self._session is session and not session.cancel_event.is_set()
            if allowed:
                self._command_gripper(gripper_action)

    def _start_session(self, sample: XrSample, *, reanchor_snapshot=None) -> None:
        with self._lock:
            recovering_xr = self._xr_pause_started is not None
        start_snapshot = self._snapshot() if reanchor_snapshot is None else reanchor_snapshot
        sync_target = None
        absolute_target = None
        absolute_camera_target = None
        absolute_syncing = False
        absolute_rejection = None
        if self.absolute_control != "off":
            if (self.allow_uncalibrated_absolute
                    and self._absolute_retarget.transform_work_virtual_head is None):
                self._absolute_retarget.capture_origin(
                    sample, start_snapshot.transform_work_tcp
                )
                self.logger.write(
                    "absolute_origin_saved",
                    head_pose=sample.head_pose_xyz_xyzw,
                    controller_head_pose=sample.pose_xyz_xyzw,
                    tcp_work=start_snapshot.transform_work_tcp,
                    camera_calibration=(
                        None if self._absolute_calibration is None
                        else self._absolute_calibration.as_dict()
                    ),
                )
                print(
                    f"[{self.hand_label} RM65 ABSOLUTE ORIGIN SAVED] "
                    "The first-Grip head/controller/TCP mapping remains fixed "
                    "until program exit.", flush=True,
                )
            absolute_target, absolute_camera_target, gap, angle_deg = (
                self._absolute_target(sample, start_snapshot)
            )
            absolute_syncing = (
                gap > self.ABSOLUTE_POSITION_TOLERANCE_M
                or angle_deg > self.ABSOLUTE_ORIENTATION_TOLERANCE_DEG
            )
            if self.absolute_control == "pose":
                try:
                    self._verify_absolute_pose_ik(start_snapshot, absolute_target)
                except AbsoluteTargetUnavailable as exc:
                    absolute_rejection = str(exc)
            else:
                from .absolute_retarget import solve_wrist_orientation
                try:
                    solve_wrist_orientation(
                        start_snapshot.q_rad,
                        absolute_target[:3, :3],
                        start_snapshot.transform_work_tcp[:3, 3],
                        lambda q: self._snapshot(q).transform_work_tcp,
                        self.config.joint_soft_lower_rad,
                        self.config.joint_soft_upper_rad,
                        max_position_error_m=self.ABSOLUTE_WRIST_POSITION_ERROR_M,
                    )
                except ValueError as exc:
                    raise TeleopSafetyError(f"absolute wrist target rejected: {exc}") from exc
        elif self.grip_sync == "persistent":
            if self._grip_reference is None or reanchor_snapshot is not None:
                self._grip_reference = _GripReference(
                    sample=sample,
                    tcp_position_work_m=(
                        start_snapshot.transform_work_tcp[:3, 3].copy()
                    ),
                )
                self.logger.write(
                    "grip_reference_saved",
                    hand_relative_head_m=sample.position,
                    head_pose_xyz_xyzw=sample.head_pose_xyz_xyzw,
                    tcp_position_work_m=self._grip_reference.tcp_position_work_m,
                    q_rad=start_snapshot.q_rad,
                )
                print(
                    f"[{self.hand_label} RM65 GRIP REFERENCE SAVED] "
                    "Hand/head/TCP reference saved. "
                    "After release, the next Grip press will synchronize to the hand.",
                    flush=True,
                )
            else:
                sync_target = self._persistent_sync_target(sample, start_snapshot)
        started_monotonic = time.monotonic()
        with self._lock:
            if self._session is not None or self._shutdown:
                return
            if recovering_xr:
                # A feedback read can block while the watchdog cancels recovery.
                # Never install a session using a now-stale packet or old target.
                now = time.monotonic()
                self._expire_xr_pause(now)
                if (self._xr_pause_started is None or self._xr_pause_stopping
                        or not self._release_seen):
                    return
                if now - sample.received_monotonic > self.config.xr_stale_timeout_s:
                    self._xr_fresh_since = None
                    return
            self._generation += 1
            session = _Session(
                session_id=self._next_session_id,
                start_sample=sample,
                start_snapshot=start_snapshot,
                last_xr_position=sample.position,
                last_requested_delta_operator_m=np.zeros(3, dtype=float),
                last_commanded_delta_work_m=np.zeros(3, dtype=float),
                requested_transform_work_tcp=(start_snapshot.transform_work_tcp.copy()),
                last_command_q_rad=start_snapshot.q_rad.copy(),
                cancel_event=threading.Event(),
                generation=self._generation,
                elbow_branch_sign=self._elbow_branch_sign(start_snapshot.q_rad[2]),
                started_monotonic=started_monotonic,
                next_diagnostic_monotonic=started_monotonic,
                sync_target_tcp=sync_target,
                sync_hand_position=(
                    None if sync_target is None else sample.position.copy()
                ),
                sync_started_monotonic=(
                    None if sync_target is None else started_monotonic
                ),
                sync_progress_monotonic=(
                    None if sync_target is None else started_monotonic
                ),
                sync_last_tcp_position=(
                    None if sync_target is None
                    else start_snapshot.transform_work_tcp[:3, 3].copy()
                ),
                absolute_hold_q_rad=(
                    start_snapshot.q_rad.copy()
                    if self.absolute_control == "wrist" else None
                ),
                absolute_hold_tcp_position=(
                    start_snapshot.transform_work_tcp[:3, 3].copy()
                    if self.absolute_control == "wrist" else None
                ),
                absolute_syncing=absolute_syncing,
                absolute_last_verified_target=(
                    absolute_target.copy() if absolute_target is not None
                    and self.absolute_control == "pose" and absolute_rejection is None else None
                ),
                absolute_last_verified_at=(
                    started_monotonic if absolute_target is not None
                    and self.absolute_control == "pose" and absolute_rejection is None else None
                ),
            )
            self._next_session_id += 1
            if self.absolute_control == "off":
                self._reset_auto_wrist(session, sample, start_snapshot)
                self._reset_pose_follow(session, sample, start_snapshot)
            self._session = session
            self._release_seen = False
            self._last_stop_reason = None
            self._feedback_fault_active = False
            resume_reason = self._tracking_resume_reason
            self._finish_xr_pause(failed=False)
            self._tracking_resume_reason = None
            self._next_tracking_resume_attempt = 0.0
        set_streaming = getattr(self.arm, "set_streaming_motion_active", None)
        if callable(set_streaming):
            set_streaming(True)
        q_deg = np.rad2deg(start_snapshot.q_rad)
        lower_margin_deg = q_deg - (
            self.config.joint_lower_deg + self.config.joint_limit_margin_deg
        )
        upper_margin_deg = (
            self.config.joint_upper_deg - self.config.joint_limit_margin_deg
        ) - q_deg
        soft_margin_deg = np.minimum(lower_margin_deg, upper_margin_deg)
        print(
            f"[{self._terminal_timestamp()}][{self.hand_label} RM65 SESSION START] "
            f"id={session.session_id} "
            f"tracking={self.tracking_frame} "
            f"xr_m={self._format_vector(sample.position, 3)} "
            f"tcp_work_m={self._format_vector(start_snapshot.transform_work_tcp[:3, 3], 3)} "
            f"q_deg={self._format_vector(q_deg, 1)} "
            f"soft_margin_deg={self._format_vector(soft_margin_deg, 1)}",
            flush=True,
        )
        self.logger.write(
            "session_start",
            session_id=session.session_id,
            xr_start=sample.position,
            q_start_rad=start_snapshot.q_rad,
            transform_base_flange_start=start_snapshot.transform_base_flange,
            transform_work_tcp_start=start_snapshot.transform_work_tcp,
            trigger=sample.trigger,
            wrist_control=self.wrist_control,
            tracking_frame=self.tracking_frame,
            grip_sync=self.grip_sync,
            sync_target_tcp=sync_target,
            absolute_control=self.absolute_control,
            absolute_target_camera=absolute_camera_target,
            resumed_after_tracking_fault=resume_reason,
        )
        if self.absolute_control != "off":
            if absolute_rejection is not None:
                session.requested_transform_work_tcp = absolute_target.copy()
                self._pause_absolute_pose(session, start_snapshot, absolute_rejection)
            self.logger.write(
                "absolute_session_start",
                session_id=session.session_id,
                mode=self.absolute_control,
                target_work=absolute_target,
                target_camera=absolute_camera_target,
                syncing=absolute_syncing,
                paused=session.absolute_pose_paused,
            )
            print(
                f"[{self.hand_label} RM65 ABSOLUTE] "
                f"{'HOLD: move hand back into reachable space' if session.absolute_pose_paused else 'SYNC to current hand pose' if absolute_syncing else 'FOLLOW active'}; "
                f"mode={self.absolute_control}.", flush=True,
            )
        if sync_target is not None:
            gap = float(np.linalg.norm(
                sync_target[:3, 3] - start_snapshot.transform_work_tcp[:3, 3]
            ))
            print(
                f"[{self.hand_label} RM65 GRIP SYNC] Moving to the saved "
                f"head-relative hand position; gap={gap * 1000:.1f} mm. "
                "The target follows hand movement during approach; release Grip to stop.",
                flush=True,
            )
            self.logger.write(
                "grip_sync_start",
                session_id=session.session_id,
                gap_m=gap,
                target_tcp_work=sync_target,
            )
        if resume_reason is not None:
            reference_message = (
                "the absolute mapping is retained."
                if self.absolute_control != "off"
                else "the current hand/robot pose is the new control reference."
            )
            print(
                f"[{self.hand_label} RM65 XR RESUMED] Fresh tracking restored; "
                + reference_message,
                flush=True,
            )
            self.logger.write(
                "tracking_auto_resumed",
                session_id=session.session_id,
                previous_reason=resume_reason,
                xr_start=sample.position,
            )

    def _advance_absolute_session(self, session: _Session, sample: XrSample) -> None:
        """Follow the fixed-registration target, never integrating deltas."""
        now = time.monotonic()
        if session.absolute_pose_paused and now < session.absolute_next_retry_at:
            return
        actual = self._snapshot()
        target, camera_target, gap, angle_deg = self._absolute_target(sample, actual)
        session.requested_transform_work_tcp = target.copy()
        session.last_requested_delta_operator_m = self.config.rotation_work_from_operator.T @ (
            target[:3, 3] - session.start_snapshot.transform_work_tcp[:3, 3]
        )
        if self.absolute_control == "wrist":
            self._advance_absolute_wrist(
                session, sample, actual, target, camera_target, now
            )
            return

        if session.absolute_pose_paused:
            # Check settling without blocking the XR loop or release watchdog.
            previous = session.absolute_settle_q_rad
            stable = (previous is not None and
                      np.max(np.abs(np.rad2deg(actual.q_rad - previous))) <= .02)
            session.absolute_settle_q_rad = actual.q_rad.copy()
            session.absolute_settle_count = session.absolute_settle_count + 1 if stable else 0
            session.absolute_next_retry_at = now + .1
            if session.absolute_settle_count < 2:
                return

        last = session.absolute_last_verified_target
        position_changed = (
            last is None or np.linalg.norm(target[:3, 3] - last[:3, 3]) > 1e-9
        )
        rotation_changed = (
            last is None or np.degrees(rotation_angle_rad(
                target[:3, :3], last[:3, :3]
            )) > 1e-5
        )
        if position_changed or rotation_changed or (
            now - (session.absolute_last_verified_at or 0.0) > .25
        ):
            try:
                self._verify_absolute_pose_ik(actual, target)
            except AbsoluteTargetUnavailable as exc:
                self._pause_absolute_pose(session, actual, str(exc))
                return
            session.absolute_last_verified_target = target.copy()
            session.absolute_last_verified_at = now

        if session.absolute_pose_paused:
            with self._lock:
                if self._session is not session or session.cancel_event.is_set():
                    return
                session.absolute_pose_paused = False
                session.last_command_q_rad = actual.q_rad.copy()
                session.last_command_monotonic = now
                session.absolute_syncing = True
            self.logger.write("absolute_pose_resumed", session_id=session.session_id,
                              target_work=target, q_actual_rad=actual.q_rad)
            print(f"[{self.hand_label} RM65 ABSOLUTE RESUMED] Target reachable; "
                  "continuing with the same absolute mapping.", flush=True)

        if (gap <= self.ABSOLUTE_POSITION_TOLERANCE_M
                and angle_deg <= self.ABSOLUTE_ORIENTATION_TOLERANCE_DEG):
            if session.absolute_syncing:
                session.absolute_syncing = False
                print(
                    f"[{self.hand_label} RM65 ABSOLUTE FOLLOW] "
                    "Absolute target reached; live pose following.",
                    flush=True,
                )
            self.logger.write(
                "absolute_pose_hold", session_id=session.session_id,
                gap_m=gap, orientation_error_deg=angle_deg,
                target_camera=camera_target,
            )
            return

        limited_target, _, _ = self._limit_translation_target(actual, target)
        q_command, command_tcp, alpha, joint_alpha = self._solve_rate_limited_step(
            actual, limited_target, session.last_command_q_rad,
            elbow_branch_sign=session.elbow_branch_sign,
            joint_speed_cap_deg_s=self.GRIP_SYNC_SPEED_DEG_S,
        )
        # Applies to ordinary FOLLOW too, not only re-press synchronization.
        # Near extension an advancing command anchor otherwise runs far ahead
        # of the physical arm even though each individual step is rate-limited.
        lead = float(np.max(np.abs(np.rad2deg(q_command - actual.q_rad))))
        if lead > self.GRIP_SYNC_MAX_FEEDBACK_LEAD_DEG:
            previous = session.last_command_q_rad
            previous_lead = float(np.max(np.abs(np.rad2deg(previous - actual.q_rad))))
            q_command = (previous.copy()
                         if previous_lead <= self.GRIP_SYNC_MAX_FEEDBACK_LEAD_DEG
                         else actual.q_rad.copy())
            command_tcp = self._snapshot(q_command).transform_work_tcp
            alpha = 0.0
            joint_alpha = 0.0
        with self._lock:
            if self._session is not session or session.cancel_event.is_set():
                return
            generation, cancel_event = session.generation, session.cancel_event
        try:
            status = self.arm.send_joint_target(q_command, cancel_event=cancel_event)
        except Exception:
            if cancel_event.is_set():
                return
            raise
        if status != 0:
            raise TeleopSafetyError(f"absolute joint command failed: {status}")
        with self._lock:
            if self._session is not session or self._generation != generation or cancel_event.is_set():
                self.arm.slow_stop()
                return
            session.last_command_q_rad = q_command.copy()
            session.last_command_monotonic = time.monotonic()
            session.requested_transform_work_tcp = target.copy()
            session.last_commanded_delta_work_m = (
                command_tcp[:3, 3] - session.start_snapshot.transform_work_tcp[:3, 3]
            )
            session.motion_started = True
        self._log_cycle(session, sample, actual, target, command_tcp, q_command,
                        "command_sent", joint_alpha=joint_alpha, alpha=alpha)
        self.logger.write(
            "absolute_pose_command", session_id=session.session_id,
            target_work=target, target_camera=camera_target,
            actual_work=actual.transform_work_tcp, command_work=command_tcp,
            q_actual_rad=actual.q_rad, q_command_rad=q_command,
            alpha=alpha, joint_alpha=joint_alpha,
            syncing=session.absolute_syncing,
        )

    def _pause_absolute_pose(self, session, actual, reason):
        """Brake once on SDK no-solution, then retry only while Grip stays held."""
        with self._lock:
            if self._session is not session or session.cancel_event.is_set():
                return
            newly_paused = not session.absolute_pose_paused
            if newly_paused:
                status = self.arm.slow_stop()
                if status != 0:
                    raise TeleopSafetyError(f"absolute pose pause failed: {status}")
                session.absolute_settle_q_rad = actual.q_rad.copy()
                session.absolute_settle_count = 0
            session.absolute_pose_paused = True
            session.absolute_next_retry_at = time.monotonic() + .1
            session.absolute_last_verified_target = None
            session.absolute_last_verified_at = None
            session.last_command_q_rad = actual.q_rad.copy()
            session.motion_started = False
        if newly_paused:
            self.logger.write("absolute_pose_unreachable_hold", session_id=session.session_id,
                              error=reason, target_work=session.requested_transform_work_tcp,
                              q_actual_rad=actual.q_rad)
            print(f"[{self.hand_label} RM65 ABSOLUTE HOLD] Exact pose is unreachable. "
                  "Move the hand to a reachable position/orientation; "
                  "a reachable target will resume while Grip is held. Release Grip to stop.", flush=True)

    def _advance_absolute_wrist(
        self, session: _Session, sample: XrSample, actual: RobotSnapshot,
        target, camera_target, now: float,
    ) -> None:
        """Hold J1-J3 and TCP (1 mm numerical tolerance), solve ray orientation."""
        if session.absolute_last_wrist_command_monotonic is not None and (
            now - session.absolute_last_wrist_command_monotonic
            < self.ABSOLUTE_WRIST_UPDATE_PERIOD_S
        ):
            return
        fixed = session.absolute_hold_q_rad
        proximal_error_deg = float(np.max(np.abs(np.rad2deg(
            actual.q_rad[:3] - fixed[:3]
        ))))
        if proximal_error_deg > .5:
            raise TeleopSafetyError(
                f"fixed J1-J3 moved {proximal_error_deg:.2f} deg during wrist control"
            )
        from .absolute_retarget import solve_wrist_orientation
        seed = actual.q_rad.copy()
        seed[:3] = fixed[:3]
        try:
            desired, endpoint_drift, endpoint_angle = solve_wrist_orientation(
                seed, target[:3, :3], session.absolute_hold_tcp_position,
                lambda q: self._snapshot(q).transform_work_tcp,
                self.config.joint_soft_lower_rad,
                self.config.joint_soft_upper_rad,
                max_position_error_m=self.ABSOLUTE_WRIST_POSITION_ERROR_M,
            )
        except ValueError as exc:
            # An infeasible ray angle is ignored; the operator can rotate back
            # without losing the active Grip session or moving J1-J3.
            self.logger.write(
                "absolute_wrist_unreachable", session_id=session.session_id,
                error=str(exc), target_camera=camera_target,
            )
            session.absolute_last_wrist_command_monotonic = now
            self._hold_absolute_wrist(session, actual)
            return
        if (np.shape(desired) != (6,) or not np.all(np.isfinite(desired))
                or not np.array_equal(desired[:3], fixed[:3])
                or np.any(desired < self.config.joint_soft_lower_rad)
                or np.any(desired > self.config.joint_soft_upper_rad)):
            raise TeleopSafetyError("absolute wrist IK returned invalid or non-fixed joints")
        period = (
            self.ABSOLUTE_WRIST_UPDATE_PERIOD_S
            if session.absolute_last_wrist_command_monotonic is None
            else min(
                now - session.absolute_last_wrist_command_monotonic,
                2 * self.ABSOLUTE_WRIST_UPDATE_PERIOD_S,
            )
        )
        maximum_step = np.deg2rad(self.GRIP_SYNC_SPEED_DEG_S) * period
        previous = session.last_command_q_rad
        q_command = previous.copy()
        q_command[:3] = fixed[:3]
        q_command[3:] += np.clip(
            desired[3:] - previous[3:], -maximum_step, maximum_step
        )
        lead = np.deg2rad(self.GRIP_SYNC_MAX_FEEDBACK_LEAD_DEG)
        q_command[3:] = np.clip(
            q_command[3:], actual.q_rad[3:] - lead, actual.q_rad[3:] + lead
        )
        command_tcp = self._snapshot(q_command).transform_work_tcp
        # Check the short joint-interpolated path too: a valid endpoint alone
        # does not guarantee that the TCP stayed fixed between updates.
        command_drift = max(float(np.linalg.norm(
            self._snapshot(actual.q_rad + fraction * (q_command - actual.q_rad))
            .transform_work_tcp[:3, 3] - session.absolute_hold_tcp_position
        )) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0))
        if command_drift > self.ABSOLUTE_WRIST_POSITION_ERROR_M:
            self.logger.write(
                "absolute_wrist_path_blocked", session_id=session.session_id,
                command_drift_m=command_drift, target_camera=camera_target,
            )
            session.absolute_last_wrist_command_monotonic = now
            self._hold_absolute_wrist(session, actual)
            return
        if np.max(np.abs(q_command - previous)) < np.deg2rad(.03):
            session.absolute_last_wrist_command_monotonic = now
            return
        with self._lock:
            if self._session is not session or session.cancel_event.is_set():
                return
            generation, cancel_event = session.generation, session.cancel_event
        try:
            status = self.arm.send_joint_target(q_command, cancel_event=cancel_event)
        except Exception:
            if cancel_event.is_set():
                return
            raise
        if status != 0:
            raise TeleopSafetyError(f"absolute wrist command failed: {status}")
        with self._lock:
            if self._session is not session or self._generation != generation or cancel_event.is_set():
                self.arm.slow_stop()
                return
            session.last_command_q_rad = q_command.copy()
            session.last_command_monotonic = time.monotonic()
            session.absolute_last_wrist_command_monotonic = now
            session.motion_started = True
        if session.absolute_syncing and endpoint_angle <= self.ABSOLUTE_ORIENTATION_TOLERANCE_DEG:
            actual_angle = float(np.degrees(rotation_angle_rad(
                actual.transform_work_tcp[:3, :3], target[:3, :3]
            )))
            if actual_angle <= self.ABSOLUTE_ORIENTATION_TOLERANCE_DEG:
                session.absolute_syncing = False
                print(
                    f"[{self.hand_label} RM65 ABSOLUTE WRIST FOLLOW] "
                    "J4-J6 reached the absolute ray orientation.", flush=True,
                )
        self.logger.write(
            "absolute_wrist_command", session_id=session.session_id,
            q_actual_rad=actual.q_rad, q_command_rad=q_command,
            target_camera=camera_target,
            fixed_proximal_q_rad=fixed[:3],
            endpoint_drift_m=endpoint_drift,
            command_drift_m=command_drift,
            endpoint_orientation_error_deg=endpoint_angle,
        )

    def _hold_absolute_wrist(self, session, actual):
        """Discard any outstanding wrist target when constraints reject it."""
        with self._lock:
            if self._session is not session or session.cancel_event.is_set():
                return
            if session.motion_started:
                status = self.arm.slow_stop()
                if status != 0:
                    raise TeleopSafetyError(f"absolute wrist hold failed: {status}")
            session.last_command_q_rad = actual.q_rad.copy()
            session.motion_started = False

    def _advance_grip_sync(self, session: _Session, sample: XrSample) -> None:
        """Move toward the saved absolute hand position before live control."""
        now = time.monotonic()
        if now - session.sync_started_monotonic > self.GRIP_SYNC_TIMEOUT_S:
            raise PositionTargetUnavailable("Grip sync timed out")
        actual = self._snapshot()
        actual_position = actual.transform_work_tcp[:3, 3]
        previous_target_position = session.sync_target_tcp[:3, 3]
        # Measure robot progress toward the PREVIOUS target. Hand movement must
        # neither trip the stall guard nor keep resetting it for a stationary arm.
        if session.sync_last_tcp_position is not None:
            session.sync_progress_distance_m += float(
                np.linalg.norm(previous_target_position - session.sync_last_tcp_position)
                - np.linalg.norm(previous_target_position - actual_position)
            )
        session.sync_last_tcp_position = actual_position.copy()
        if session.sync_progress_distance_m >= 0.001:
            session.sync_progress_distance_m = 0.0
            session.sync_progress_monotonic = now
        hand_step = float(np.linalg.norm(sample.position - session.sync_hand_position))
        if hand_step > 1.0e-9:
            # Track every position update, using the normal XR jump guard in
            # _advance_session. Keep the press-time orientation rather than
            # adopting FK drift or accumulating optional orientation relaxation.
            session.sync_target_tcp = self._persistent_sync_target(
                sample, actual,
                rotation_work_tcp=session.start_snapshot.transform_work_tcp[:3, :3],
            )
            session.sync_hand_position = sample.position.copy()
        target = session.sync_target_tcp
        gap = float(np.linalg.norm(
            target[:3, 3] - actual.transform_work_tcp[:3, 3]
        ))
        joint_gap_deg = float(np.max(np.abs(np.rad2deg(
            session.last_command_q_rad - actual.q_rad
        ))))
        if gap <= self.GRIP_SYNC_POSITION_TOLERANCE_M and (
            joint_gap_deg <= self.GRIP_SYNC_JOINT_TOLERANCE_DEG
        ):
            # Once close enough, remaining close while moving also counts as
            # settled. Continue streaming during this interval to avoid a pause.
            session.sync_progress_distance_m = 0.0
            session.sync_progress_monotonic = now
            if session.sync_stable_since is None:
                session.sync_stable_since = now
            elif now - session.sync_stable_since >= self.GRIP_SYNC_STABLE_S:
                reference = self._grip_reference
                # Preserve the original position mapping. The current TCP
                # orientation is used so a prior J4/J5/J6 gesture is retained.
                anchor_tcp = actual.transform_work_tcp.copy()
                anchor_tcp[:3, 3] = reference.tcp_position_work_m
                session.start_sample = reference.sample
                session.start_snapshot = RobotSnapshot(
                    actual.q_rad.copy(),
                    actual.transform_base_flange.copy(),
                    anchor_tcp,
                )
                session.last_xr_position = sample.position.copy()
                # Retain the last sent command and its timestamp so FOLLOW
                # does not jump backward to delayed feedback or bank extra time.
                session.motion_started = True
                session.sync_target_tcp = None
                session.sync_hand_position = None
                self._reset_auto_wrist(session, sample, actual)
                self._reset_pose_follow(session, sample, actual)
                print(
                    f"[{self.hand_label} RM65 GRIP SYNC COMPLETE] "
                    "FOLLOW active at saved head-relative coordinates.",
                    flush=True,
                )
                self.logger.write(
                    "grip_sync_complete",
                    session_id=session.session_id,
                    gap_m=gap,
                    q_actual_rad=actual.q_rad,
                )
                return
        else:
            session.sync_stable_since = None
        if now - session.sync_progress_monotonic > self.GRIP_SYNC_STALL_S:
            raise PositionTargetUnavailable(
                f"Grip sync stalled with {gap * 1000:.1f} mm remaining"
            )
        solver_target, _, _ = self._limit_translation_target(actual, target)
        q_command, command_tcp, alpha, joint_alpha = self._solve_rate_limited_step(
            actual,
            solver_target,
            session.last_command_q_rad,
            elbow_branch_sign=session.elbow_branch_sign,
            joint_speed_cap_deg_s=self.grip_sync_speed_deg_s,
        )
        lead_deg = float(np.max(np.abs(np.rad2deg(q_command - actual.q_rad))))
        if lead_deg > self.GRIP_SYNC_MAX_FEEDBACK_LEAD_DEG:
            previous_lead_deg = float(np.max(np.abs(np.rad2deg(
                session.last_command_q_rad - actual.q_rad
            ))))
            q_command = (
                session.last_command_q_rad.copy()
                if previous_lead_deg <= self.GRIP_SYNC_MAX_FEEDBACK_LEAD_DEG
                else actual.q_rad.copy()
            )
            command_tcp = self._snapshot(q_command).transform_work_tcp
            alpha = 0.0
            joint_alpha = 0.0
        with self._lock:
            if self._session is not session or session.cancel_event.is_set():
                return
            generation = session.generation
            cancel_event = session.cancel_event
        try:
            status = self.arm.send_joint_target(q_command, cancel_event=cancel_event)
        except Exception:
            if cancel_event.is_set():
                return
            raise
        if status != 0:
            raise TeleopSafetyError(f"Grip sync joint command failed: {status}")
        with self._lock:
            if (
                self._session is not session
                or self._generation != generation
                or cancel_event.is_set()
            ):
                self.arm.slow_stop()
                return
            session.last_command_q_rad = q_command.copy()
            session.last_command_monotonic = time.monotonic()
        self.logger.write(
            "grip_sync_step",
            session_id=session.session_id,
            gap_m=gap,
            target_tcp_work=target,
            hand_position_m=sample.position,
            q_actual_rad=actual.q_rad,
            q_command_rad=q_command,
            command_tcp_work=command_tcp,
            alpha=alpha,
            joint_alpha=joint_alpha,
        )

    def _advance_session(self, session: _Session, sample: XrSample) -> None:
        with self._lock:
            if self._session is not session:
                return
            sample_step = sample.position - session.last_xr_position
            if (self.absolute_control != "wrist"
                    and np.linalg.norm(sample_step) > self.config.max_xr_sample_step_m):
                raise XrTrackingDiscontinuity(
                    "XR position jumped "
                    f"{np.linalg.norm(sample_step):.3f} m in one sample"
                )
            session.last_xr_position = sample.position
            syncing = session.sync_target_tcp is not None
            wrist_requested = (
                self.wrist_control == "buttons"
                and
                self.config.wrist_joint_control_enabled
                and sample.trigger >= self.config.wrist_trigger_threshold
            )
            leaving_wrist = session.wrist_active and not wrist_requested
            pose_rotation = None
            if self.wrist_control == "pose":
                try:
                    pose_rotation = session.pose_orientation.update(
                        quaternion_xyzw_to_rotation(sample.pose_xyz_xyzw[3:7])
                    )
                except ValueError as exc:
                    raise XrTrackingDiscontinuity(str(exc)) from exc
        if self.absolute_control != "off":
            self._advance_absolute_session(session, sample)
            return
        if syncing:
            self._advance_grip_sync(session, sample)
            return
        if self.wrist_control in ("auto", "auto-ik") and not session.soft_limit_recovery_announced:
            if self._advance_auto_controls(session, sample):
                session.follow_checked_at = None
                return
        if wrist_requested:
            session.follow_checked_at = None
            self._advance_wrist_session(session, sample)
            return
        if leaving_wrist:
            session.follow_checked_at = None
            self._reanchor_translation_session(session, sample)
            return

        with self._lock:
            if self._session is not session:
                return
            delta_xr = sample.position - session.start_sample.position
            delta_operator = self._mapped_translation(
                sample, session.start_sample, delta_xr_override=delta_xr
            )
            deadband = self.config.translation_axis_deadband_m
            delta_operator = self._apply_grasp_lift_assist(
                session,
                delta_operator,
            )
            if self._should_use_shoulder_swing(session, delta_operator):
                generation = session.generation
                cancel_event = session.cancel_event
                session.last_requested_delta_operator_m = delta_operator.copy()
                session.last_commanded_delta_work_m = np.zeros(3, dtype=float)
            else:
                generation = None
                cancel_event = None
        if generation is not None:
            self._advance_shoulder_swing(
                session,
                delta_operator,
                generation,
                cancel_event,
            )
            return

        with self._lock:
            if self._session is not session:
                return
            delta_work = self._rotation_work_operator @ delta_operator
            requested_tcp = session.start_snapshot.transform_work_tcp.copy()
            requested_tcp[:3, 3] += delta_work
            if pose_rotation is not None:
                requested_tcp[:3, :3] = pose_rotation
            session.last_requested_delta_operator_m = delta_operator.copy()
            session.last_commanded_delta_work_m = delta_work.copy()
            session.requested_transform_work_tcp = requested_tcp.copy()
            if (
                not session.motion_started
                and (pose_rotation is None or rotation_angle_rad(
                    pose_rotation, session.start_snapshot.transform_work_tcp[:3, :3]
                ) <= 1e-8)
                and (
                    np.linalg.norm(delta_operator) <= 1.0e-12
                    if deadband > 0.0
                    else np.linalg.norm(delta_xr)
                    <= self.config.motion_start_threshold_m
                )
            ):
                snapshot = self._snapshot()
                if self._recover_soft_limits(session, sample, snapshot):
                    return
                self._log_cycle(
                    session,
                    sample,
                    snapshot,
                    requested_tcp,
                    None,
                    None,
                    "stationary_hold",
                )
                return
            session.motion_started = True
            generation = session.generation
            cancel_event = session.cancel_event
            previous_command = session.last_command_q_rad.copy()

        actual = self._snapshot()
        if self._recover_soft_limits(session, sample, actual):
            return
        if self._position_recovery_enabled or self.wrist_control == "pose":
            self._check_position_progress(session, actual, requested_tcp)
        solver_target, target_limited, target_distance_m = (
            self._limit_translation_target(actual, requested_tcp)
        )
        q_command, command_tcp, alpha, joint_alpha = self._solve_rate_limited_step(
            actual,
            solver_target,
            previous_command,
            elbow_branch_sign=session.elbow_branch_sign,
        )
        with self._lock:
            if (
                self._session is not session
                or self._generation != generation
                or cancel_event.is_set()
            ):
                return
        try:
            status = self.arm.send_joint_target(
                q_command,
                cancel_event=cancel_event,
            )
        except Exception:
            # XR stale/Grip release can cancel while the blocking SDK call is
            # in flight.  That is an expected session stop, not a reason to
            # terminate the complete dual-arm runtime.
            if cancel_event.is_set():
                self.logger.write(
                    "joint_command_cancelled",
                    session_id=session.session_id,
                    reason="session_cancelled_during_sdk_call",
                )
                return
            raise
        if status != 0 and not cancel_event.is_set():
            raise TeleopSafetyError(f"position joint command failed: {status}")
        with self._lock:
            cancelled = (
                self._session is not session
                or self._generation != generation
                or cancel_event.is_set()
            )
            if not cancelled:
                session.last_command_q_rad = q_command.copy()
                session.last_command_monotonic = time.monotonic()
                command_elbow_sign = self._elbow_branch_sign(q_command[2])
                if command_elbow_sign:
                    session.elbow_branch_sign = command_elbow_sign
        if cancelled:
            self.arm.slow_stop()
            return
        self._log_cycle(
            session,
            sample,
            actual,
            requested_tcp,
            command_tcp,
            q_command,
            "command_sent",
            alpha=alpha,
            joint_alpha=joint_alpha,
            status=status,
            solver_target_transform_work_tcp=solver_target,
            target_limited=target_limited,
            target_distance_m=target_distance_m,
            translation_response=self.translation_response,
            rate_budget_ms=self._translation_rate_period_s * 1000.0,
            wrist_control=self.wrist_control,
        )

    def _reset_auto_wrist(self, session, sample, snapshot, *, mode="translation"):
        if self.wrist_control not in ("auto", "auto-ik"):
            return
        rotation = quaternion_xyzw_to_rotation(sample.pose_xyz_xyzw[3:7])
        first_reset = session.auto_switch is None
        if session.auto_switch is None:
            session.auto_switch = AutoWristSwitch(
                self.auto_wrist_axes,
                self.config.wrist_max_rotation_step_deg,
            )
        session.auto_switch.reset(sample.position, rotation, time.monotonic(), mode=mode)
        if first_reset:
            session.auto_switch.last_switch = float("-inf")
        session.auto_transition_started = None
        session.auto_transition_from = None
        session.auto_transition_hand_position = None
        session.auto_settle_anchor_q = None
        session.auto_settle_since = None
        session.auto_wrist_reference_rotation = rotation
        session.auto_wrist_target_q = snapshot.q_rad.copy()
        session.auto_wrist_component = None
        session.auto_wrist_component_last_active = None
        session.auto_wrist_j5_target_q = None
        if self.wrist_control == "auto-ik":
            # Reverse twist only; keep the commissioned local ray axes.
            signs = ([self.ray_j4_sign, self.config.j5_controller_sign, -self.config.j6_controller_sign]
                     if self.ray_j4_control else [self.config.j5_controller_sign, -self.config.j6_controller_sign])
            scales = ([1., self.effective_wrist_scale, self.effective_wrist_scale]
                      if self.ray_j4_control else [self.effective_wrist_scale] * 2)
            session.auto_wrist_intent = RayWristIntent(
                rotation, self.auto_wrist_axes, signs, scales,
                deadband_deg=self.config.wrist_rotation_deadband_deg,
                jump_deg=self.config.wrist_max_rotation_step_deg,
            )
            session.auto_wrist_pose_target = snapshot.transform_work_tcp.copy()
            session.pose_progress_at = None
            session.pose_last_actual = None
            session.pose_last_target = None
            session.pose_progress = 0.

    def _advance_auto_controls(self, session, sample) -> bool:
        """Return true when switching or wrist control owns this cycle."""
        now = time.monotonic()
        with self._lock:
            if self._session is not session or session.cancel_event.is_set():
                return True
            transitioning = session.auto_transition_started is not None
        if transitioning:
            actual = self._snapshot()
            with self._lock:
                if self._session is not session or session.cancel_event.is_set():
                    return True
                if now - session.auto_transition_started > 2.0:
                    # Low-follow CANFD can keep coasting after slow-stop. Do
                    # not lock the Grip session when it cannot settle in time:
                    # return to the last active mode at fresh feedback and let
                    # the operator make another clear gesture.
                    fallback = session.auto_transition_from or "translation"
                    requested_mode = session.auto_switch.mode
                    self._reanchor_translation_session(session, sample, actual=actual)
                    self._reset_auto_wrist(session, sample, actual, mode=fallback)
                    session.wrist_active = fallback == "wrist"
                    print(
                        f"[{self.hand_label} RM65 AUTO] Switch timed out; "
                        f"continuing in {fallback} mode. Grip remains active.",
                        flush=True,
                    )
                    self.logger.write(
                        "auto_wrist_switch_timeout_recovered",
                        session_id=session.session_id,
                        requested_mode=requested_mode,
                        fallback_mode=fallback,
                        q_rad=actual.q_rad,
                    )
                    return True
                if session.auto_settle_anchor_q is None or np.max(np.abs(
                    actual.q_rad - session.auto_settle_anchor_q
                )) > np.deg2rad(0.15):
                    session.auto_settle_anchor_q = actual.q_rad.copy()
                    session.auto_settle_since = now
                    return True
                if now - session.auto_settle_since < 0.06:
                    return True
                mode = session.auto_switch.mode
                transition_started = session.auto_transition_started
                hand_position_at_switch = session.auto_transition_hand_position
                self._reanchor_translation_session(session, sample, actual=actual)
                self._reset_auto_wrist(session, sample, actual, mode=mode)
                if mode == "translation" and hand_position_at_switch is not None:
                    hand_delta = sample.position - hand_position_at_switch
                    # This movement requested a mode switch. Replaying it
                    # would immediately move joints/TCP after the user stopped.
                    # Translation starts at the current hand and settled TCP.
                    self.logger.write(
                        "auto_wrist_translation_handoff",
                        session_id=session.session_id,
                        settle_s=now - transition_started,
                        hand_delta_m=hand_delta,
                        carried_delta_m=np.zeros(3),
                        discarded_delta_m=hand_delta,
                        handoff_policy="discard_switch_gesture",
                    )
                session.wrist_active = mode == "wrist"
                wrist_label = "J4+J5+J6 wrist" if self.ray_j4_control else "J5+J6 wrist"
                if self.wrist_control == "auto-ik":
                    wrist_label += " (J5 direct; J4/J6 whole-arm IK)"
                print(f"[{self.hand_label} RM65 AUTO] {wrist_label if mode == 'wrist' else 'XYZ translation'}", flush=True)
                self.logger.write("auto_wrist_mode", session_id=session.session_id, mode=mode, q_rad=actual.q_rad)
            return True

        rotation = quaternion_xyzw_to_rotation(sample.pose_xyz_xyzw[3:7])
        with self._lock:
            if self._session is not session or session.cancel_event.is_set():
                return True
            previous_mode = session.auto_switch.mode
            changed = session.auto_switch.update(sample.position, rotation, now)
            early_exit = previous_mode == "wrist" and session.auto_switch.wrist_exit_pending
            if early_exit and not changed:
                # Braking must not wait for 18 mm of hand travel or the dwell
                # interval while wrist targets continue accumulating.
                session.auto_switch.reset(sample.position, rotation, now, mode="translation")
                changed = True
            discarded_wrist_target = None
            if changed:
                session.auto_transition_started = now
                session.auto_transition_from = previous_mode
                session.auto_transition_hand_position = (
                    sample.position if session.auto_switch.mode == "translation" else None
                )
                session.auto_settle_anchor_q = None
                session.auto_settle_since = None
                if previous_mode == "wrist":
                    discarded_wrist_target = session.auto_wrist_target_q
                    session.auto_wrist_target_q = None
                    session.auto_wrist_reference_rotation = rotation.copy()
                    session.auto_wrist_component = None
                    session.auto_wrist_component_last_active = None
        if changed:
            status = self.arm.slow_stop()
            if status != 0:
                raise TeleopSafetyError(f"automatic wrist switch slow-stop failed: {status}")
            self.logger.write(
                "auto_wrist_switch_start", session_id=session.session_id,
                mode=session.auto_switch.mode, early_translation_exit=bool(early_exit),
                discarded_wrist_target_q_rad=discarded_wrist_target,
            )
            return True
        if session.auto_switch.mode == "wrist":
            if session.auto_switch.wrist_translation_blocked:
                # Do not integrate incidental rotation while confirming the
                # exit gesture, even before braking is triggered.
                session.auto_wrist_reference_rotation = rotation.copy()
                if session.auto_wrist_intent is not None:
                    session.auto_wrist_intent.discard_rotation(rotation)
                self._observe_left_gesture_hold(session, sample)
                return True
            self._advance_auto_wrist(session, sample)
            return True
        if session.auto_switch.translation_blocked:
            # A rotation gesture is emerging but has not crossed the wrist
            # threshold yet. Hold the arm rather than leak its hand arc into IK.
            self._observe_left_gesture_hold(session, sample)
            return True
        return False

    def _observe_left_gesture_hold(self, session, sample) -> None:
        # No new motion target is sent while a gesture is undecided, but the
        # arm may still be approaching the last accepted target. Keep recording
        # real encoders instead of repeating a frozen state then jumping on exit.
        if self._left_reference_work_rotation is None:
            return
        actual = self._snapshot()
        with self._lock:
            if self._session is not session or session.cancel_event.is_set():
                return
            self._log_cycle(
                session, sample, actual, session.requested_transform_work_tcp,
                None, None, "gesture_hold_feedback",
            )

    def _advance_auto_wrist(self, session, sample):
        """Lock one clear ray gesture; hold the other joints."""
        if self.wrist_control == "auto-ik":
            self._advance_auto_wrist_ik(session, sample)
            return
        actual = self._snapshot()
        current = quaternion_xyzw_to_rotation(sample.pose_xyz_xyzw[3:7])
        now = time.monotonic()
        with self._lock:
            if self._session is not session or session.cancel_event.is_set():
                return
            indices = self.auto_wrist_indices
            raw_ray_input, total_ray_rotation = ray_rotation_components(
                current,
                session.auto_wrist_reference_rotation,
                self.auto_wrist_axes,
            )
            if total_ray_rotation > np.deg2rad(
                self.config.wrist_max_rotation_step_deg
            ):
                session.auto_wrist_reference_rotation = current.copy()
                session.auto_wrist_target_q[indices] = actual.q_rad[indices]
                session.auto_wrist_component = None
                session.auto_wrist_component_last_active = None
                self.logger.write(
                    "auto_wrist_input_jump_ignored",
                    session_id=session.session_id,
                )
                return
            if session.auto_wrist_component is None:
                selected = select_wrist_component(raw_ray_input)
                if selected is None:
                    return
                session.auto_wrist_component = selected
                session.auto_wrist_component_last_active = now
                self.logger.write(
                    "auto_wrist_component_locked",
                    session_id=session.session_id,
                    component=self.auto_wrist_labels[selected],
                    raw_ray_components_deg=np.rad2deg(raw_ray_input),
                )
            active_component = session.auto_wrist_component
            filtered, reference, jumped = filter_wrist_rotation(
                current, session.auto_wrist_reference_rotation,
                self.auto_wrist_axes,
                self.config.wrist_rotation_deadband_deg,
                self.config.wrist_max_rotation_step_deg,
                active_component=active_component,
            )
            session.auto_wrist_reference_rotation = reference
            if jumped:
                session.auto_wrist_target_q[indices] = actual.q_rad[indices]
                self.logger.write("auto_wrist_input_jump_ignored", session_id=session.session_id)
                return
            signs = (
                np.array([self.ray_j4_sign, self.config.j5_controller_sign, self.config.j6_controller_sign])
                if self.ray_j4_control
                else np.array([self.config.j5_controller_sign, self.config.j6_controller_sign])
            )
            scales = (
                np.array([1.0, self.effective_wrist_scale, self.effective_wrist_scale])
                if self.ray_j4_control
                else np.array([self.effective_wrist_scale, self.effective_wrist_scale])
            )
            increment = filtered * signs * scales
            lower, upper = self.config.joint_soft_lower_rad[indices], self.config.joint_soft_upper_rad[indices]
            lag = np.deg2rad(self.config.wrist_max_target_lag_deg)
            target_lower = np.maximum(lower, actual.q_rad[indices] - lag)
            target_upper = np.minimum(upper, actual.q_rad[indices] + lag)
            if np.any(target_lower > target_upper):
                raise TeleopSafetyError("automatic wrist feedback outside joint bounds")
            session.auto_wrist_target_q[indices] = np.clip(
                session.auto_wrist_target_q[indices] + increment, target_lower, target_upper,
            )
            target = session.auto_wrist_target_q
            target_error = np.max(np.abs(target[indices] - actual.q_rad[indices]))
            if np.max(np.abs(filtered)) > 1.0e-10:
                session.auto_wrist_component_last_active = now
            elif (
                target_error <= np.deg2rad(0.05)
                and session.auto_wrist_component_last_active is not None
                and now - session.auto_wrist_component_last_active
                >= self.AUTO_WRIST_COMPONENT_RELEASE_S
            ):
                released_component = session.auto_wrist_component
                session.auto_wrist_component = None
                session.auto_wrist_component_last_active = None
                session.auto_wrist_reference_rotation = current.copy()
                self.logger.write(
                    "auto_wrist_component_released",
                    session_id=session.session_id,
                    component=self.auto_wrist_labels[released_component],
                )
                return
            if np.max(np.abs(increment)) <= 1e-10 and np.max(np.abs(target - actual.q_rad)) <= np.deg2rad(0.05):
                return
            period = self._command_budget_period()
            step = np.deg2rad(min(self.config.wrist_max_speed_deg_s, self.config.max_joint_speed_deg_s)) * period
            previous = session.last_command_q_rad[indices]
            command_lower = np.maximum(target_lower, previous - step)
            command_upper = np.minimum(target_upper, previous + step)
            if np.any(command_lower > command_upper):
                raise TeleopSafetyError("automatic wrist feedback moved outside bounded command interval")
            q_command = target.copy()
            q_command[indices] = np.clip(target[indices], command_lower, command_upper)
            if np.any(q_command < self.config.joint_soft_lower_rad) or np.any(q_command > self.config.joint_soft_upper_rad):
                raise TeleopSafetyError("automatic wrist hold pose outside joint soft limits")
            target_for_log = target[indices].copy()
            cancel_event, generation = session.cancel_event, session.generation
        try:
            status = self.arm.send_joint_target(q_command, cancel_event=cancel_event)
        except Exception:
            if cancel_event.is_set():
                return
            raise
        with self._lock:
            cancelled = self._session is not session or self._generation != generation or cancel_event.is_set()
            if not cancelled:
                session.last_command_q_rad = q_command.copy()
                session.last_command_monotonic = time.monotonic()
        if cancelled:
            self.arm.slow_stop()
            return
        self.logger.write(
            "auto_wrist_joint_command", session_id=session.session_id,
            target_deg=np.rad2deg(target_for_log), q_command_rad=q_command,
            q_actual_rad=actual.q_rad, wrist_scale=self.effective_wrist_scale,
            raw_ray_pitch_twist_deg=np.rad2deg(raw_ray_input[-2:]),
            filtered_j5_j6_input_deg=np.rad2deg(filtered[-2:]),
            raw_ray_yaw_deg=(float(np.rad2deg(raw_ray_input[0])) if self.ray_j4_control else None),
            filtered_j4_input_deg=(float(np.rad2deg(filtered[0])) if self.ray_j4_control else None),
            locked_component=self.auto_wrist_labels[active_component],
            controlled_joints=("J4", "J5", "J6") if self.ray_j4_control else ("J5", "J6"),
            total_ray_rotation_deg=float(np.rad2deg(total_ray_rotation)),
            rate_budget_ms=period * 1000.0, status=status,
        )

    def _advance_auto_wrist_ik(self, session, sample):
        """Direct J5 pitch; use the original FK/IK mapping for J4/J6 gestures."""
        actual = self._snapshot()
        self._recover_soft_limits(session, sample, actual)
        current = quaternion_xyzw_to_rotation(sample.pose_xyz_xyzw[3:7])
        now = time.monotonic()
        with self._lock:
            if self._session is not session or session.cancel_event.is_set():
                return
            target = session.auto_wrist_pose_target.copy()
            angle = rotation_angle_rad(target[:3, :3], actual.transform_work_tcp[:3, :3])
            gap = np.linalg.norm(target[:3, 3] - actual.transform_work_tcp[:3, 3])
            try:
                increment = session.auto_wrist_intent.update(
                    current, now, settled=angle <= np.deg2rad(.5) and gap <= .001,
                )
            except ValueError as exc:
                raise XrTrackingDiscontinuity(str(exc)) from exc
            component = session.auto_wrist_intent.component
            previous = session.last_command_q_rad.copy()
            changed_component = session.auto_wrist_intent.component_changed
            held_component = session.auto_wrist_intent.last_component
            was_direct_j5 = session.auto_wrist_j5_target_q is not None
            cancel_event = session.cancel_event

        if changed_component:
            # Drop only the unsent old-axis goal. Start the new intent from
            # the last bounded command so encoder lag cannot create a jump.
            target[:3, :3] = self._snapshot(previous).transform_work_tcp[:3, :3]
            self.logger.write("auto_wrist_ik_component_changed", session_id=session.session_id,
                              component=self.auto_wrist_labels[component], pending_goal_discarded=True)
        if held_component is not None and self.auto_wrist_indices[held_component] == 4:
            self._advance_auto_wrist_j5(session, actual, previous, increment[held_component])
            return
        if was_direct_j5:
            # J5-only rotation moves the TCP. Start subsequent wrist IK at
            # the last sent pose, never at the grasp point before that pitch.
            target = self._snapshot(previous).transform_work_tcp.copy()
            with self._lock:
                if self._session is not session or cancel_event.is_set():
                    return
                session.auto_wrist_j5_target_q = None
                session.pose_progress_at = None
                session.pose_last_actual = None
                session.pose_last_target = None
                session.pose_progress = 0.
            self.logger.write("auto_wrist_ik_reanchored_after_j5", session_id=session.session_id,
                              target_work_tcp=target, q_anchor_rad=previous)
        if np.max(np.abs(increment)) > 1e-10:
            # FK supplies the real robot's local wrist axes, including its
            # tool frame. Virtual joint angles are never actuator commands.
            virtual = previous.copy()
            virtual[self.auto_wrist_indices] += increment
            before = self._snapshot(previous).transform_work_tcp
            after = self._snapshot(virtual).transform_work_tcp
            target[:3, :3] = after[:3, :3] @ before[:3, :3].T @ target[:3, :3]
            # Preserve the old wrist target lag bound without forcing actual
            # J4/J5/J6 to equal the virtual gesture angles after compensation.
            turn = rotation_log(target[:3, :3] @ actual.transform_work_tcp[:3, :3].T)
            angle = np.linalg.norm(turn)
            lag = np.deg2rad(self.config.wrist_max_target_lag_deg)
            if angle > lag:
                target[:3, :3] = rotation_exp(turn * (lag / angle)) @ actual.transform_work_tcp[:3, :3]

        with self._lock:
            if self._session is not session or cancel_event.is_set():
                return
            session.auto_wrist_pose_target = target.copy()
            session.requested_transform_work_tcp = target.copy()
        self._check_pose_progress(session, actual, target)
        if (np.max(np.abs(increment)) <= 1e-10
                and np.linalg.norm(target[:3, 3] - actual.transform_work_tcp[:3, 3]) <= .001
                and rotation_angle_rad(target[:3, :3], actual.transform_work_tcp[:3, :3]) <= np.deg2rad(.05)):
            return
        command, command_tcp, alpha, _ = self._solve_pose_step(
            actual, target, previous, elbow_branch_sign=session.elbow_branch_sign,
            joint_speed_cap_deg_s=self.config.wrist_max_speed_deg_s,
            assisted_wrist=True,
        )
        self._send_auto_wrist_command(
            session, command, "auto_wrist_ik_command", actual,
            component=None if component is None else self.auto_wrist_labels[component],
            virtual_joint_increment_deg=np.rad2deg(increment),
            orientation_mapping="virtual_joint_fk", execution="whole_arm_ik",
            target_work_tcp=target, command_work_tcp=command_tcp, alpha=alpha,
        )

    def _advance_auto_wrist_j5(self, session, actual, previous, increment):
        """Change only J5; retain the last commands for the other five joints."""
        lead_limit = np.deg2rad([self.GRIP_SYNC_MAX_FEEDBACK_LEAD_DEG] * 3
                               + [self.config.wrist_max_target_lag_deg] * 3)
        if (np.any(previous < self.config.joint_soft_lower_rad)
                or np.any(previous > self.config.joint_soft_upper_rad)):
            raise TeleopSafetyError("invalid previous direct J5 command")
        if np.any(np.abs(previous - actual.q_rad) > lead_limit + 1e-9):
            raise PositionTargetUnavailable("direct J5 feedback fell behind the bounded command")
        lag = lead_limit[4] - np.deg2rad(self.POSE_COMMAND_LEAD_MARGIN_DEG)
        lower = max(self.config.joint_soft_lower_rad[4], actual.q_rad[4] - lag)
        upper = min(self.config.joint_soft_upper_rad[4], actual.q_rad[4] + lag)
        period = self._command_budget_period()
        self._translation_rate_period_s = period
        step = np.deg2rad(min(self.config.wrist_max_speed_deg_s, self.config.max_joint_speed_deg_s)) * period
        command_lower, command_upper = max(lower, previous[4] - step), min(upper, previous[4] + step)
        if command_lower > command_upper:
            raise PositionTargetUnavailable("direct J5 has no bounded command interval")
        with self._lock:
            if self._session is not session or session.cancel_event.is_set():
                return
            if session.auto_wrist_j5_target_q is None:
                session.auto_wrist_j5_target_q = previous.copy()
                session.pose_progress_at = None
                session.pose_last_actual = None
                session.pose_last_target = None
                session.pose_progress = 0.
            target_q = session.auto_wrist_j5_target_q.copy()
            target_q[4] = np.clip(target_q[4] + increment, lower, upper)
            session.auto_wrist_j5_target_q = target_q
        target = self._snapshot(target_q).transform_work_tcp
        with self._lock:
            if self._session is not session or session.cancel_event.is_set():
                return
            session.auto_wrist_pose_target = target.copy()
            session.requested_transform_work_tcp = target.copy()
        self._check_pose_progress(session, actual, target)
        if abs(increment) <= 1e-10 and abs(target_q[4] - actual.q_rad[4]) <= np.deg2rad(.05):
            return
        command = previous.copy()
        command[4] = np.clip(target_q[4], command_lower, command_upper)
        command_tcp = self._snapshot(command).transform_work_tcp
        self._send_auto_wrist_command(
            session, command, "auto_wrist_j5_command", actual,
            component="J5_PITCH", execution="direct_joint", controlled_joints=["J5"],
            j5_increment_deg=float(np.rad2deg(increment)), target_q_rad=target_q,
            target_work_tcp=target, command_work_tcp=command_tcp,
        )

    def _send_auto_wrist_command(self, session, command, event, actual, **details):
        """Commit either wrist execution path only after a successful send."""
        with self._lock:
            if self._session is not session or session.cancel_event.is_set():
                return
            generation, cancel_event = session.generation, session.cancel_event
        try:
            status = self.arm.send_joint_target(command, cancel_event=cancel_event)
        except Exception:
            if cancel_event.is_set():
                return
            raise
        if status != 0:
            raise TeleopSafetyError(f"assisted wrist joint command failed: {status}")
        with self._lock:
            if self._session is not session or self._generation != generation or cancel_event.is_set():
                self.arm.slow_stop()
                return
            session.last_command_q_rad = command.copy()
            session.last_command_monotonic = time.monotonic()
            session.motion_started = True
        self.logger.write(
            event, session_id=session.session_id,
            actual_work_tcp=actual.transform_work_tcp,
            q_actual_rad=actual.q_rad, q_command_rad=command,
            rate_budget_ms=self._translation_rate_period_s * 1000.,
            status=status, **details,
        )

    def _should_use_shoulder_swing(
        self,
        session: _Session,
        delta_operator,
    ) -> bool:
        """Latch J2-only swing for a left arm lowered from its raised pose."""
        if (
            not self.config.shoulder_swing_enabled
            or self.hand != "left"
        ):
            return False
        if session.shoulder_swing_active:
            return True
        requested = np.asarray(delta_operator, dtype=float)
        downward = -float(requested[2])
        downward_dominant = (
            downward >= self.config.shoulder_swing_activation_m
            and downward >= abs(float(requested[0]))
            and downward >= abs(float(requested[1]))
        )
        elbow_deg = abs(float(np.rad2deg(session.start_snapshot.q_rad[2])))
        if (
            downward_dominant
            and not session.motion_started
            and elbow_deg <= self.config.shoulder_swing_elbow_max_deg
        ):
            session.shoulder_swing_active = True
            session.motion_started = True
            print(
                f"[{self.hand_label} RM65 SHOULDER SWING] Raised-arm downward "
                f"mode active (start J3={elbow_deg:.1f}deg). Hand up/down "
                "controls J2 only; release Grip to return to XYZ mode.",
                flush=True,
            )
            self.logger.write(
                "shoulder_swing_start",
                session_id=session.session_id,
                start_j2_rad=float(session.start_snapshot.q_rad[1]),
                start_j3_rad=float(session.start_snapshot.q_rad[2]),
            )
        return session.shoulder_swing_active

    def _advance_shoulder_swing(
        self,
        session: _Session,
        delta_operator,
        generation: int,
        cancel_event: threading.Event,
    ) -> None:
        """Move only J2 so a vertically raised arm swings down as one link."""
        actual = self._snapshot()
        requested = np.asarray(delta_operator, dtype=float)
        target = float(session.start_snapshot.q_rad[1]) + float(requested[2]) * np.deg2rad(
            self.config.shoulder_swing_scale_deg_per_m
        )
        # This mode lowers from the captured raised pose and may return to it,
        # but never drives J2 beyond that known-safe startup-side endpoint.
        target = min(target, float(session.start_snapshot.q_rad[1]))
        target = float(
            np.clip(
                target,
                self.config.joint_soft_lower_rad[1],
                self.config.joint_soft_upper_rad[1],
            )
        )
        maximum_step = np.deg2rad(
            self.config.shoulder_swing_max_speed_deg_s
        ) * self.config.period_s
        command = actual.q_rad.copy()
        command[1] += float(
            np.clip(target - actual.q_rad[1], -maximum_step, maximum_step)
        )
        status = self.arm.send_joint_target(command, cancel_event=cancel_event)
        with self._lock:
            cancelled = (
                self._session is not session
                or self._generation != generation
                or cancel_event.is_set()
            )
            if not cancelled:
                session.last_command_q_rad = command.copy()
        if cancelled:
            self.arm.slow_stop()
            return
        self.logger.write(
            "shoulder_swing_command",
            session_id=session.session_id,
            hand_vertical_m=float(requested[2]),
            target_j2_rad=target,
            actual_q_rad=actual.q_rad,
            command_q_rad=command,
            status=status,
        )

    def _apply_grasp_lift_assist(
        self,
        session: _Session,
        delta_operator,
    ) -> np.ndarray:
        """Retreat the left TCP before lifting from a straight elbow.

        Operator X- is the commissioned shelf-forward direction, therefore X+
        is the safe retreat direction.  The assist is deliberately session
        scoped: Grip release cancels it together with every pending target.
        """
        requested = np.asarray(delta_operator, dtype=float).copy()
        if (
            not self.config.grasp_lift_assist_enabled
            or self.hand != "left"
        ):
            return requested

        upward_dominant = (
            requested[2] >= self.config.grasp_lift_activation_m
            and requested[2] >= abs(requested[0])
            and requested[2] >= abs(requested[1])
        )
        elbow_deg = abs(float(np.rad2deg(session.last_command_q_rad[2])))
        if (
            not session.grasp_lift_assist_active
            and upward_dominant
            and elbow_deg <= self.config.grasp_lift_elbow_trigger_deg
        ):
            session.grasp_lift_assist_active = True
            print(
                f"[{self.hand_label} RM65 GRASP LIFT] Straight elbow "
                f"J3={elbow_deg:.1f}deg: retreating "
                f"{self.config.grasp_lift_retract_m * 100.0:.0f} cm before lift. "
                "Release Grip to cancel.",
                flush=True,
            )
            self.logger.write(
                "grasp_lift_assist_start",
                session_id=session.session_id,
                elbow_deg=elbow_deg,
                retract_m=self.config.grasp_lift_retract_m,
            )

        if not session.grasp_lift_assist_active:
            return requested

        if (
            not session.grasp_lift_retract_complete
            and elbow_deg >= self.config.grasp_lift_elbow_release_deg
        ):
            session.grasp_lift_retract_complete = True
            print(
                f"[{self.hand_label} RM65 GRASP LIFT] Elbow clearance reached "
                f"at J3={elbow_deg:.1f}deg; continuing upward.",
                flush=True,
            )
            self.logger.write(
                "grasp_lift_assist_clearance",
                session_id=session.session_id,
                elbow_deg=elbow_deg,
            )

        if not session.grasp_lift_retract_complete:
            # Stage one is a pure, bounded retreat. This avoids asking the
            # straight-arm IK to retreat and lift at the same instant.
            return np.array(
                [self.config.grasp_lift_retract_m, 0.0, 0.0],
                dtype=float,
            )

        # Keep the gained shelf clearance while honoring the hand's lift and
        # lateral intent. Suppress any accidental forward component during the
        # same lift session so the elbow cannot immediately straighten again.
        requested[0] = max(0.0, requested[0]) + self.config.grasp_lift_retract_m
        return requested

    def _limit_translation_target(
        self,
        actual: RobotSnapshot,
        requested_tcp,
    ) -> tuple[np.ndarray, bool, float]:
        """Keep a delayed physical arm on the requested Cartesian ray."""
        target = np.asarray(requested_tcp, dtype=float).copy()
        delta = target[:3, 3] - actual.transform_work_tcp[:3, 3]
        distance = float(np.linalg.norm(delta))
        limit = float(self.config.max_target_position_lag_m)
        if limit <= 0.0 or distance <= limit:
            return target, False, distance
        target[:3, 3] = (
            actual.transform_work_tcp[:3, 3] + delta * (limit / distance)
        )
        return target, True, distance

    def _soft_limit_recovery_target(
        self,
        q_rad,
        *,
        continue_to_clearance: bool = False,
    ) -> tuple[np.ndarray | None, list[int]]:
        """Create a feedback-anchored command that only reduces limit violation."""
        q_rad = np.asarray(q_rad, dtype=float)
        hard_lower = np.deg2rad(self.config.joint_lower_deg)
        hard_upper = np.deg2rad(self.config.joint_upper_deg)
        if np.any(q_rad < hard_lower - 1.0e-9) or np.any(
            q_rad > hard_upper + 1.0e-9
        ):
            raise TeleopSafetyError(
                "joint feedback is outside configured mechanical limits"
            )
        recovery = q_rad.copy()
        violating = []
        step = np.deg2rad(self.SOFT_LIMIT_RECOVERY_SPEED_DEG_S) * (
            self.config.period_s
        )
        clearance = np.deg2rad(self.SOFT_LIMIT_RECOVERY_CLEARANCE_DEG)
        for index, value in enumerate(q_rad):
            lower = float(self.config.joint_soft_lower_rad[index])
            upper = float(self.config.joint_soft_upper_rad[index])
            inner_lower = lower + clearance
            inner_upper = upper - clearance
            lower_trigger = inner_lower if continue_to_clearance else lower
            upper_trigger = inner_upper if continue_to_clearance else upper
            if value < lower_trigger:
                recovery[index] = min(value + step, inner_lower)
                violating.append(index)
            elif value > upper_trigger:
                recovery[index] = max(value - step, inner_upper)
                violating.append(index)
        return (recovery, violating) if violating else (None, [])

    def _recover_soft_limits(
        self,
        session: _Session,
        sample: XrSample,
        actual: RobotSnapshot,
    ) -> bool:
        if self.wrist_control in ("pose", "auto-ik"):
            # A joint-only inward correction would violate the TCP constraint.
            if np.any(actual.q_rad < self.config.joint_soft_lower_rad) or np.any(
                actual.q_rad > self.config.joint_soft_upper_rad
            ):
                raise TeleopSafetyError("pose following feedback outside joint soft limits")
            return False
        q_command, violating = self._soft_limit_recovery_target(
            actual.q_rad,
            continue_to_clearance=session.soft_limit_recovery_announced,
        )
        if q_command is None:
            with self._lock:
                completed = session.soft_limit_recovery_announced
                if completed:
                    session.soft_limit_recovery_announced = False
                    session.start_sample = sample
                    session.start_snapshot = actual
                    session.last_xr_position = sample.position
                    session.last_requested_delta_operator_m = np.zeros(3)
                    session.last_commanded_delta_work_m = np.zeros(3)
                    session.requested_transform_work_tcp = (
                        actual.transform_work_tcp.copy()
                    )
                    session.last_command_q_rad = actual.q_rad.copy()
                    session.last_command_monotonic = time.monotonic()
                    session.motion_started = False
                    self._reset_auto_wrist(session, sample, actual)
                    self._home_target_q_rad = actual.q_rad.copy()
            if completed:
                print(
                    f"[{self.hand_label} RM65 SOFT-LIMIT RECOVERY COMPLETE] "
                    "All joints are inside the configured soft limits; "
                    "hand origin re-anchored.",
                    flush=True,
                )
                self.logger.write(
                    "soft_limit_recovery_complete",
                    session_id=session.session_id,
                    q_rad=actual.q_rad,
                    home_target_updated=True,
                )
                return True
            return False

        with self._lock:
            if self._session is not session or session.cancel_event.is_set():
                return True
            first_cycle = not session.soft_limit_recovery_announced
            session.soft_limit_recovery_announced = True
            generation = session.generation
            cancel_event = session.cancel_event
        if first_cycle:
            description = ", ".join(
                f"J{index + 1}={np.degrees(actual.q_rad[index]):+.2f}deg"
                for index in violating
            )
            print(
                f"[{self.hand_label} RM65 SOFT-LIMIT RECOVERY] {description}; "
                f"moving only violating joints inward at <="
                f"{self.SOFT_LIMIT_RECOVERY_SPEED_DEG_S:.0f} deg/s. "
                "Keep Grip held and keep the hand still.",
                flush=True,
            )
        status = self.arm.send_joint_target(q_command, cancel_event=cancel_event)
        with self._lock:
            cancelled = (
                self._session is not session
                or self._generation != generation
                or cancel_event.is_set()
            )
            if not cancelled:
                session.start_sample = sample
                session.start_snapshot = actual
                session.last_xr_position = sample.position
                session.last_requested_delta_operator_m = np.zeros(3)
                session.last_commanded_delta_work_m = np.zeros(3)
                session.requested_transform_work_tcp = (
                    actual.transform_work_tcp.copy()
                )
                session.last_command_q_rad = q_command.copy()
                session.last_command_monotonic = time.monotonic()
                session.motion_started = False
                self._reset_auto_wrist(session, sample, actual)
        if cancelled:
            self.arm.slow_stop()
            return True
        self.logger.write(
            "soft_limit_recovery_command",
            session_id=session.session_id,
            joints=[index + 1 for index in violating],
            q_actual_rad=actual.q_rad,
            q_command_rad=q_command,
            speed_limit_deg_s=self.SOFT_LIMIT_RECOVERY_SPEED_DEG_S,
            status=status,
        )
        return True

    def _reanchor_translation_session(
        self, session: _Session, sample: XrSample, *, actual: RobotSnapshot | None = None
    ) -> None:
        if actual is None:
            actual = self._snapshot()
        with self._lock:
            if self._session is not session:
                return
            session.start_sample = sample
            session.start_snapshot = actual
            session.last_xr_position = sample.position
            session.last_requested_delta_operator_m = np.zeros(3, dtype=float)
            session.last_commanded_delta_work_m = np.zeros(3, dtype=float)
            session.requested_transform_work_tcp = actual.transform_work_tcp.copy()
            session.last_command_q_rad = actual.q_rad.copy()
            session.last_command_monotonic = time.monotonic()
            session.motion_started = False
            session.wrist_active = False
            session.wrist_reference_rotation = None
            session.wrist_target_rad = None
            session.wrist_selected_joint = None
        self.logger.write(
            "wrist_mode_exit",
            session_id=session.session_id,
            q_rad=actual.q_rad,
        )

    def _advance_wrist_session(self, session: _Session, sample: XrSample) -> None:
        actual = self._snapshot()
        current_rotation = quaternion_xyzw_to_rotation(sample.pose_xyz_xyzw[3:7])
        with self._lock:
            if self._session is not session:
                return
            selected = self._selected_wrist_joint
            entering = (
                not session.wrist_active
                or session.wrist_selected_joint != selected
                or session.wrist_reference_rotation is None
            )
            if entering:
                session.start_sample = sample
                session.start_snapshot = actual
                session.last_requested_delta_operator_m = np.zeros(3, dtype=float)
                session.last_commanded_delta_work_m = np.zeros(3, dtype=float)
                session.requested_transform_work_tcp = actual.transform_work_tcp.copy()
                session.wrist_active = True
                session.wrist_reference_rotation = current_rotation
                session.wrist_target_rad = float(actual.q_rad[selected - 1])
                session.wrist_selected_joint = selected
                session.last_command_q_rad = actual.q_rad.copy()
                session.last_command_monotonic = time.monotonic()
                session.motion_started = True
                print(
                    f"[{self.hand_label} RM65 WRIST ACTIVE] "
                    f"Grip+Trigger controls only J{selected}.",
                    flush=True,
                )
                self.logger.write(
                    "wrist_mode_enter",
                    session_id=session.session_id,
                    joint=selected,
                    q_rad=actual.q_rad,
                )
                return
            reference_rotation = session.wrist_reference_rotation.copy()
            raw_rotation = rotation_log(current_rotation @ reference_rotation.T)
            raw_angle_deg = float(np.degrees(np.linalg.norm(raw_rotation)))
            if raw_angle_deg > self.config.wrist_max_rotation_step_deg:
                # A tracking re-localization can make the controller quaternion
                # jump by tens of degrees in one frame.  Never turn that jump
                # into robot motion, but do not tear down the whole teleop
                # process either: hold the joint and re-anchor at the new XR
                # orientation so that the next valid frame can continue.
                session.wrist_reference_rotation = current_rotation
                session.wrist_target_rad = float(actual.q_rad[selected - 1])
                session.last_command_q_rad = actual.q_rad.copy()
                session.last_command_monotonic = time.monotonic()
                self.logger.write(
                    "wrist_input_jump_ignored",
                    session_id=session.session_id,
                    joint=selected,
                    jump_deg=raw_angle_deg,
                    limit_deg=self.config.wrist_max_rotation_step_deg,
                    q_rad=actual.q_rad,
                )
                print(
                    f"[{self.hand_label} RM65 WRIST INPUT RESET] "
                    "Ignored XR rotation jump "
                    f"{raw_angle_deg:.1f} deg; J{selected} held and input "
                    "re-anchored.",
                    flush=True,
                )
                return
            deadband_rad = np.deg2rad(self.config.wrist_rotation_deadband_deg)
            raw_norm = float(np.linalg.norm(raw_rotation))
            responsive = self.translation_response == "responsive"
            if raw_norm <= deadband_rad:
                pending = abs(float(session.wrist_target_rad) - actual.q_rad[selected - 1])
                if not responsive or pending <= np.deg2rad(0.05):
                    self.logger.write(
                        "wrist_joint_hold",
                        session_id=session.session_id,
                        joint=selected,
                        raw_rotation_deg=np.degrees(raw_rotation),
                    )
                    return
                # Keep driving the existing target while Grip+Trigger is held.
                # A stationary controller adds no new rotation to that target.
                filtered_rotation = np.zeros(3, dtype=float)
            else:
                filtered_rotation = raw_rotation * (1.0 - deadband_rad / raw_norm)
                session.wrist_reference_rotation = (
                    rotation_exp(filtered_rotation) @ reference_rotation
                )
            if selected == 5:
                axis = self.config.j5_controller_axis
                sign = self.config.j5_controller_sign
            else:
                axis = self.config.j6_controller_axis
                sign = self.config.j6_controller_sign
            input_rad = float(filtered_rotation[axis]) * float(sign)
            session.wrist_target_rad = float(session.wrist_target_rad) + (
                input_rad * self.effective_wrist_scale
            )
            index = selected - 1
            lower = float(self.config.joint_soft_lower_rad[index])
            upper = float(self.config.joint_soft_upper_rad[index])
            lag = np.deg2rad(self.config.wrist_max_target_lag_deg)
            target = float(
                np.clip(
                    session.wrist_target_rad,
                    max(lower, actual.q_rad[index] - lag),
                    min(upper, actual.q_rad[index] + lag),
                )
            )
            session.wrist_target_rad = target
            wrist_period = self.config.period_s
            if responsive:
                wrist_period = self._command_budget_period()
                maximum_step = np.deg2rad(min(
                    self.config.wrist_max_speed_deg_s,
                    self.config.max_joint_speed_deg_s,
                )) * wrist_period
                previous = float(session.last_command_q_rad[index])
                command_lower = max(lower, actual.q_rad[index] - lag, previous - maximum_step)
                command_upper = min(upper, actual.q_rad[index] + lag, previous + maximum_step)
                if command_lower > command_upper:
                    raise TeleopSafetyError("wrist feedback moved outside the bounded command interval")
                command_value = float(np.clip(target, command_lower, command_upper))
            else:
                command_value = float(
                    np.clip(
                        target,
                        actual.q_rad[index] - self.config.wrist_max_joint_step_rad,
                        actual.q_rad[index] + self.config.wrist_max_joint_step_rad,
                    )
                )
            q_command = actual.q_rad.copy()
            q_command[index] = command_value
            generation = session.generation
            cancel_event = session.cancel_event

        status = self.arm.send_joint_target(q_command, cancel_event=cancel_event)
        with self._lock:
            cancelled = (
                self._session is not session
                or self._generation != generation
                or cancel_event.is_set()
            )
            if not cancelled:
                session.last_command_q_rad = q_command.copy()
                session.last_command_monotonic = time.monotonic()
        if cancelled:
            self.arm.slow_stop()
            return
        self.logger.write(
            "wrist_joint_command",
            session_id=session.session_id,
            joint=selected,
            controller_axis="xyz"[axis],
            input_deg=float(np.degrees(input_rad)),
            target_deg=float(np.degrees(target)),
            feedback_deg=float(np.degrees(actual.q_rad[index])),
            command_deg=float(np.degrees(command_value)),
            wrist_scale=self.effective_wrist_scale,
            wrist_response=self.translation_response,
            rate_budget_ms=wrist_period * 1000.0,
            q_command_rad=q_command,
            status=status,
        )

    @property
    def response_budget_cycles(self) -> float:
        return {"original": 1.0, "faster": 1.5, "responsive": 2.0}[self.translation_response]

    @property
    def effective_wrist_scale(self) -> float:
        return self.config.wrist_scale_factor

    def _command_budget_period(self) -> float:
        """Bound elapsed time so long SDK stalls never accumulate large steps."""
        period = self.config.period_s
        if self.translation_response != "original":
            with self._lock:
                last_sent = (
                    None if self._session is None
                    else self._session.last_command_monotonic
                )
            if last_sent is not None:
                period = min(max(0.0, time.monotonic() - last_sent), self.response_budget_cycles * period)
        return period

    def _translation_joint_step_limit(self) -> float:
        """Budget 180 deg/s from successful sends, with a 3 or 4 degree cap."""
        period = self._command_budget_period()
        self._translation_rate_period_s = period
        return float(np.deg2rad(self.config.max_joint_speed_deg_s) * period)

    def _solve_rate_limited_step(
        self,
        actual: RobotSnapshot,
        requested_tcp,
        previous_command_q,
        *,
        elbow_branch_sign: int = 0,
        joint_speed_cap_deg_s: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        if self.wrist_control == "pose":
            return self._solve_pose_step(
                actual, requested_tcp, previous_command_q,
                elbow_branch_sign=elbow_branch_sign,
                joint_speed_cap_deg_s=joint_speed_cap_deg_s,
            )
        previous_command_q = np.asarray(previous_command_q, dtype=float)
        path_preserving = self.config.cartesian_path_rate_limit
        interpolation_start = actual.transform_work_tcp
        ik_seed = actual.q_rad
        angle_reference = actual.q_rad
        command_anchor_q = actual.q_rad
        maximum_feedback_lead = (
            self.config.max_joint_step_rad * self.CARTESIAN_COMMAND_LEAD_CYCLES
        )
        if path_preserving:
            target_delta = (
                np.asarray(requested_tcp, dtype=float)[:3, 3]
                - actual.transform_work_tcp[:3, 3]
            )
            target_distance = float(np.linalg.norm(target_delta))
            previous_tcp = self._snapshot(previous_command_q).transform_work_tcp
            previous_offset = (
                previous_tcp[:3, 3] - actual.transform_work_tcp[:3, 3]
            )
            previous_joint_lead = float(
                np.max(np.abs(previous_command_q - actual.q_rad))
            )
            if target_distance > 1.0e-9:
                path_direction = target_delta / target_distance
                projected_lead = float(np.dot(previous_offset, path_direction))
                projected_lead = float(
                    np.clip(projected_lead, 0.0, target_distance)
                )
                lateral_error = float(
                    np.linalg.norm(
                        previous_offset - projected_lead * path_direction
                    )
                )
            else:
                projected_lead = 0.0
                lateral_error = float(np.linalg.norm(previous_offset))
            if (
                previous_joint_lead <= maximum_feedback_lead + 1.0e-12
                and lateral_error <= self.CARTESIAN_PATH_REANCHOR_M
            ):
                # Keep a small, bounded amount of command accumulation so a
                # delayed real arm can continue moving.  Reproject the command
                # onto the current feedback-to-target ray first, preventing
                # that lead from reintroducing cross-axis motion.
                lead_fraction = (
                    projected_lead / target_distance
                    if target_distance > 1.0e-9
                    else 0.0
                )
                interpolation_start = interpolate_transform(
                    actual.transform_work_tcp,
                    requested_tcp,
                    lead_fraction,
                )
                ik_seed = previous_command_q
                angle_reference = previous_command_q
                command_anchor_q = previous_command_q
        last_rejection = "no IK attempt"
        attempts = 4 if self._position_recovery_enabled else 17
        search_started = time.monotonic()
        for attempt in range(attempts):
            if attempt and self._position_recovery_enabled and time.monotonic() - search_started >= 0.02:
                break
            alpha = 0.5**attempt
            command_tcp = interpolate_transform(
                interpolation_start,
                requested_tcp,
                alpha,
            )
            command_flange = base_flange_from_work_tcp(
                command_tcp,
                self._transform_base_work,
                self._transform_flange_tcp,
            )
            flange_pose = transform_to_pose6(command_flange)
            try:
                q_solution = self._solve_verified_ik(
                    ik_seed,
                    flange_pose,
                    angle_reference,
                )
            except ArmFeedbackTimeout:
                raise
            except Exception as exc:
                last_rejection = f"IK: {exc}"
                continue
            if np.shape(q_solution) != (6,) or not np.all(np.isfinite(q_solution)):
                last_rejection = "invalid IK joints"
                continue
            actual_elbow_sign = self._elbow_branch_sign(angle_reference[2])
            solution_elbow_sign = self._elbow_branch_sign(q_solution[2])
            if (
                not self.config.position_priority_ik
                and
                elbow_branch_sign
                and actual_elbow_sign == elbow_branch_sign
                and solution_elbow_sign
                and solution_elbow_sign != elbow_branch_sign
            ):
                last_rejection = "elbow branch discontinuity"
                continue
            if np.any(q_solution < self.config.joint_soft_lower_rad) or np.any(
                q_solution > self.config.joint_soft_upper_rad
            ):
                last_rejection = "joint soft limit"
                continue
            if path_preserving:
                command_delta = q_solution - command_anchor_q
                requested_command_step = float(np.max(np.abs(command_delta)))
                maximum_path_step = self.config.max_joint_step_rad
                if joint_speed_cap_deg_s is not None:
                    maximum_path_step = min(
                        maximum_path_step,
                        np.deg2rad(joint_speed_cap_deg_s) * self._command_budget_period(),
                    )
                if requested_command_step > maximum_path_step + 1.0e-12:
                    last_rejection = "Cartesian path joint-rate step"
                    continue
                feedback_lead = float(
                    np.max(np.abs(q_solution - actual.q_rad))
                )
                if feedback_lead > maximum_feedback_lead + 1.0e-12:
                    last_rejection = "Cartesian path feedback lead"
                    continue
                # Send the complete IK result for an intentionally shortened
                # Cartesian step. Scaling the joint vector would bend the TCP
                # away from the requested straight line.
                command_tcp = self._snapshot(q_solution).transform_work_tcp
                return q_solution, command_tcp, alpha, 1.0
            command_anchor = (
                actual.q_rad
                if self.config.feedback_anchored_rate_limit
                else previous_command_q
            )
            command_delta = q_solution - command_anchor
            requested_command_step = float(np.max(np.abs(command_delta)))
            maximum_step = self._translation_joint_step_limit()
            if joint_speed_cap_deg_s is not None:
                maximum_step = min(
                    maximum_step,
                    np.deg2rad(joint_speed_cap_deg_s) * self._translation_rate_period_s,
                )
            if requested_command_step <= maximum_step + 1.0e-12:
                joint_alpha = 1.0
            else:
                joint_alpha = maximum_step / requested_command_step
            q_command = command_anchor + joint_alpha * command_delta
            command_tcp = self._snapshot(q_command).transform_work_tcp
            return q_command, command_tcp, alpha, float(joint_alpha)
        if path_preserving:
            # A delayed physical arm or a transient SDK IK failure is not a
            # reason to cancel the operator's whole Grip session.  Keep the
            # last bounded, valid command so the arm can catch up, then retry
            # from fresh feedback on the next cycle.  If that command is no
            # longer a safe bounded lead, hold measured feedback instead.
            hold_q = previous_command_q.copy()
            hold_lead = float(np.max(np.abs(hold_q - actual.q_rad)))
            hold_tcp = self._snapshot(hold_q).transform_work_tcp
            actual_target_distance = float(
                np.linalg.norm(
                    np.asarray(requested_tcp, dtype=float)[:3, 3]
                    - actual.transform_work_tcp[:3, 3]
                )
            )
            hold_target_distance = float(
                np.linalg.norm(
                    np.asarray(requested_tcp, dtype=float)[:3, 3]
                    - hold_tcp[:3, 3]
                )
            )
            hold_is_bounded = (
                hold_lead <= maximum_feedback_lead + 1.0e-12
                and np.all(hold_q >= self.config.joint_soft_lower_rad)
                and np.all(hold_q <= self.config.joint_soft_upper_rad)
                and hold_target_distance <= actual_target_distance + 1.0e-6
            )
            if not hold_is_bounded:
                hold_q = actual.q_rad.copy()
                hold_tcp = actual.transform_work_tcp.copy()
            self.logger.write(
                "cartesian_path_wait",
                reason=last_rejection,
                q_actual_rad=actual.q_rad,
                q_hold_rad=hold_q,
                maximum_feedback_lead_rad=maximum_feedback_lead,
            )
            return hold_q, hold_tcp, 0.0, 0.0
        raise PositionTargetUnavailable(
            f"no valid RM65 IK step after {attempt + 1} attempts; "
            f"last rejection={last_rejection}"
        )

    def _solve_pose_step(
        self, actual, requested_tcp, previous_command_q, *,
        elbow_branch_sign=0, joint_speed_cap_deg_s=None, assisted_wrist=False,
    ):
        """Bound six-axis motion and verify the commanded TCP.

        Continuous pose mode shortens the Cartesian step before solving IK.
        Assisted wrist mode first tries a bounded full-goal joint step, but
        accepts it only after checking its TCP against the fixed grasp point
        and orientation path. Otherwise it uses the same Cartesian search.
        """
        previous = np.asarray(previous_command_q, dtype=float)
        lead_limit = np.full(6, np.deg2rad(self.GRIP_SYNC_MAX_FEEDBACK_LEAD_DEG))
        if assisted_wrist:
            # Match the commissioned wrist lag budget; proximal compensation
            # retains the tighter five-degree feedback bound.
            lead_limit[3:] = np.deg2rad(self.config.wrist_max_target_lag_deg)
        command_lead_limit = lead_limit - np.deg2rad(self.POSE_COMMAND_LEAD_MARGIN_DEG)
        if (previous.shape != (6,) or not np.all(np.isfinite(previous))
                or np.any(previous < self.config.joint_soft_lower_rad)
                or np.any(previous > self.config.joint_soft_upper_rad)):
            raise TeleopSafetyError("invalid previous pose command")
        feedback_lead = np.abs(previous - actual.q_rad)
        if np.any(feedback_lead > lead_limit + 1e-9):
            joint = int(np.argmax(feedback_lead - lead_limit))
            self.logger.write(
                "pose_feedback_lead_exceeded", joint=joint + 1,
                lead_deg=float(np.rad2deg(feedback_lead[joint])),
                limit_deg=float(np.rad2deg(lead_limit[joint])),
                q_actual_rad=actual.q_rad, q_last_command_rad=previous,
            )
            raise PositionTargetUnavailable(
                f"pose feedback fell behind the bounded command: J{joint + 1} "
                f"{np.rad2deg(feedback_lead[joint]):.3f} deg"
            )
        start = self._snapshot(previous).transform_work_tcp
        period = self._command_budget_period()
        self._translation_rate_period_s = period
        speed = self.config.max_joint_speed_deg_s
        if joint_speed_cap_deg_s is not None:
            speed = min(speed, joint_speed_cap_deg_s)
        maximum_step = np.deg2rad(speed) * period
        if maximum_step <= 1e-12:
            return previous.copy(), start, 0.0, 0.0
        if assisted_wrist:
            shortcut = self._try_wrist_ik_step(
                actual, requested_tcp, previous, start, maximum_step,
                command_lead_limit, elbow_branch_sign,
            )
            if shortcut is not None:
                self.logger.write("wrist_ik_solver", path="verified_joint_step", alpha=shortcut[2])
                return shortcut
        distance = np.linalg.norm(requested_tcp[:3, 3] - start[:3, 3])
        angle = rotation_angle_rad(requested_tcp[:3, :3], start[:3, :3])
        alpha = min(1.0, .01 / max(distance, 1e-12), maximum_step / max(angle, 1e-12))
        search_started = time.monotonic()
        last_rejection = "no IK attempt"
        waiting_for_feedback = False
        for attempt in range(8):
            if attempt and time.monotonic() - search_started >= .02:
                break
            target = interpolate_transform(start, requested_tcp, alpha)
            if assisted_wrist:
                # Correct the full fixed-position residual on every attempt.
                # Shrinking it along with rotation can trap a pitch command
                # at the 1 mm boundary left by the previous numerical IK.
                target[:3, 3] = requested_tcp[:3, 3]
            flange = base_flange_from_work_tcp(
                target, self._transform_base_work, self._transform_flange_tcp
            )
            try:
                solution = self._solve_verified_ik(previous, transform_to_pose6(flange), previous)
            except (InverseKinematicsError, ValueError, TeleopSafetyError) as exc:
                # SDK communication/status errors propagate; only an invalid
                # or infeasible pose is eligible for automatic re-anchoring.
                last_rejection = f"IK: {exc}"
                alpha *= .5
                continue
            if np.shape(solution) != (6,) or not np.all(np.isfinite(solution)):
                raise TeleopSafetyError("pose IK returned invalid joints")
            if np.any(solution < self.config.joint_soft_lower_rad) or np.any(
                solution > self.config.joint_soft_upper_rad
            ):
                last_rejection = "joint soft limit"
                alpha *= .5
                continue
            previous_sign = self._elbow_branch_sign(previous[2])
            solution_sign = self._elbow_branch_sign(solution[2])
            if elbow_branch_sign and previous_sign == elbow_branch_sign and (
                solution_sign and solution_sign != elbow_branch_sign
            ):
                last_rejection = "elbow branch discontinuity"
                alpha *= .5
                continue
            joint_step = float(np.max(np.abs(solution - previous)))
            if joint_step > maximum_step + 1e-9:
                last_rejection = "joint-rate step"
                alpha *= min(.5, .9 * maximum_step / joint_step)
                continue
            # Keep an encoder margin below the unchanged hard stop threshold.
            # Subdivide the TCP path; never clip joints independently, which
            # could move the grasp point during an in-place rotation.
            if np.any(np.abs(solution - actual.q_rad) > command_lead_limit + 1e-9):
                waiting_for_feedback = True
                last_rejection = "feedback lead"
                alpha *= .5
                continue
            reached = self._snapshot(solution).transform_work_tcp
            if assisted_wrist and np.linalg.norm(reached[:3, 3] - requested_tcp[:3, 3]) > self.ABSOLUTE_POSITION_TOLERANCE_M:
                last_rejection = "wrist TCP anchor residual"
                alpha *= .5
                continue
            # Check the actual TCP, not just the flange: this catches tool
            # coordinate mistakes that would otherwise move the grasp point.
            if (np.linalg.norm(reached[:3, 3] - target[:3, 3]) > self.ABSOLUTE_POSITION_TOLERANCE_M
                    or np.degrees(rotation_angle_rad(reached[:3, :3], target[:3, :3]))
                    > self.ABSOLUTE_ORIENTATION_TOLERANCE_DEG):
                last_rejection = "TCP FK residual"
                alpha *= .5
                continue
            if assisted_wrist:
                self.logger.write("wrist_ik_solver", path="cartesian_ik", attempts=attempt + 1, alpha=alpha)
            return solution, reached, alpha, 1.0
        if waiting_for_feedback:
            self.logger.write("pose_feedback_wait", reason=last_rejection)
            return previous.copy(), start, 0.0, 0.0
        raise PositionTargetUnavailable(f"no valid whole-arm pose step: {last_rejection}")

    def _try_wrist_ik_step(self, actual, target, previous, start, maximum_step,
                         command_lead_limit, elbow_branch_sign):
        """Try one full IK and a bounded step, accepting only verified TCP motion.

        This avoids repeated tiny Cartesian solves when the wrist can turn
        normally. If the bounded joint step bends the TCP path too far, the
        caller falls back to Cartesian subdivision with complete IK solves.
        """
        flange = base_flange_from_work_tcp(target, self._transform_base_work, self._transform_flange_tcp)
        try:
            goal = self._solve_verified_ik(previous, transform_to_pose6(flange), previous)
        except (InverseKinematicsError, ValueError, TeleopSafetyError):
            return None
        if np.shape(goal) != (6,) or not np.all(np.isfinite(goal)):
            raise TeleopSafetyError("assisted wrist IK returned invalid joints")
        if np.any(goal < self.config.joint_soft_lower_rad) or np.any(goal > self.config.joint_soft_upper_rad):
            return None
        before, after = self._elbow_branch_sign(previous[2]), self._elbow_branch_sign(goal[2])
        if elbow_branch_sign and before == elbow_branch_sign and after and after != elbow_branch_sign:
            return None
        delta = goal - previous
        fraction = min(1., maximum_step / max(float(np.max(np.abs(delta))), 1e-12))
        for offset, movement, bound in zip(previous - actual.q_rad, delta, command_lead_limit):
            if movement > 0:
                fraction = min(fraction, (bound - offset) / movement)
            elif movement < 0:
                fraction = min(fraction, (-bound - offset) / movement)
        if fraction <= 1e-9:
            return None
        command = previous + fraction * delta
        if np.any(np.abs(command - actual.q_rad) > command_lead_limit + 1e-9):
            return None
        reached = self._snapshot(command).transform_work_tcp
        expected = interpolate_transform(start, target, fraction)
        # Wrist control has a fixed TCP position. Verify against that anchor,
        # not just the previous command, so residuals cannot accumulate.
        if (np.linalg.norm(reached[:3, 3] - target[:3, 3]) > self.ABSOLUTE_POSITION_TOLERANCE_M
                or np.degrees(rotation_angle_rad(reached[:3, :3], expected[:3, :3]))
                > self.ABSOLUTE_ORIENTATION_TOLERANCE_DEG):
            return None
        return command, reached, fraction, 1.0

    def _solve_verified_ik(self, primary_seed, flange_pose, angle_reference):
        """Solve IK and reject SDK answers that do not reproduce the target.

        Near a straight J3 posture the RealMan numerical solver can return the
        current singular solution even for a displaced target.  Left-arm
        position-priority mode retries the exact same TCP pose from bent-elbow
        seeds, then selects the nearest verified solution.  Seeds only guide
        IK; they are never sent to the physical arm.
        """
        primary_seed = np.asarray(primary_seed, dtype=float)
        angle_reference = np.asarray(angle_reference, dtype=float)
        target_pose = np.asarray(flange_pose, dtype=float)
        if not self.config.position_priority_ik:
            solution = self.arm.inverse_kinematics(primary_seed, target_pose)
            solution = nearest_equivalent_angles(solution, angle_reference)
            if self.absolute_control != "off" or self.wrist_control in ("pose", "auto-ik"):
                achieved = pose6_to_transform(self.arm.forward_pose(solution))
                target = pose6_to_transform(target_pose)
                if (np.linalg.norm(achieved[:3, 3] - target[:3, 3])
                        > self.ABSOLUTE_POSITION_TOLERANCE_M
                        or np.degrees(rotation_angle_rad(achieved[:3, :3], target[:3, :3]))
                        > self.ABSOLUTE_ORIENTATION_TOLERANCE_DEG):
                    raise TeleopSafetyError("pose intermediate IK failed FK verification")
            return solution
        seeds = [primary_seed.copy()]
        for elbow_deg in (30.0, -30.0, 60.0, -60.0):
            seed = primary_seed.copy()
            elbow_delta = np.deg2rad(elbow_deg) - seed[2]
            seed[2] += elbow_delta
            # Counter-bias shoulder/wrist pitch to give the numerical IK
            # a bent, approximately pose-compatible initial branch.
            seed[1] -= 0.5 * elbow_delta
            seed[3] -= 0.5 * elbow_delta
            seed = np.clip(
                seed,
                np.deg2rad(self.config.joint_lower_deg),
                np.deg2rad(self.config.joint_upper_deg),
            )
            seeds.append(seed)

        candidates = []
        errors = []
        target_transform = pose6_to_transform(target_pose)
        for seed in seeds:
            try:
                solution = self.arm.inverse_kinematics(seed, target_pose)
                solution = nearest_equivalent_angles(solution, angle_reference)
                achieved_transform = pose6_to_transform(
                    self.arm.forward_pose(solution)
                )
                position_error = float(
                    np.linalg.norm(
                        achieved_transform[:3, 3] - target_transform[:3, 3]
                    )
                )
                orientation_error_deg = float(
                    np.degrees(
                        rotation_angle_rad(
                            target_transform[:3, :3],
                            achieved_transform[:3, :3],
                        )
                    )
                )
                if position_error > self.IK_POSITION_TOLERANCE_M:
                    errors.append(f"position residual {position_error * 1000.0:.1f}mm")
                    continue
                if orientation_error_deg > self.IK_ORIENTATION_TOLERANCE_DEG:
                    errors.append(
                        f"orientation residual {orientation_error_deg:.1f}deg"
                    )
                    continue
                score = float(np.linalg.norm(solution - angle_reference))
                candidates.append((score, solution))
                # A verified solution already reaches the exact requested
                # TCP. Return immediately so normal cycles still use one IK
                # call; alternate seeds are only tried after a bad/failed
                # primary solution near the straight-elbow singularity.
                return solution
            except Exception as exc:
                errors.append(str(exc))

        if not candidates:
            detail = errors[-1] if errors else "no IK candidate"
            raise TeleopSafetyError(f"no verified IK solution: {detail}")
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _elbow_branch_sign(self, joint_3_rad: float) -> int:
        threshold = np.deg2rad(self.ELBOW_BRANCH_HYSTERESIS_DEG)
        if joint_3_rad > threshold:
            return 1
        if joint_3_rad < -threshold:
            return -1
        return 0

    def _log_cycle(
        self,
        session,
        sample,
        actual,
        requested_tcp,
        command_tcp,
        q_command,
        phase,
        **extra,
    ) -> None:
        now = time.monotonic()
        actual_delta_work = (
            actual.transform_work_tcp[:3, 3]
            - session.start_snapshot.transform_work_tcp[:3, 3]
        )
        actual_delta_operator = (
            self._rotation_work_operator.T @ actual_delta_work
        )
        target_gap_m = float(
            np.linalg.norm(
                requested_tcp[:3, 3] - actual.transform_work_tcp[:3, 3]
            )
        )
        orientation_change_deg = float(
            np.degrees(
                rotation_angle_rad(
                    session.start_snapshot.transform_work_tcp[:3, :3],
                    actual.transform_work_tcp[:3, :3],
                )
            )
        )
        joint_alpha = extra.get("joint_alpha")
        if self.wrist_control in ("pose", "auto-ik"):
            extra["pose_orientation_error_deg"] = float(np.degrees(rotation_angle_rad(
                requested_tcp[:3, :3], actual.transform_work_tcp[:3, :3]
            )))
        session.maximum_target_gap_m = max(
            session.maximum_target_gap_m,
            target_gap_m,
        )
        session.maximum_orientation_change_deg = max(
            session.maximum_orientation_change_deg,
            orientation_change_deg,
        )
        if joint_alpha is not None:
            session.minimum_joint_alpha = min(
                session.minimum_joint_alpha,
                float(joint_alpha),
            )
        if phase == "command_sent" and now >= session.next_diagnostic_monotonic:
            session.next_diagnostic_monotonic = (
                now + self.TERMINAL_DIAGNOSTIC_PERIOD_S
            )
            raw_delta_xr_mm = (
                sample.position - session.start_sample.position
            ) * 1000.0
            requested_operator_mm = (
                session.last_requested_delta_operator_m * 1000.0
            )
            actual_operator_mm = actual_delta_operator * 1000.0
            q_actual_deg = np.rad2deg(actual.q_rad)
            q_command_deg = (
                None if q_command is None else np.rad2deg(q_command)
            )
            frame_age_ms = max(
                0.0,
                (time.monotonic() - sample.received_monotonic) * 1000.0,
            )
            limit_state = []
            if extra.get("target_limited"):
                limit_state.append("TARGET_LAG")
            if joint_alpha is not None and float(joint_alpha) < 0.999:
                limit_state.append("JOINT_RATE")
            if not limit_state:
                limit_state.append("FREE")
            print(
                f"[{self._terminal_timestamp()}][{self.hand_label} RM65 TRACK] "
                f"id={session.session_id} phase={phase} "
                f"raw_xr_mm={self._format_vector(raw_delta_xr_mm, 1)} "
                f"target_operator_mm={self._format_vector(requested_operator_mm, 1)} "
                f"actual_operator_mm={self._format_vector(actual_operator_mm, 1)} "
                f"target_gap_mm={target_gap_m * 1000.0:.1f} "
                f"orientation_deg={orientation_change_deg:.2f} "
                f"joint_alpha={self._format_optional_float(joint_alpha, 3)} "
                f"limit={'+'.join(limit_state)} frame_age_ms={frame_age_ms:.1f} "
                f"q_actual_deg={self._format_vector(q_actual_deg, 1)} "
                f"q_cmd_deg={self._format_optional_vector(q_command_deg, 1)}",
                flush=True,
            )
        self.logger.write(
            "control_cycle",
            session_id=session.session_id,
            phase=phase,
            xr_position=sample.position,
            grip=sample.grip,
            trigger=sample.trigger,
            timestamp_ns=sample.timestamp_ns,
            requested_delta_operator_m=session.last_requested_delta_operator_m,
            commanded_delta_work_m=session.last_commanded_delta_work_m,
            requested_transform_work_tcp=requested_tcp,
            command_transform_work_tcp=command_tcp,
            q_actual_rad=actual.q_rad,
            q_command_rad=q_command,
            actual_transform_work_tcp=actual.transform_work_tcp,
            **extra,
        )

    @property
    def _position_recovery_enabled(self) -> bool:
        return (
            self.position_fault_recovery == "reanchor"
            and (self.config.controller_name == "right_controller"
                 or self._left_reference_work_rotation is not None)
            and self.grip_sync == "persistent"
            and self.absolute_control == "off"
        )

    def _check_position_progress(self, session, actual, target) -> None:
        """Detect successful SDK calls that leave the TCP stationary."""
        if self.wrist_control == "pose":
            self._check_pose_progress(session, actual, target)
            return
        now = time.monotonic()
        position = actual.transform_work_tcp[:3, 3]
        gap = float(np.linalg.norm(target[:3, 3] - position))
        if session.follow_checked_at is None or gap <= .01:
            session.follow_progress_at = now
            session.follow_progress_m = 0.0
        elif session.follow_last_position is not None:
            session.follow_progress_m += float(
                np.linalg.norm(session.follow_last_target - session.follow_last_position)
                - np.linalg.norm(session.follow_last_target - position)
            )
            if session.follow_progress_m >= .001:
                session.follow_progress_at = now
                session.follow_progress_m = 0.0
        session.follow_checked_at = now
        session.follow_last_position = position.copy()
        session.follow_last_target = target[:3, 3].copy()
        if now - session.follow_progress_at > 1.5:
            raise PositionTargetUnavailable(f"position following stalled with {gap * 1000:.1f} mm remaining")

    def _check_pose_progress(self, session, actual, target):
        """Include rotation-only stalls; moving the hand is not robot progress."""
        now = time.monotonic()
        measured = actual.transform_work_tcp
        gap = np.linalg.norm(target[:3, 3] - measured[:3, 3])
        angle = np.degrees(rotation_angle_rad(target[:3, :3], measured[:3, :3]))
        if session.pose_progress_at is None or (gap <= .005 and angle <= 2.0):
            session.pose_progress_at = now
            session.pose_progress = 0.0
        elif session.pose_last_actual is not None:
            def remaining(pose):
                return (np.linalg.norm(session.pose_last_target[:3, 3] - pose[:3, 3])
                        + .1 * rotation_angle_rad(session.pose_last_target[:3, :3], pose[:3, :3]))
            session.pose_progress += remaining(session.pose_last_actual) - remaining(measured)
            if session.pose_progress >= .001:
                session.pose_progress_at = now
                session.pose_progress = 0.0
        session.pose_last_actual = measured.copy()
        session.pose_last_target = target.copy()
        if now - session.pose_progress_at > 1.5:
            raise PositionTargetUnavailable(
                f"pose following stalled with {gap * 1000:.1f} mm / {angle:.1f} deg remaining"
            )

    def _queue_position_reanchor(self, detail: str) -> None:
        """Stop once, then recover from current feedback, never an old target."""
        with self._lock:
            self._tracking_resume_reason = None
            self._release_seen = False
            self._position_reanchor_pending = False
            active = self._session is not None
        if active:
            self._stop_session("position_reanchor", require_release=True)
        else:
            status = self.arm.slow_stop()
            if status != 0:
                raise TeleopSafetyError(f"position reanchor slow-stop failed: {status}")
        with self._lock:
            if self._shutdown:
                return
            self._position_reanchor_pending = True
            self._position_reanchor_epoch += 1
            self._position_reanchor_after = time.monotonic() + .15
            self._position_reanchor_since = None
            self._position_reanchor_q = None
            self._position_reanchor_hand = None
        self.logger.write("position_reanchor_wait", reason=detail)
        print(f"[{self.hand_label} RM65 REANCHOR WAIT] {detail}. "
              "Stopping, then resetting the hand/TCP reference while Grip stays held.", flush=True)

    def _try_position_reanchor(self, sample: XrSample) -> None:
        now = time.monotonic()
        with self._lock:
            if not self._position_reanchor_pending or now < self._position_reanchor_after:
                return
            epoch = self._position_reanchor_epoch
        try:
            actual = self._snapshot()
        except ArmFeedbackTimeout as exc:
            with self._lock:
                self._position_reanchor_since = None
                self._position_reanchor_after = now + .25
            self.logger.write("position_reanchor_feedback_wait", error=str(exc))
            return
        except ValueError as exc:
            with self._lock:
                self._position_reanchor_pending = False
                self._release_seen = False
            self.logger.write("position_reanchor_rejected", reason=f"invalid feedback: {exc}")
            print(f"[{self.hand_label} RM65 REANCHOR REJECTED] Invalid robot feedback: {exc}", flush=True)
            return
        with self._lock:
            # A watchdog invalidation during the feedback read cancels this sample.
            if (not self._position_reanchor_pending or epoch != self._position_reanchor_epoch
                    or self._shutdown or self._session is not None):
                return
            if (np.shape(actual.q_rad) != (6,) or not np.all(np.isfinite(actual.q_rad))
                    or not np.all(np.isfinite(actual.transform_work_tcp))
                    or np.any(actual.q_rad < self.config.joint_soft_lower_rad)
                    or np.any(actual.q_rad > self.config.joint_soft_upper_rad)):
                self._position_reanchor_pending = False
                self._release_seen = False
                self.logger.write("position_reanchor_rejected", reason="invalid feedback or joint soft limits")
                print(f"[{self.hand_label} RM65 REANCHOR REJECTED] Check robot feedback / joint limits.", flush=True)
                return
            unstable = (
                self._position_reanchor_since is None
                or np.max(np.abs(actual.q_rad - self._position_reanchor_q)) > np.deg2rad(.3)
                or np.linalg.norm(sample.position - self._position_reanchor_hand) > self.config.max_xr_sample_step_m
            )
            self._position_reanchor_hand = sample.position.copy()
            if unstable:
                self._position_reanchor_since = now
                self._position_reanchor_q = actual.q_rad.copy()
                return
            if now - self._position_reanchor_since < .1:
                return
            previous = self._grip_reference
            # Only capture measured position: this call sends no joint command.
            self._start_session(sample, reanchor_snapshot=actual)
            self._position_reanchor_pending = False
            self.logger.write(
                "position_reanchored", hand_position_m=sample.position,
                old_tcp_origin_m=None if previous is None else previous.tcp_position_work_m,
                tcp_origin_m=actual.transform_work_tcp[:3, 3], q_actual_rad=actual.q_rad,
            )
            print(f"[{self.hand_label} RM65 REANCHORED] FOLLOW ready from current hand/TCP position.", flush=True)

    def _finish_xr_pause(self, *, failed: bool) -> None:
        if self._xr_pause_started is None:
            return
        duration = time.monotonic() - self._xr_pause_started
        self._xr_pause_started = None
        self._xr_fresh_since = None
        self._xr_fresh_timestamp = None
        if failed:
            self._xr_pause_failures += 1
            self._xr_requires_release = True
            self._tracking_resume_reason = None
            self._release_seen = False
        self.logger.write("bounded_xr_recovery", duration_s=duration, failed=failed)

    def _expire_xr_pause(self, now: float) -> None:
        if (self._xr_pause_started is not None
                and now - self._xr_pause_started >= self.xr_recovery_window_s):
            self._finish_xr_pause(failed=True)
            print(f"[{self.hand_label} RM65 XR] 恢复超过 1 秒；请松开再按 Grip。", flush=True)

    @property
    def feedback_fault_active(self) -> bool:
        with self._lock:
            return self._feedback_fault_active

    @property
    def xr_recovery_status(self) -> dict:
        with self._lock:
            started = self._xr_pause_started
            return {
                "xr_recovery_pending": started is not None,
                "xr_recovery_elapsed_s": 0.0 if started is None else time.monotonic() - started,
                "xr_recovery_failures": self._xr_pause_failures,
            }

    def _bounded_xr_resume_ready(self, sample: XrSample) -> bool:
        if self.xr_recovery_window_s is None:
            return True
        with self._lock:
            self._expire_xr_pause(time.monotonic())
            if self._xr_pause_stopping:
                return False
            if self._xr_requires_release:
                if sample.grip >= self.GRIP_RELEASE:
                    return False
                self._xr_requires_release = False
            if self._xr_pause_started is None:
                return True
            if sample.grip < self.GRIP_RELEASE:
                self._finish_xr_pause(failed=False)
                self._tracking_resume_reason = None
                return True
            # Require advancing, fresh packets over 50 ms, with Trigger/buttons
            # released. Do not dispatch a held gripper chord on reconnection.
            if sample.trigger >= self.GRIP_RELEASE or sample.primary_button or sample.secondary_button:
                self._xr_fresh_since = None
                return False
            if self._xr_fresh_since is None:
                self._xr_fresh_since = time.monotonic()
                self._xr_fresh_timestamp = sample.timestamp_ns
                return False
            return (sample.timestamp_ns > self._xr_fresh_timestamp
                    and time.monotonic() - self._xr_fresh_since >= .05)

    def handle_stream_stale(self, reason: str = "xr_stale") -> None:
        with self._lock:
            home_active = self._home_active
        self._handle_tracking_fault(reason)
        if home_active:
            self._cancel_home(reason, require_release=True)

    def _handle_tracking_fault(self, reason: str, detail: str | None = None) -> None:
        with self._lock:
            if self.gripper_chord:
                self._gripper_latch.reset()
            if self._position_reanchor_pending:
                self._position_reanchor_epoch += 1
                self._position_reanchor_since = None
                self._position_reanchor_after = time.monotonic() + .15
                return
            active = self._session is not None
            if reason == "robot_feedback_timeout" and self._left_reference_work_rotation is not None:
                self._feedback_fault_active = True
                self._xr_requires_release = True
            pending = self._tracking_resume_reason is not None
            auto_resume = self.resume_tracking_faults and (active or pending)
            bounded = self.xr_recovery_window_s is not None
            if bounded:
                self._expire_xr_pause(time.monotonic())
                is_xr_gap = reason in ("xr_stale", "xr_watchdog_timeout")
                auto_resume = (auto_resume and is_xr_gap
                               and (active or self._xr_pause_started is not None))
                if auto_resume and self._xr_pause_started is None:
                    self._xr_pause_started = time.monotonic()
                if not is_xr_gap:
                    self._finish_xr_pause(failed=True)
                self._xr_fresh_since = None
                if active:
                    self._xr_pause_stopping += 1
            if auto_resume:
                self._tracking_resume_reason = reason
                if active:
                    self._next_tracking_resume_attempt = time.monotonic() + 0.25
            elif not self._home_active:
                self._release_seen = False
        if active:
            try:
                self._stop_session(reason, require_release=not auto_resume)
            finally:
                if bounded:
                    with self._lock:
                        self._xr_pause_stopping -= 1
        if auto_resume and active:
            source = "RM65 FEEDBACK" if reason == "robot_feedback_timeout" else "RM65 XR"
            print(
                f"[{self.hand_label} {source} PAUSED] {detail or reason}. "
                "Holding position; a fresh robot read will re-anchor and resume "
                "while Grip remains held.",
                flush=True,
            )
        if active:
            self.logger.write(
                "tracking_fault",
                reason=reason,
                detail=detail,
                active=True,
                auto_resume=auto_resume,
            )

    def _stop_session(self, reason: str, *, require_release: bool) -> None:
        with self._lock:
            session = self._session
            if session is None:
                if require_release:
                    self._release_seen = False
                return
            self._generation += 1
            self._session = None
            session.cancel_event.set()
            self._release_seen = not require_release
            self._last_stop_reason = reason

        set_streaming = getattr(self.arm, "set_streaming_motion_active", None)
        defer_health_until_stopped = self._left_reference_work_rotation is not None
        if callable(set_streaming) and not defer_health_until_stopped:
            set_streaming(False)
        # A cancelled session is not yet a stationary robot. In left collection,
        # keep slow periodic health queries deferred while measuring braking.
        # The next idle/start read still performs the overdue health check.
        stop_status = self.arm.slow_stop()
        try:
            try:
                final_snapshot = self._snapshot_after_stop()
            finally:
                if callable(set_streaming) and defer_health_until_stopped:
                    with self._lock:
                        if self._session is None:
                            set_streaming(False)
            actual_delta_work = (
                final_snapshot.transform_work_tcp[:3, 3]
                - session.start_snapshot.transform_work_tcp[:3, 3]
            )
            actual_delta_operator = (
                self._rotation_work_operator.T @ actual_delta_work
            )
            summary = evaluate_session(
                session_id=session.session_id,
                stop_reason=reason,
                requested_delta_work_m=session.last_requested_delta_operator_m,
                actual_delta_work_m=actual_delta_operator,
                start_rotation_work_tcp=(
                    session.start_snapshot.transform_work_tcp[:3, :3]
                ),
                actual_rotation_work_tcp=(final_snapshot.transform_work_tcp[:3, :3]),
            )
            if self.absolute_control != "off" or self.wrist_control in ("pose", "auto-ik"):
                requested = session.requested_transform_work_tcp
                position_error = float(np.linalg.norm(
                    requested[:3, 3] - final_snapshot.transform_work_tcp[:3, 3]))
                angle_error = float(np.degrees(rotation_angle_rad(
                    requested[:3, :3], final_snapshot.transform_work_tcp[:3, :3])))
                label = "pose" if self.wrist_control in ("pose", "auto-ik") else "absolute"
                checks = {
                    f"{label}_position_within_tolerance": position_error <= self.ABSOLUTE_POSITION_TOLERANCE_M,
                    f"{label}_orientation_within_tolerance": angle_error <= self.ABSOLUTE_ORIENTATION_TOLERANCE_DEG,
                }
                summary = replace(summary, distance_error_ratio=None, cross_axis_limit_m=None,
                    result="TARGET_REACHED" if all(checks.values()) else "STOPPED", checks=checks)
                if self.print_session_summary:
                    print(f"[{self.hand_label} RM65 {label.upper()} END] id={session.session_id} "
                          f"reason={reason} "
                          f"target_error_mm={position_error * 1000:.2f} "
                          f"target_error_deg={angle_error:.2f} result={summary.result}", flush=True)
            self.last_summary = summary
            if self.print_session_summary and self.absolute_control == "off" and self.wrist_control not in ("pose", "auto-ik"):
                requested_norm_m = float(
                    np.linalg.norm(summary.requested_delta_work_m)
                )
                actual_norm_m = float(np.linalg.norm(summary.actual_delta_work_m))
                completion_percent = (
                    0.0
                    if requested_norm_m <= 1.0e-9
                    else 100.0 * actual_norm_m / requested_norm_m
                )
                duration_s = max(0.0, time.monotonic() - session.started_monotonic)
                if summary.requested_axis in ("IDLE", "INVALID"):
                    calibration_state = "NO_MOTION"
                elif summary.checks.get("requested_motion_is_single_axis", False):
                    calibration_state = "VALID_SINGLE_AXIS"
                else:
                    calibration_state = "INVALID_MULTI_AXIS"
                print(
                    f"[{self._terminal_timestamp()}][{self.hand_label} RM65 SESSION END] "
                    f"id={summary.session_id} reason={summary.stop_reason} "
                    f"duration_s={duration_s:.2f} "
                    f"request={summary.requested_axis} actual={summary.actual_axis} "
                    f"requested_mm={self._format_vector(summary.requested_delta_work_m * 1000.0, 1)} "
                    f"actual_mm={self._format_vector(summary.actual_delta_work_m * 1000.0, 1)} "
                    f"completion={completion_percent:.1f}% "
                    f"orientation_final_deg={summary.orientation_change_deg:.2f} "
                    f"orientation_max_deg={session.maximum_orientation_change_deg:.2f} "
                    f"target_gap_max_mm={session.maximum_target_gap_m * 1000.0:.1f} "
                    f"joint_alpha_min={session.minimum_joint_alpha:.3f} "
                    f"result={summary.result} calibration={calibration_state}",
                    flush=True,
                )
                if calibration_state == "INVALID_MULTI_AXIS":
                    print(
                        f"[{self._terminal_timestamp()}][{self.hand_label} "
                        "CALIBRATION INVALID] multi-axis hand movement; "
                        "do not use this session to change coordinate signs.",
                        flush=True,
                    )
            self.logger.write(
                "session_end",
                stop_status=stop_status,
                summary=summary.as_dict(),
                commanded_delta_work_m=session.last_commanded_delta_work_m,
                actual_delta_raw_416table_m=actual_delta_work,
                q_final_rad=final_snapshot.q_rad,
                transform_work_tcp_final=final_snapshot.transform_work_tcp,
            )
        except Exception as exc:
            self.logger.write(
                "session_end_feedback_error",
                session_id=session.session_id,
                reason=reason,
                stop_status=stop_status,
                error=f"{type(exc).__name__}: {exc}",
            )
        get_stop_details = getattr(self.arm, "last_stop_diagnostics", None)
        if callable(get_stop_details):
            self.logger.write("arm_stop_result", reason=reason, status=stop_status,
                              details=get_stop_details())
        if stop_status != 0:
            raise TeleopSafetyError(
                f"RM65 slow-stop failed during {reason}: status={stop_status}"
            )

    @staticmethod
    def _terminal_timestamp() -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    @staticmethod
    def _format_vector(values, decimals: int) -> str:
        vector = np.asarray(values, dtype=float)
        return "(" + ",".join(f"{value:+.{decimals}f}" for value in vector) + ")"

    @staticmethod
    def _format_optional_vector(values, decimals: int) -> str:
        if values is None:
            return "-"
        return CleanTeleopController._format_vector(values, decimals)

    @staticmethod
    def _format_optional_float(value, decimals: int) -> str:
        if value is None:
            return "-"
        return f"{float(value):.{decimals}f}"

    def _snapshot_after_stop(self, timeout_s: float = 0.5) -> RobotSnapshot:
        """Read the final TCP after encoder feedback becomes stationary."""
        deadline = time.monotonic() + float(timeout_s)
        previous = self._snapshot()
        stable_count = 0
        left_collection = self._left_reference_work_rotation is not None
        stable_anchor = previous.q_rad.copy()
        stable_since = time.monotonic()
        while time.monotonic() < deadline:
            # Feedback polling during braking is independent of dataset FPS.
            # Preserve a >=100 ms stationary window, not merely two faster reads.
            time.sleep(0.02 if left_collection else 0.05)
            current = self._snapshot()
            if left_collection:
                if np.max(np.abs(np.rad2deg(current.q_rad - stable_anchor))) > 0.02:
                    stable_anchor = current.q_rad.copy()
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= 0.1:
                    return current
            else:
                max_delta_deg = float(
                    np.max(np.abs(np.rad2deg(current.q_rad - previous.q_rad)))
                )
                if max_delta_deg <= 0.02:
                    stable_count += 1
                    if stable_count >= 2:
                        return current
                else:
                    stable_count = 0
            previous = current
        return previous

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            active = self._session is not None
        if active:
            self._stop_session("program_exit", require_release=True)
        if self.home_active:
            self._cancel_home("program_exit", require_release=True)
        # Program exit always sends a stop even if no Grip session is active.
        set_streaming = getattr(self.arm, "set_streaming_motion_active", None)
        if callable(set_streaming):
            set_streaming(False)
        status = self.arm.slow_stop()
        gripper_error = None
        if self.gripper is not None:
            try:
                self.gripper.shutdown()
            except Exception as exc:
                gripper_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"[{self.hand_label} GRIPPER SHUTDOWN ERROR] {exc}",
                    flush=True,
                )
        with self._lock:
            self._shutdown = True
            self._release_seen = False
        self.logger.write(
            "shutdown",
            slow_stop_status=status,
            gripper_shutdown_error=gripper_error,
        )
        if status != 0:
            raise TeleopSafetyError(
                f"RM65 slow-stop failed during shutdown: status={status}"
            )


class TeleopRuntime:
    """Configured-rate loop plus an independent XR-staleness stop watchdog."""

    def __init__(self, controller: CleanTeleopController, sample_buffer, *, xr_reader=None):
        self.controller = controller
        self.sample_buffer = sample_buffer
        self.xr_reader = xr_reader
        self.stop_event = threading.Event()
        self._watchdog_error = None
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="rm65-clean-safety-watchdog",
            daemon=True,
        )

    def _watchdog_loop(self) -> None:
        reported_stale = False
        while not self.stop_event.is_set():
            fresh, ever_fresh, age_s, error = self.sample_buffer.status()
            stale = (
                ever_fresh
                and not fresh
                and age_s > self.controller.config.xr_stale_timeout_s
            )
            if stale:
                try:
                    self.controller.handle_stream_stale("xr_watchdog_timeout")
                except Exception as exc:
                    self._watchdog_error = exc
                    self.stop_event.set()
                    try:
                        self.controller.logger.write(
                            "watchdog_stop_error",
                            error=f"{type(exc).__name__}: {exc}",
                            xr_error=error,
                        )
                    except Exception:
                        pass
                    return
                if not reported_stale:
                    reported_stale = True
                    try:
                        self.controller.logger.write(
                            "xr_watchdog_diagnostic",
                            xr_age_s=age_s,
                            reader_error=error,
                            buffer=self.sample_buffer.diagnostics(),
                            sdk=(self.xr_reader.diagnostics()
                                 if self.xr_reader is not None else None),
                        )
                    except Exception:
                        # Diagnostic collection must never prevent the stop.
                        pass
            else:
                reported_stale = False
            self.stop_event.wait(0.01)

    def _log_runtime_error(self, event, error):
        # Logging must not replace the fault that caused motion to stop.
        try:
            self.controller.logger.write(
                event,
                error=f"{type(error).__name__}: {error}",
                traceback="".join(traceback.format_exception(
                    type(error), error, error.__traceback__
                )),
            )
        except Exception:
            pass

    def run(self) -> None:
        self._watchdog.start()
        next_tick = time.monotonic()
        runtime_error = None
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                sample = self.sample_buffer.latest(now)
                if sample is not None:
                    self.controller.step(sample)
                next_tick += self.controller.config.period_s
                delay = next_tick - time.monotonic()
                if delay > 0.0:
                    self.stop_event.wait(delay)
                else:
                    # Never run catch-up command bursts after a delayed SDK call.
                    next_tick = time.monotonic()
            if self._watchdog_error is not None:
                raise TeleopSafetyError(
                    f"XR watchdog could not confirm arm stop: {self._watchdog_error}"
                ) from self._watchdog_error
        except BaseException as exc:
            runtime_error = exc
            self._log_runtime_error("control_runtime_error", exc)
            raise
        finally:
            self.stop_event.set()
            self._watchdog.join(timeout=1.0)
            try:
                self.controller.shutdown()
            except BaseException as exc:
                self._log_runtime_error("control_shutdown_error", exc)
                if runtime_error is None:
                    raise
                print(
                    f"[RM65 STOP NOT CONFIRMED] {exc}; "
                    "check the robot locally. Original control error preserved.",
                    flush=True,
                )


class DualTeleopRuntime:
    """Run two independent clean controllers from synchronized XR buffers."""

    def __init__(self, controllers: dict[str, CleanTeleopController], buffers):
        if set(controllers) != {"right", "left"} or set(buffers) != {
            "right",
            "left",
        }:
            raise ValueError("dual runtime requires exactly left and right")
        self.controllers = dict(controllers)
        self.buffers = dict(buffers)
        self.stop_event = threading.Event()
        self._error_lock = threading.Lock()
        self._errors: list[tuple[str, BaseException]] = []
        self._workers = {
            hand: threading.Thread(
                target=self._control_loop,
                args=(hand,),
                name=f"rm65-clean-{hand}-control",
                daemon=True,
            )
            for hand in ("right", "left")
        }
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="rm65-clean-dual-safety-watchdog",
            daemon=True,
        )

    def _record_error(self, hand: str, exc: BaseException) -> None:
        with self._error_lock:
            self._errors.append((hand, exc))
        self.controllers[hand].logger.write(
            "dual_runtime_error",
            hand=hand,
            error=f"{type(exc).__name__}: {exc}",
        )
        self.stop_event.set()

    def _control_loop(self, hand: str) -> None:
        controller = self.controllers[hand]
        buffer = self.buffers[hand]
        next_tick = time.monotonic()
        try:
            while not self.stop_event.is_set():
                sample = buffer.latest(time.monotonic())
                if sample is not None:
                    controller.step(sample)
                next_tick += controller.config.period_s
                delay = next_tick - time.monotonic()
                if delay > 0.0:
                    self.stop_event.wait(delay)
                else:
                    next_tick = time.monotonic()
        except Exception as exc:
            self._record_error(hand, exc)

    def _watchdog_loop(self) -> None:
        while not self.stop_event.is_set():
            for hand in ("right", "left"):
                controller = self.controllers[hand]
                fresh, ever_fresh, age_s, error = self.buffers[hand].status()
                if (
                    ever_fresh
                    and not fresh
                    and age_s > controller.config.xr_stale_timeout_s
                ):
                    try:
                        controller.handle_stream_stale("xr_watchdog_timeout")
                    except Exception as exc:
                        controller.logger.write(
                            "watchdog_stop_error",
                            hand=hand,
                            error=f"{type(exc).__name__}: {exc}",
                            xr_error=error,
                        )
                        self._record_error(hand, exc)
                        return
            self.stop_event.wait(0.01)

    def run(self) -> None:
        self._watchdog.start()
        for worker in self._workers.values():
            worker.start()
        shutdown_errors: list[tuple[str, BaseException]] = []
        try:
            while not self.stop_event.wait(0.1):
                pass
        finally:
            self.stop_event.set()
            for worker in self._workers.values():
                worker.join(timeout=1.0)
            self._watchdog.join(timeout=1.0)
            for hand, controller in self.controllers.items():
                try:
                    controller.shutdown()
                except Exception as exc:
                    shutdown_errors.append((hand, exc))
        with self._error_lock:
            errors = list(self._errors)
        if errors:
            hand, exc = errors[0]
            raise TeleopSafetyError(
                f"{hand} arm dual runtime failed: {type(exc).__name__}: {exc}"
            ) from exc
        if shutdown_errors:
            hand, exc = shutdown_errors[0]
            raise TeleopSafetyError(
                f"{hand} arm shutdown failed: {type(exc).__name__}: {exc}"
            ) from exc
