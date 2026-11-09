"""Validated configuration loading for one fixed RM65 installation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import FrameSpec
from .transforms import finite_vector, validate_rotation


@dataclass(frozen=True)
class TeleopConfig:
    arm_ip: str
    arm_port: int
    controller_name: str
    grip_name: str
    trigger_name: str
    work_frame: FrameSpec
    tool_frame: FrameSpec
    rotation_base_xr: np.ndarray
    rotation_work_from_operator: np.ndarray
    translation_xr_axis_signs: np.ndarray
    position_scale: float
    teleop_position_gain: float
    control_rate_hz: float
    max_joint_speed_deg_s: float
    max_target_position_lag_m: float
    feedback_anchored_rate_limit: bool
    cartesian_path_rate_limit: bool
    position_priority_ik: bool
    translation_axis_deadband_m: float
    grasp_lift_assist_enabled: bool
    grasp_lift_elbow_trigger_deg: float
    grasp_lift_elbow_release_deg: float
    grasp_lift_retract_m: float
    grasp_lift_activation_m: float
    shoulder_swing_enabled: bool
    shoulder_swing_activation_m: float
    shoulder_swing_elbow_max_deg: float
    shoulder_swing_scale_deg_per_m: float
    shoulder_swing_max_speed_deg_s: float
    home_enabled: bool
    home_button_hold_s: float
    home_speed_percent: int
    home_timeout_s: float
    wrist_joint_control_enabled: bool
    wrist_trigger_threshold: float
    wrist_rotation_deadband_deg: float
    wrist_max_rotation_step_deg: float
    wrist_scale_factor: float
    wrist_max_speed_deg_s: float
    wrist_max_target_lag_deg: float
    j5_controller_axis: int
    j5_controller_sign: float
    j6_controller_axis: int
    j6_controller_sign: float
    gripper_enabled: bool
    gripper_required: bool
    gripper_protocol: str
    gripper_device_id: int
    gripper_force_percent: int
    gripper_tool_voltage: int
    gripper_modbus_timeout_units: int
    gripper_initialization_timeout_s: float
    gripper_button_debounce_s: float
    xr_stale_timeout_s: float
    motion_start_threshold_m: float
    max_xr_sample_step_m: float
    frame_position_tolerance_m: float
    frame_rotation_tolerance_deg: float
    joint_lower_deg: np.ndarray
    joint_upper_deg: np.ndarray
    joint_limit_margin_deg: float
    collision_level: int
    enable_self_collision: bool
    calibration_rms_error_deg: float

    def __post_init__(self):
        object.__setattr__(
            self,
            "rotation_base_xr",
            validate_rotation(self.rotation_base_xr),
        )
        object.__setattr__(
            self,
            "rotation_work_from_operator",
            validate_rotation(self.rotation_work_from_operator),
        )
        object.__setattr__(
            self,
            "translation_xr_axis_signs",
            finite_vector(
                self.translation_xr_axis_signs,
                3,
                "XR translation axis signs",
            ),
        )
        if not np.all(np.isclose(np.abs(self.translation_xr_axis_signs), 1.0)):
            raise ValueError("translation_xr_axis_signs must contain only -1 or +1")
        object.__setattr__(
            self,
            "joint_lower_deg",
            finite_vector(self.joint_lower_deg, 6, "joint lower limits"),
        )
        object.__setattr__(
            self,
            "joint_upper_deg",
            finite_vector(self.joint_upper_deg, 6, "joint upper limits"),
        )
        positive = {
            "arm_port": self.arm_port,
            "position_scale": self.position_scale,
            "teleop_position_gain": self.teleop_position_gain,
            "control_rate_hz": self.control_rate_hz,
            "max_joint_speed_deg_s": self.max_joint_speed_deg_s,
            "home_button_hold_s": self.home_button_hold_s,
            "home_timeout_s": self.home_timeout_s,
            "wrist_rotation_deadband_deg": self.wrist_rotation_deadband_deg,
            "wrist_max_rotation_step_deg": self.wrist_max_rotation_step_deg,
            "wrist_scale_factor": self.wrist_scale_factor,
            "wrist_max_speed_deg_s": self.wrist_max_speed_deg_s,
            "wrist_max_target_lag_deg": self.wrist_max_target_lag_deg,
            "gripper_initialization_timeout_s": self.gripper_initialization_timeout_s,
            "xr_stale_timeout_s": self.xr_stale_timeout_s,
            "max_xr_sample_step_m": self.max_xr_sample_step_m,
            "frame_position_tolerance_m": self.frame_position_tolerance_m,
            "frame_rotation_tolerance_deg": self.frame_rotation_tolerance_deg,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if (
            not np.isfinite(self.max_target_position_lag_m)
            or self.max_target_position_lag_m < 0.0
        ):
            raise ValueError("max_target_position_lag_m must be nonnegative")
        if not 0.0 <= self.motion_start_threshold_m <= 0.005:
            raise ValueError("motion_start_threshold_m must be in [0, 0.005]")
        if not 0.0 <= self.translation_axis_deadband_m <= 0.02:
            raise ValueError("translation_axis_deadband_m must be in [0, 0.02]")
        if not 0.0 <= self.grasp_lift_elbow_trigger_deg < 30.0:
            raise ValueError("grasp_lift_elbow_trigger_deg must be in [0, 30)")
        if not (
            self.grasp_lift_elbow_trigger_deg
            < self.grasp_lift_elbow_release_deg
            <= 45.0
        ):
            raise ValueError(
                "grasp_lift_elbow_release_deg must be above the trigger and <= 45"
            )
        if not 0.0 < self.grasp_lift_retract_m <= 0.10:
            raise ValueError("grasp_lift_retract_m must be in (0, 0.10]")
        if not 0.0 < self.grasp_lift_activation_m <= 0.05:
            raise ValueError("grasp_lift_activation_m must be in (0, 0.05]")
        if not 0.0 < self.shoulder_swing_activation_m <= 0.05:
            raise ValueError("shoulder_swing_activation_m must be in (0, 0.05]")
        if not 0.0 <= self.shoulder_swing_elbow_max_deg <= 30.0:
            raise ValueError("shoulder_swing_elbow_max_deg must be in [0, 30]")
        if not 1.0 <= self.shoulder_swing_scale_deg_per_m <= 360.0:
            raise ValueError("shoulder_swing_scale_deg_per_m must be in [1, 360]")
        if not 1.0 <= self.shoulder_swing_max_speed_deg_s <= 180.0:
            raise ValueError("shoulder_swing_max_speed_deg_s must be in [1, 180]")
        if not 1 <= int(self.home_speed_percent) <= 10:
            raise ValueError("home_speed_percent must be in [1, 10]")
        if not 0.0 < self.wrist_trigger_threshold <= 1.0:
            raise ValueError("wrist_trigger_threshold must be in (0, 1]")
        if self.wrist_rotation_deadband_deg >= self.wrist_max_rotation_step_deg:
            raise ValueError("wrist deadband must be below the maximum rotation step")
        for name, axis in (
            ("j5_controller_axis", self.j5_controller_axis),
            ("j6_controller_axis", self.j6_controller_axis),
        ):
            if int(axis) not in (0, 1, 2):
                raise ValueError(f"{name} must be 0, 1, or 2")
        for name, sign in (
            ("j5_controller_sign", self.j5_controller_sign),
            ("j6_controller_sign", self.j6_controller_sign),
        ):
            if not np.isfinite(sign) or abs(float(sign)) < 1.0e-12:
                raise ValueError(f"{name} must be finite and nonzero")
        if self.gripper_protocol not in {"ag", "mini_9c"}:
            raise ValueError(
                "gripper_protocol must be 'ag' or 'mini_9c'"
            )
        if not 1 <= int(self.gripper_device_id) <= 247:
            raise ValueError("gripper_device_id must be in [1, 247]")
        minimum_force = 20 if self.gripper_protocol == "ag" else 0
        if not minimum_force <= int(self.gripper_force_percent) <= 100:
            raise ValueError(
                f"{self.gripper_protocol} gripper_force_percent must be in "
                f"[{minimum_force}, 100]"
            )
        if int(self.gripper_tool_voltage) not in (0, 24):
            raise ValueError("gripper_tool_voltage must be 0 or 24")
        if int(self.gripper_modbus_timeout_units) <= 0:
            raise ValueError("gripper_modbus_timeout_units must be positive")
        if (
            not np.isfinite(self.gripper_button_debounce_s)
            or self.gripper_button_debounce_s < 0
        ):
            raise ValueError("gripper_button_debounce_s must be nonnegative")
        if not 1 <= int(self.collision_level) <= 8:
            raise ValueError("collision_level must be in [1, 8]")
        if np.any(self.joint_lower_deg >= self.joint_upper_deg):
            raise ValueError("joint lower limits must be below upper limits")
        if self.joint_limit_margin_deg < 0.0:
            raise ValueError("joint_limit_margin_deg must be nonnegative")
        if np.any(
            self.joint_lower_deg + self.joint_limit_margin_deg
            >= self.joint_upper_deg - self.joint_limit_margin_deg
        ):
            raise ValueError("joint limit margin leaves no valid range")
        routing = {
            "right_controller": {
                "arm_ip": "192.168.1.19",
                "grip_name": "right_grip",
                "trigger_name": "right_trigger",
            },
            "left_controller": {
                "arm_ip": "192.168.1.18",
                "grip_name": "left_grip",
                "trigger_name": "left_trigger",
            },
        }
        try:
            expected_routing = routing[self.controller_name]
        except KeyError as exc:
            raise ValueError(
                "controller_name must be 'right_controller' or 'left_controller'"
            ) from exc
        fixed_values = {
            "arm_ip": (self.arm_ip, expected_routing["arm_ip"]),
            "arm_port": (self.arm_port, 8080),
            "grip_name": (self.grip_name, expected_routing["grip_name"]),
            "trigger_name": (self.trigger_name, expected_routing["trigger_name"]),
            "work_frame": (self.work_frame.name, "416table"),
            "tool_frame": (self.tool_frame.name, "leboripper"),
        }
        for name, (actual, expected) in fixed_values.items():
            if actual != expected:
                raise ValueError(
                    f"clean requirement baseline fixes {name}={expected!r}, "
                    f"got {actual!r}"
                )
        if not np.isclose(self.position_scale, 1.0, atol=1.0e-12):
            raise ValueError("clean requirement baseline fixes position_scale=1.0")
        if not np.isclose(self.teleop_position_gain, 1.0, atol=1.0e-12):
            raise ValueError("this installation fixes teleop_position_gain=1.0")
        if not np.isclose(self.control_rate_hz, 90.0, atol=1.0e-12):
            raise ValueError("this installation fixes control_rate_hz=90")
        if self.max_joint_speed_deg_s > 180.0:
            raise ValueError("max_joint_speed_deg_s may not exceed 180 deg/s")
        if self.xr_stale_timeout_s > 0.2:
            raise ValueError("xr_stale_timeout_s may not exceed 0.2 seconds")
        expected_axis_alignment = {
            "right_controller": np.diag([-1.0, -1.0, 1.0]),
            "left_controller": np.diag([-1.0, -1.0, 1.0]),
        }[self.controller_name]
        if not np.allclose(
            self.rotation_work_from_operator,
            expected_axis_alignment,
            atol=1.0e-12,
        ):
            raise ValueError(
                "this installation requires the calibrated operator-to-416table "
                f"correction for {self.controller_name}"
            )

    @property
    def period_s(self) -> float:
        return 1.0 / self.control_rate_hz

    @property
    def max_joint_step_rad(self) -> float:
        return float(np.deg2rad(self.max_joint_speed_deg_s) * self.period_s)

    @property
    def wrist_max_joint_step_rad(self) -> float:
        return float(np.deg2rad(self.wrist_max_speed_deg_s) * self.period_s)

    @property
    def joint_soft_lower_rad(self) -> np.ndarray:
        return np.deg2rad(self.joint_lower_deg + self.joint_limit_margin_deg)

    @property
    def joint_soft_upper_rad(self) -> np.ndarray:
        return np.deg2rad(self.joint_upper_deg - self.joint_limit_margin_deg)


def load_config(path: str | Path) -> TeleopConfig:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read configuration {source}: {exc}") from exc
    if payload.get("schema_version") != 2:
        raise ValueError("unsupported configuration schema_version")
    arm = payload["arm"]
    xr = payload["xr"]
    control = payload["control"]
    safety = payload["safety"]
    calibration = payload["calibration"]
    auxiliary = payload["auxiliary_controls"]
    gripper = payload["gripper"]
    return TeleopConfig(
        arm_ip=str(arm["ip"]),
        arm_port=int(arm["port"]),
        controller_name=str(xr["controller"]),
        grip_name=str(xr["grip"]),
        trigger_name=str(xr["trigger"]),
        work_frame=FrameSpec(
            str(arm["work_frame"]["name"]),
            arm["work_frame"]["pose_in_base_m_rad"],
        ),
        tool_frame=FrameSpec(
            str(arm["tool_frame"]["name"]),
            arm["tool_frame"]["pose_from_flange_m_rad"],
        ),
        rotation_base_xr=calibration["rotation_xr_to_rm65_base"],
        rotation_work_from_operator=calibration["rotation_416table_from_operator_axes"],
        translation_xr_axis_signs=calibration.get(
            "translation_xr_axis_signs",
            [1.0, 1.0, 1.0],
        ),
        position_scale=float(control["position_scale"]),
        teleop_position_gain=float(control["teleop_position_gain"]),
        control_rate_hz=float(control["rate_hz"]),
        max_joint_speed_deg_s=float(control["max_joint_speed_deg_s"]),
        max_target_position_lag_m=float(
            control.get("max_target_position_lag_m", 0.0)
        ),
        feedback_anchored_rate_limit=bool(
            control.get("feedback_anchored_rate_limit", False)
        ),
        cartesian_path_rate_limit=bool(
            control.get("cartesian_path_rate_limit", False)
        ),
        position_priority_ik=bool(control.get("position_priority_ik", False)),
        translation_axis_deadband_m=float(
            control.get("translation_axis_deadband_m", 0.0)
        ),
        grasp_lift_assist_enabled=bool(
            control.get("grasp_lift_assist_enabled", False)
        ),
        grasp_lift_elbow_trigger_deg=float(
            control.get("grasp_lift_elbow_trigger_deg", 10.0)
        ),
        grasp_lift_elbow_release_deg=float(
            control.get("grasp_lift_elbow_release_deg", 20.0)
        ),
        grasp_lift_retract_m=float(control.get("grasp_lift_retract_m", 0.05)),
        grasp_lift_activation_m=float(
            control.get("grasp_lift_activation_m", 0.01)
        ),
        shoulder_swing_enabled=bool(
            control.get("shoulder_swing_enabled", False)
        ),
        shoulder_swing_activation_m=float(
            control.get("shoulder_swing_activation_m", 0.01)
        ),
        shoulder_swing_elbow_max_deg=float(
            control.get("shoulder_swing_elbow_max_deg", 10.0)
        ),
        shoulder_swing_scale_deg_per_m=float(
            control.get("shoulder_swing_scale_deg_per_m", 180.0)
        ),
        shoulder_swing_max_speed_deg_s=float(
            control.get("shoulder_swing_max_speed_deg_s", 120.0)
        ),
        home_enabled=bool(auxiliary["home_enabled"]),
        home_button_hold_s=float(auxiliary["home_button_hold_s"]),
        home_speed_percent=int(auxiliary["home_speed_percent"]),
        home_timeout_s=float(auxiliary["home_timeout_s"]),
        wrist_joint_control_enabled=bool(auxiliary["wrist_joint_control_enabled"]),
        wrist_trigger_threshold=float(auxiliary["wrist_trigger_threshold"]),
        wrist_rotation_deadband_deg=float(auxiliary["wrist_rotation_deadband_deg"]),
        wrist_max_rotation_step_deg=float(auxiliary["wrist_max_rotation_step_deg"]),
        wrist_scale_factor=float(auxiliary["wrist_scale_factor"]),
        wrist_max_speed_deg_s=float(auxiliary["wrist_max_speed_deg_s"]),
        wrist_max_target_lag_deg=float(auxiliary["wrist_max_target_lag_deg"]),
        j5_controller_axis=int(auxiliary["j5_controller_axis"]),
        j5_controller_sign=float(auxiliary["j5_controller_sign"]),
        j6_controller_axis=int(auxiliary["j6_controller_axis"]),
        j6_controller_sign=float(auxiliary["j6_controller_sign"]),
        gripper_enabled=bool(gripper["enabled"]),
        gripper_required=bool(gripper["required"]),
        gripper_protocol=str(gripper["protocol"]),
        gripper_device_id=int(gripper["device_id"]),
        gripper_force_percent=int(gripper["force_percent"]),
        gripper_tool_voltage=int(gripper["tool_voltage"]),
        gripper_modbus_timeout_units=int(gripper["modbus_timeout_units"]),
        gripper_initialization_timeout_s=float(gripper["initialization_timeout_s"]),
        gripper_button_debounce_s=float(gripper["button_debounce_s"]),
        xr_stale_timeout_s=float(safety["xr_stale_timeout_s"]),
        motion_start_threshold_m=float(control["motion_start_threshold_m"]),
        max_xr_sample_step_m=float(safety["max_xr_sample_step_m"]),
        frame_position_tolerance_m=float(safety["frame_position_tolerance_m"]),
        frame_rotation_tolerance_deg=float(safety["frame_rotation_tolerance_deg"]),
        joint_lower_deg=safety["joint_lower_deg"],
        joint_upper_deg=safety["joint_upper_deg"],
        joint_limit_margin_deg=float(safety["joint_limit_margin_deg"]),
        collision_level=int(safety["collision_level"]),
        enable_self_collision=bool(safety["enable_self_collision"]),
        calibration_rms_error_deg=float(calibration["rms_angular_error_deg"]),
    )
