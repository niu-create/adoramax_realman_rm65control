from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


class TrackingArm:
    MAX_RECORDED_ACTION_SPEED_RAD_S = np.deg2rad(180.0)
    MAX_RECORDED_ACTION_STEP_RAD = np.deg2rad(9.0)

    """Transparent arm proxy that exposes feedback and successful commands."""

    def __init__(self, arm):
        self._arm = arm
        self._lock = threading.RLock()
        self._io_lock = threading.RLock()
        self._feedback = None
        self._feedback_time = None
        self._action = None
        self._action_time = None
        self._hold_action_at_feedback = True
        self._recorded_action = None
        self._recorded_action_time = None

    def __getattr__(self, name):
        return getattr(self._arm, name)

    def read_joint_positions(self):
        with self._io_lock:
            value = np.asarray(self._arm.read_joint_positions(), dtype=float)
        now = time.monotonic()
        with self._lock:
            self._feedback = value.copy()
            self._feedback_time = now
        return value

    def send_joint_target(self, positions, **kwargs):
        with self._io_lock:
            status = self._arm.send_joint_target(positions, **kwargs)
        if int(status) == 0:
            with self._lock:
                self._action = np.asarray(positions, dtype=float).copy()
                self._action_time = time.monotonic()
                self._hold_action_at_feedback = False
        return status

    def synchronize_for_episode(
        self,
        *,
        samples=3,
        interval_s=0.05,
        max_step_deg=2.0,
    ):
        """Read real encoders and seed state/action before an episode starts."""
        samples = int(samples)
        if samples < 2:
            raise ValueError("episode synchronization requires at least 2 samples")
        values = []
        for index in range(samples):
            values.append(self.read_joint_positions())
            if index + 1 < samples:
                time.sleep(float(interval_s))
        stacked = np.asarray(values, dtype=float)
        max_step_rad = float(np.max(np.abs(np.diff(stacked, axis=0))))
        max_step_deg_seen = float(np.rad2deg(max_step_rad))
        if max_step_deg_seen > float(max_step_deg):
            raise RuntimeError(
                "RM65 is still moving before episode start: "
                f"max_encoder_step_deg={max_step_deg_seen:.3f}, "
                f"limit_deg={float(max_step_deg):.3f}"
            )
        value = stacked[-1].copy()
        now = time.monotonic()
        with self._lock:
            self._feedback = value.copy()
            self._feedback_time = now
            self._action = value.copy()
            self._action_time = now
            self._hold_action_at_feedback = True
            self._recorded_action = value.copy()
            self._recorded_action_time = now
        return {
            "joint_positions": value.tolist(),
            "max_encoder_step_deg": max_step_deg_seen,
            "samples": samples,
        }

    def get_collection_snapshot(
        self,
        *,
        hold_action_at_feedback=False,
        smooth_recorded_action=False,
    ):
        with self._lock:
            if hold_action_at_feedback:
                self._hold_action_at_feedback = True
            # Collection observes a zero-order hold of the last feedback read
            # by the verified controller.  Refresh only the sample timestamp;
            # never issue a second SDK read from the collection thread.
            now = time.monotonic()
            feedback_time = (
                now if self._feedback is not None else None
            )
            use_feedback = (
                self._hold_action_at_feedback and self._feedback is not None
            )
            action_positions = self._feedback if use_feedback else self._action
            action_source = (
                "feedback_hold_inactive" if use_feedback
                else "successful_send_joint_target" if self._action is not None
                else "feedback_hold"
            )
            if smooth_recorded_action and action_positions is not None:
                target = np.asarray(action_positions, dtype=float)
                if self._recorded_action is None:
                    seed = self._feedback if self._feedback is not None else target
                    self._recorded_action = np.asarray(seed, dtype=float).copy()
                else:
                    elapsed = max(
                        0.0,
                        now - (
                            self._recorded_action_time
                            if self._recorded_action_time is not None
                            else now
                        ),
                    )
                    limit = min(
                        self.MAX_RECORDED_ACTION_STEP_RAD,
                        self.MAX_RECORDED_ACTION_SPEED_RAD_S * elapsed,
                    )
                    if limit > 0.0:
                        delta = np.clip(
                            target - self._recorded_action, -limit, limit
                        )
                        self._recorded_action = self._recorded_action + delta
                self._recorded_action_time = now
                action_positions = self._recorded_action
                action_source = f"smoothed_{action_source}"
            return {
                "feedback_positions": None if self._feedback is None else self._feedback.copy(),
                "feedback_time": feedback_time,
                # Preserve the real read time separately from the zero-order
                # hold timestamp used by the existing frame builder.
                "feedback_read_time": self._feedback_time,
                "sent_positions": None if self._action is None else self._action.copy(),
                "sent_time": self._action_time,
                "action_positions": None if action_positions is None else action_positions.copy(),
                "action_source": action_source,
            }


class TrackingGripper:
    def __init__(self, gripper, *, initialized_status=None):
        self._gripper = gripper
        self._lock = threading.RLock()
        now = time.monotonic()
        initialized_status = initialized_status or {}
        self._position = initialized_status.get("position")
        self._feedback_time = now if self._position is not None else None
        self._target = initialized_status.get(
            "target_position", getattr(gripper, "POSITION_CLOSED", 0)
        )
        self._target_time = now

    def __getattr__(self, name):
        return getattr(self._gripper, name)

    def get_motion_status(self):
        status = self._gripper.get_motion_status()
        with self._lock:
            self._position = float(status["position"])
            self._feedback_time = time.monotonic()
        return status

    def synchronize_for_episode(self):
        """Align the recorded target with measured jaw position without motion."""
        status = self.get_motion_status()
        now = time.monotonic()
        with self._lock:
            self._target = float(self._position)
            self._target_time = now
            self._feedback_time = now
        return dict(status)

    def command_open_state(self, target_open):
        result = self._gripper.command_open_state(target_open)
        with self._lock:
            self._target = float(result["target_position"])
            self._target_time = time.monotonic()
            # The verified command completed successfully.  Use that confirmed
            # discrete open/close state without adding a Modbus feedback poll.
            self._position = self._target
            self._feedback_time = self._target_time
        return result

    def get_collection_snapshot(self):
        with self._lock:
            return {"position": self._position, "feedback_time": self._feedback_time,
                    "target_position": self._target, "target_time": self._target_time}


class CachedGripperSampler:
    """LeRobot sampler interface with no background Modbus transactions."""

    def __init__(self, grippers):
        self.grippers = dict(grippers)

    def start(self):
        return None

    def close(self):
        return None

    def prepare_episode(self):
        """Synchronize measured and commanded jaw state before recording."""
        for gripper in self.grippers.values():
            synchronize = getattr(gripper, "synchronize_for_episode", None)
            if callable(synchronize):
                synchronize()
            else:
                gripper.get_motion_status()

    def snapshots(self):
        result = {}
        for name, gripper in self.grippers.items():
            snapshot = dict(gripper.get_collection_snapshot())
            if snapshot.get("position") is not None:
                snapshot["feedback_time"] = time.monotonic()
            native_open = float(getattr(gripper, "POSITION_OPEN", 100))
            if native_open <= 0:
                raise ValueError(f"{name} POSITION_OPEN must be positive")
            snapshot["position"] = (
                None if snapshot.get("position") is None
                else float(snapshot["position"]) * 100.0 / native_open
            )
            snapshot["target_position"] = (
                None if snapshot.get("target_position") is None
                else float(snapshot["target_position"]) * 100.0 / native_open
            )
            snapshot["sampler_error"] = None
            snapshot["sampler_error_time"] = None
            result[name] = snapshot
        return result


class _ControllerView:
    def __init__(self, owner):
        self.owner = owner
        self.rm65 = owner.arm

    def get_collection_snapshot(self):
        home_active = bool(self.owner.controller.home_active)
        snapshot = self.rm65.get_collection_snapshot(
            hold_action_at_feedback=not bool(self.owner.controller.active),
            smooth_recorded_action=True,
        )
        snapshot["home_active"] = home_active
        return snapshot

    def prepare_collection_episode(self):
        return self.rm65.synchronize_for_episode()

    def get_teleop_status(self, _name=None):
        sample = self.owner.sample_buffer.latest(time.monotonic())
        if sample is None:
            return {"reason": "no_xr_packet", "grip": 0.0,
                    "grip_pressed": False, "grip_held": False, "stream_fresh": False,
                    "follow_ready": False, "axis_xy": (0.0, 0.0)}
        home_active = bool(self.owner.controller.home_active)
        active = bool(self.owner.controller.active)
        if home_active:
            reason = "home_active"
        elif active:
            reason = "active"
        elif self.owner.controller.release_seen:
            reason = "grip_released"
        else:
            reason = "release_required"
        return {
            "active": active,
            "home_active": home_active,
            "reason": reason,
            "grip": float(sample.grip),
            "grip_pressed": bool(sample.grip > self.owner.controller.GRIP_ACTIVATE),
            "grip_held": bool(sample.grip >= self.owner.controller.GRIP_RELEASE),
            "stream_fresh": bool(sample.stream_fresh),
            "follow_ready": bool(sample.stream_fresh and
                                 self.owner.controller.collection_follow_ready),
            "axis_xy": tuple(
                float(value) for value in getattr(sample, "axis_xy", (0.0, 0.0))
            ),
        }


class VerifiedRightRM65CollectionController:
    """Commissioned right-arm stack with selectable response and collection adapters."""

    def __init__(
        self,
        *,
        log_dir,
        arm_response="original",
        forward_reach_gain=1.0,
        right_reach_gain=1.0,
        vertical_reach_gain=1.0,
        grip_sync_speed_deg_s=60.0,
        position_fault_recovery="release",
        wrist_control="buttons",
        ray_j4_control=False,
        ray_j4_sign=1.0,
        tracking_frame="head_relative",
        resume_tracking_faults=True,
        grip_sync="reanchor",
        absolute_control="off",
        head_camera_calibration=None,
        allow_uncalibrated_absolute=False,
        grip_sync_orientation_relax_deg=0.0,
    ):
        if arm_response not in ("original", "faster", "responsive"):
            raise ValueError("arm_response must be original, faster or responsive")
        if wrist_control not in ("buttons", "auto", "pose", "auto-ik"):
            raise ValueError("wrist_control must be buttons, auto, pose or auto-ik")
        if wrist_control in ("pose", "auto-ik") and (absolute_control != "off"
                or (ray_j4_control and wrist_control == "pose") or grip_sync_orientation_relax_deg != 0.0):
            raise ValueError("pose following cannot combine absolute, individual-joint or relaxed-orientation control")
        if allow_uncalibrated_absolute and absolute_control == "off":
            raise ValueError("uncalibrated absolute test requires pose or wrist control")
        if absolute_control != "off" and not head_camera_calibration and not allow_uncalibrated_absolute:
            raise ValueError("absolute control requires calibration or explicit uncalibrated test mode")
        if allow_uncalibrated_absolute and head_camera_calibration:
            raise ValueError("choose calibrated or uncalibrated absolute control")
        from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.accessories import AGGripper
        from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.config import load_config
        from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.controller import CleanTeleopController, TeleopRuntime
        if absolute_control != "off" and head_camera_calibration:
            from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.absolute_retarget import HeadCameraCalibration
            HeadCameraCalibration.load(head_camera_calibration)
        from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.hardware import RealManArm
        from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.logging_jsonl import JsonlLogger
        from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.xr import FreshSampleBuffer, XrReaderThread, XrSdkClient

        vendor = Path(__file__).resolve().parents[1] / "vendor" / "xr_robot1_right_rm65"
        self.config = load_config(vendor / "config" / "rm65_192_168_1_19.json")
        self.logger = JsonlLogger(Path(log_dir) / f"right_teleop_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
        raw_arm = raw_gripper = None
        stage = "arm_connect"
        self.logger.write("collection_startup", stage=stage, arm_ip=self.config.arm_ip)
        try:
            raw_arm = RealManArm(self.config.arm_ip, self.config.arm_port,
                                 collision_level=self.config.collision_level,
                                 enable_self_collision=self.config.enable_self_collision,
                                 normalize_kinematics=absolute_control != "off" or wrist_control in ("pose", "auto-ik"))
            self.arm = TrackingArm(raw_arm)
            stage = "gripper_initialize"
            self.logger.write("collection_startup", stage=stage,
                              device_id=self.config.gripper_device_id,
                              baudrate=AGGripper.DEFAULT_BAUDRATE)
            raw_gripper = AGGripper(self.arm, device_id=self.config.gripper_device_id,
                                    force_percent=self.config.gripper_force_percent,
                                    timeout_units=self.config.gripper_modbus_timeout_units,
                                    tool_voltage=self.config.gripper_tool_voltage)
            initialized_status = raw_gripper.initialize(
                timeout_s=self.config.gripper_initialization_timeout_s,
                verify_communication=True,
            )
        except BaseException as exc:
            # __init__ has not returned, so the caller cannot close this object.
            # Closing Modbus and the SDK handle here also prevents stale sessions.
            cleanup_errors = []
            for resource, method in ((raw_gripper, "shutdown"), (raw_arm, "close")):
                if resource is not None:
                    try:
                        getattr(resource, method)()
                    except Exception as cleanup_exc:
                        cleanup_errors.append(f"{method}: {cleanup_exc}")
            try:
                self.logger.write("collection_startup_failed", stage=stage,
                                  error_type=type(exc).__name__, error=str(exc),
                                  cleanup_errors=cleanup_errors)
            except Exception as log_exc:
                print(f"[STARTUP LOG ERROR] {log_exc}", flush=True)
            finally:
                try:
                    self.logger.close()
                except Exception as log_exc:
                    print(f"[STARTUP LOG CLOSE ERROR] {log_exc}", flush=True)
            raise
        self.gripper = TrackingGripper(
            raw_gripper, initialized_status=initialized_status
        )
        self.controller = CleanTeleopController(
            self.config, self.arm, self.logger, gripper=self.gripper, gripper_chord=True,
            translation_response=arm_response,
            forward_reach_gain=forward_reach_gain,
            right_reach_gain=right_reach_gain,
            vertical_reach_gain=vertical_reach_gain,
            grip_sync_speed_deg_s=grip_sync_speed_deg_s,
            position_fault_recovery=position_fault_recovery,
            wrist_control=wrist_control,
            ray_j4_control=ray_j4_control,
            ray_j4_sign=ray_j4_sign,
            tracking_frame=tracking_frame,
            resume_tracking_faults=resume_tracking_faults,
            grip_sync=grip_sync,
            absolute_control=absolute_control,
            head_camera_calibration=head_camera_calibration,
            allow_uncalibrated_absolute=allow_uncalibrated_absolute,
            grip_sync_orientation_relax_deg=grip_sync_orientation_relax_deg,
        )
        self.logger.write(
            "collection_response_config",
            arm_response=arm_response,
            forward_reach_gain=forward_reach_gain,
            forward_reach_knee_m=self.controller.FORWARD_REACH_KNEE_M,
            right_reach_gain=right_reach_gain,
            vertical_reach_gain=vertical_reach_gain,
            grip_sync_speed_deg_s=grip_sync_speed_deg_s,
            grip_sync_max_distance_m=self.controller.GRIP_SYNC_MAX_DISTANCE_M,
            grip_sync_max_feedback_lead_deg=self.controller.GRIP_SYNC_MAX_FEEDBACK_LEAD_DEG,
            position_fault_recovery=position_fault_recovery,
            wrist_control=wrist_control,
            ray_j4_control=bool(ray_j4_control),
            gripper_control="Grip+Trigger_toggle_release_trigger_to_rearm",
            pose_orientation_gain=1.0 if wrist_control == "pose" else None,
            wrist_ik_position="hold_for_j4_j6" if wrist_control == "auto-ik" else None,
            wrist_j5_execution="direct_joint" if wrist_control == "auto-ik" else None,
            wrist_j6_effective_sign=-self.config.j6_controller_sign if wrist_control == "auto-ik" else None,
            wrist_ik_orientation_mapping="virtual_joint_fk" if wrist_control == "auto-ik" else None,
            wrist_ik_feedback_lead_deg=(
                [5., 5., 5., self.config.wrist_max_target_lag_deg,
                 self.config.wrist_max_target_lag_deg, self.config.wrist_max_target_lag_deg]
                if wrist_control == "auto-ik" else None
            ),
            pose_command_lead_cap_deg=(
                self.controller.GRIP_SYNC_MAX_FEEDBACK_LEAD_DEG
                - self.controller.POSE_COMMAND_LEAD_MARGIN_DEG
                if wrist_control == "pose" else None
            ),
            ray_j4_sign=float(ray_j4_sign),
            tracking_frame=tracking_frame,
            resume_tracking_faults=bool(resume_tracking_faults),
            grip_sync=grip_sync,
            absolute_control=absolute_control,
            head_camera_calibration=str(head_camera_calibration) if head_camera_calibration else None,
            allow_uncalibrated_absolute=bool(allow_uncalibrated_absolute),
            grip_sync_orientation_relax_deg=float(grip_sync_orientation_relax_deg),
            max_joint_speed_deg_s=self.config.max_joint_speed_deg_s,
            max_translation_step_deg=float(np.rad2deg(self.config.max_joint_step_rad))
                * self.controller.response_budget_cycles,
            wrist_scale=None if wrist_control == "pose" else self.controller.effective_wrist_scale,
            wrist_max_speed_deg_s=min(self.config.wrist_max_speed_deg_s,
                                      self.config.max_joint_speed_deg_s),
            wrist_command_anchor="previous_command" if arm_response == "responsive" or wrist_control in ("auto", "pose", "auto-ik") else "feedback",
            follow=False,
        )
        self.controller.initialize()
        self.sample_buffer = FreshSampleBuffer(self.config.xr_stale_timeout_s)
        self.reader = XrReaderThread(XrSdkClient(self.config.controller_name, self.config.grip_name, self.config.trigger_name), self.sample_buffer)
        self.runtime = TeleopRuntime(self.controller, self.sample_buffer, xr_reader=self.reader)
        self.controllers = {"right_arm": _ControllerView(self)}
        self.grippers = {"right_arm": self.gripper}
        self._error = None
        self._closed = False

    def start_xr_collection(self): self.reader.start()
    def run(self):
        try: self.runtime.run()
        except BaseException as exc:
            self._error = exc
            raise
    def request_stop(self): self.runtime.stop_event.set()
    def raise_if_failed(self):
        if self._error is not None: raise RuntimeError(f"verified right-arm control failed: {self._error}") from self._error
    def close(self):
        if self._closed: return
        self._closed = True
        self.request_stop()
        try: self.reader.close()
        finally:
            try: self.arm.close()
            finally: self.logger.close()
