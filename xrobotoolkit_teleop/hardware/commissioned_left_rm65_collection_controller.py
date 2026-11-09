"""Left collection using the right controller and an explicit installation map."""

from __future__ import annotations

from dataclasses import replace
import json
import time
from datetime import datetime
from pathlib import Path

from xrobotoolkit_teleop.hardware.right_rm65_collection_controller import (
    TrackingArm,
    TrackingGripper,
)


COMMISSIONED_SOURCE_COMMIT = "56f7eae"
_VENDOR_ROOT = (
    Path(__file__).resolve().parents[1] / "vendor" / "xr_robot1_dual_rm65"
)
_RIGHT_VENDOR_ROOT = _VENDOR_ROOT.with_name("xr_robot1_right_rm65")
LEFT_SPEED_SCALE = 0.85


def load_left_shared_config():
    """Derive left directions from the right mapping and measured table frames.

    Both live 416table/tool frames were read on 2026-10-15 and match these
    installation files. Do not apply install-pose angles again: the work
    frames already express each base relative to the shared table.
    """
    from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.config import load_config
    from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.models import FrameSpec
    from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.transforms import pose6_to_transform

    right = load_config(_RIGHT_VENDOR_ROOT / "config" / "rm65_192_168_1_19.json")
    left = json.loads((_VENDOR_ROOT / "config" / "rm65_192_168_1_18.json").read_text())
    arm, xr = left["arm"], left["xr"]
    work = FrameSpec(arm["work_frame"]["name"], arm["work_frame"]["pose_in_base_m_rad"])
    tool = FrameSpec(arm["tool_frame"]["name"], arm["tool_frame"]["pose_from_flange_m_rad"])
    right_work_rotation = pose6_to_transform(right.work_frame.pose_m_rad)[:3, :3]
    left_work_rotation = pose6_to_transform(work.pose_m_rad)[:3, :3]
    # R_left_work.T @ R_left_xr == R_right_work.T @ R_right_xr.
    # Operator axes and directional reach gains are consequently identical.
    config = replace(
        right, arm_ip=arm["ip"], arm_port=arm["port"],
        controller_name=xr["controller"], grip_name=xr["grip"], trigger_name=xr["trigger"],
        work_frame=work, tool_frame=tool,
        max_joint_speed_deg_s=right.max_joint_speed_deg_s * LEFT_SPEED_SCALE,
        wrist_max_speed_deg_s=right.wrist_max_speed_deg_s * LEFT_SPEED_SCALE,
        rotation_base_xr=left_work_rotation @ right_work_rotation.T @ right.rotation_base_xr,
    )
    return config, right_work_rotation


class _LeftArmCollectionView:
    """Expose the commissioned left controller to the collection runtime."""

    def __init__(self, owner):
        self.owner = owner
        self.rm65 = owner.arm

    def get_collection_snapshot(self):
        active = bool(self.owner.controller.active)
        snapshot = self.rm65.get_collection_snapshot(
            hold_action_at_feedback=not active,
            smooth_recorded_action=True,
        )
        snapshot["home_active"] = bool(self.owner.controller.home_active)
        snapshot["commissioned_source_commit"] = COMMISSIONED_SOURCE_COMMIT
        snapshot["teleop_source"] = "shared_current_right_controller_left_installation"
        return snapshot

    def prepare_collection_episode(self):
        return self.rm65.synchronize_for_episode()

    def get_teleop_status(self, _name=None):
        sample = self.owner.sample_buffer.latest(time.monotonic())
        if sample is None:
            return {
                "active": False,
                "home_active": bool(self.owner.controller.home_active),
                "reason": "no_xr_packet",
                "grip": 0.0,
                "grip_pressed": False,
                "grip_held": False,
                "stream_fresh": False,
                "follow_ready": False,
            }
        # Recovery can finish between collection ticks. Read the gate fields
        # together so a resumed session is not mistaken for release_required.
        with self.owner.controller._lock:
            active = bool(self.owner.controller.active)
            home_active = bool(self.owner.controller.home_active)
            recovery = self.owner.controller.xr_recovery_status
            release_seen = self.owner.controller.release_seen
            follow_ready = self.owner.controller.collection_follow_ready
            feedback_fault = self.owner.controller.feedback_fault_active
        if home_active:
            reason = "home_active"
        elif feedback_fault:
            reason = "robot_feedback_timeout"
        elif recovery["xr_recovery_pending"]:
            reason = "waiting_xr_recovery"
        elif not sample.stream_fresh:
            reason = "xr_stale"
        elif active:
            reason = "active"
        elif release_seen:
            reason = "grip_released"
        else:
            reason = "release_required"
        return {
            **recovery,
            "active": active,
            "home_active": home_active,
            "reason": reason,
            "grip": float(sample.grip),
            "grip_pressed": bool(sample.grip > self.owner.controller.GRIP_ACTIVATE),
            "grip_held": bool(sample.grip >= self.owner.controller.GRIP_RELEASE),
            "stream_fresh": bool(sample.stream_fresh),
            # The shared controller may still be synchronizing after Grip.
            "follow_ready": bool(sample.stream_fresh and follow_ready),
            "axis_xy": tuple(float(value) for value in getattr(sample, "axis_xy", (0.0, 0.0))),
        }


class CommissionedLeftRM65CollectionController:
    """Run the .18 left arm through the shared right-arm control pipeline."""

    def __init__(self, *, log_dir, forward_reach_gain=2.0, arm_response="responsive",
                 resume_tracking_faults=True, right_reach_gain=1.5, vertical_reach_gain=1.5,
                 wrist_control="auto", ray_j4_control=False, ray_j4_sign=1.0,
                 tracking_frame="head_relative", grip_sync="reanchor",
                 grip_sync_speed_deg_s=90.0, position_fault_recovery="release"):
        from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.accessories import (
            AGGripper,
        )
        from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.controller import (
            CleanTeleopController,
            TeleopRuntime,
        )
        from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.hardware import (
            RealManArm,
        )
        from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.logging_jsonl import (
            JsonlLogger,
        )
        from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.xr import (
            FreshSampleBuffer,
            XrReaderThread,
            XrSdkClient,
        )

        self._RealManArm = RealManArm
        self._closed = False
        self._error = None
        self._runtime_finished = False
        self._sdk_initialized = False
        self.config = None
        self.logger = None
        self.arm = None
        self.gripper = None
        self._raw_gripper = None
        self.controller = None
        self.sample_buffer = None
        self.reader = None
        self.runtime = None
        self.controllers = {}
        self.grippers = {}

        stage = "load_config"
        try:
            self.config, reference_work_rotation = load_left_shared_config()
            self._validate_route()
            self.logger = JsonlLogger(
                Path(log_dir)
                / f"left_teleop_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
            )
            stage = "arm_connect"
            self.logger.write("collection_startup", stage=stage, arm_ip=self.config.arm_ip)
            self._sdk_initialized = True
            raw_arm = RealManArm(
                self.config.arm_ip,
                self.config.arm_port,
                collision_level=self.config.collision_level,
                enable_self_collision=self.config.enable_self_collision,
                normalize_kinematics=True,
                idle_health_read_attempts=3,
                retry_stop_receive_timeout=True,
                initialize_sdk=True,
            )
            self.arm = TrackingArm(raw_arm)
            stage = "gripper_initialize"
            self.logger.write("collection_startup", stage=stage)
            self._raw_gripper = AGGripper(
                self.arm,
                device_id=self.config.gripper_device_id,
                force_percent=self.config.gripper_force_percent,
                timeout_units=self.config.gripper_modbus_timeout_units,
                tool_voltage=self.config.gripper_tool_voltage,
            )
            initialized = self._raw_gripper.initialize(
                timeout_s=self.config.gripper_initialization_timeout_s,
                verify_communication=True,
                retry_startup_writes=True,
            )
            self.gripper = TrackingGripper(
                self._raw_gripper,
                initialized_status=initialized,
            )
            self.gripper._initialized = True
            stage = "controller_initialize"
            self.controller = CleanTeleopController(
                self.config,
                self.arm,
                self.logger,
                gripper=self.gripper,
                forward_reach_gain=forward_reach_gain,
                right_reach_gain=right_reach_gain,
                vertical_reach_gain=vertical_reach_gain,
                translation_response=arm_response,
                resume_tracking_faults=resume_tracking_faults,
                xr_recovery_window_s=1.0 if resume_tracking_faults else None,
                gripper_chord=True,
                wrist_control=wrist_control,
                ray_j4_control=ray_j4_control,
                ray_j4_sign=ray_j4_sign,
                tracking_frame=tracking_frame,
                grip_sync=grip_sync,
                grip_sync_speed_deg_s=grip_sync_speed_deg_s,
                position_fault_recovery=position_fault_recovery,
                left_reference_work_rotation=reference_work_rotation,
            )
            self.controller.initialize()
            stage = "xr_initialize"
            self.sample_buffer = FreshSampleBuffer(
                self.config.xr_stale_timeout_s,
            )
            self.reader = XrReaderThread(
                XrSdkClient(
                    self.config.controller_name,
                    self.config.grip_name,
                    self.config.trigger_name,
                ),
                self.sample_buffer,
            )
            self.runtime = TeleopRuntime(self.controller, self.sample_buffer, xr_reader=self.reader)
            self.controllers = {"left_arm": _LeftArmCollectionView(self)}
            self.grippers = {"left_arm": self.gripper}
        except BaseException as exc:
            if self.logger is not None:
                try:
                    self.logger.write("collection_startup_failed", stage=stage,
                                      error_type=type(exc).__name__, error=str(exc))
                except Exception as log_exc:
                    print(f"[STARTUP LOG ERROR] {log_exc}", flush=True)
            try:
                self.close()
            except Exception as cleanup_exc:
                print(f"[STARTUP CLEANUP ERROR] {cleanup_exc}", flush=True)
            raise

    def _validate_route(self):
        if self.config.arm_ip != "192.168.1.18":
            raise ValueError("left collection must route RM65 192.168.1.18")
        if self.config.controller_name != "left_controller":
            raise ValueError("left collection must route left_controller")

    def start_xr_collection(self):
        self.reader.start()

    def run(self):
        try:
            self.runtime.run()
        except BaseException as exc:
            self._error = exc
            raise
        finally:
            self._runtime_finished = True

    def request_stop(self):
        if self.runtime is not None:
            self.runtime.stop_event.set()

    def raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError(
                f"commissioned left RM65 control failed: {self._error}"
            ) from self._error
        if self._runtime_finished:
            raise RuntimeError("commissioned left RM65 control exited; collection cannot continue")

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.request_stop()
        errors = []
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception as exc:
                errors.append(exc)
        if self.controller is not None:
            try:
                # Idempotent; also retries shutdown if runtime cleanup raised.
                self.controller.shutdown()
            except Exception as exc:
                errors.append(exc)
        if self._raw_gripper is not None:
            # Idempotent, and still needed if initialization or the controller's
            # slow-stop failed before it could release the gripper bus.
            try:
                self._raw_gripper.shutdown()
            except Exception as exc:
                errors.append(exc)
        if self.arm is not None:
            try:
                self.arm.close()
            except Exception as exc:
                errors.append(exc)
        if self._sdk_initialized:
            try:
                status = self._RealManArm.destroy_sdk()
                if status != 0:
                    errors.append(RuntimeError(f"destroy SDK status={status}"))
            except Exception as exc:
                errors.append(exc)
        if self.logger is not None:
            try:
                self.logger.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(
                "left RM65 cleanup errors: "
                + "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
            )
