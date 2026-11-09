"""Narrow RealMan SDK adapter used by the clean controller."""

from __future__ import annotations

import importlib
import threading
import time
from typing import Protocol

import numpy as np

from .models import FrameSpec
from .transforms import (
    finite_vector, pose6_to_transform, transform_to_pose6,
    base_flange_from_work_tcp, work_tcp_from_base_flange,
)


class ArmError(RuntimeError):
    pass


class ToolRegisterWriteError(ArmError):
    """Preserve the SDK result for bounded, idempotent startup retries."""

    def __init__(self, address: int, status: int):
        self.status = int(status)
        super().__init__(f"rm_write_single_register address=0x{address:04X} failed: {status}")


class ArmFeedbackTimeout(ArmError):
    """A missed feedback/health read; stop active commands before retrying."""


class ArmCommunicationError(ArmError):
    """The SDK could not send or decode a query; its payload is not valid."""


class InverseKinematicsError(ArmError):
    """The SDK found no IK solution, distinct from communication/feedback faults."""


class ArmProtocol(Protocol):
    def read_frames(self) -> tuple[FrameSpec, FrameSpec]: ...

    def read_joint_positions(self) -> np.ndarray: ...

    def forward_pose(self, q_rad) -> np.ndarray: ...

    def inverse_kinematics(self, seed_q_rad, flange_pose_m_rad) -> np.ndarray: ...

    def send_joint_target(
        self,
        q_rad,
        *,
        cancel_event: threading.Event,
    ) -> int: ...

    def slow_stop(self) -> int: ...

    def start_home_joint_move(self, q_rad, *, speed_percent: int) -> int: ...

    def close(self) -> None: ...


class RealManArm:
    """One RM65 connection with no Cartesian motion-dispatch method.

    With normalize_kinematics enabled, expose base/flange FK and IK while the
    connected SDK consumes/returns active-work/TCP poses. Legacy incremental
    callers keep their commissioned behavior until explicitly migrated.
    Physical motion is published only with
    ``rm_movej_canfd`` after the controller has validated a rate-compatible
    joint target.
    """

    def __init__(
        self,
        ip: str,
        port: int,
        *,
        collision_level: int,
        enable_self_collision: bool,
        configure_safety: bool = True,
        verify_motion_safety: bool = True,
        verify_self_collision: bool = True,
        initialize_sdk: bool = True,
        normalize_kinematics: bool = False,
        idle_health_read_attempts: int = 1,
        retry_stop_receive_timeout: bool = False,
    ):
        sdk = importlib.import_module("Robotic_Arm.rm_robot_interface")
        self._inverse_params_type = sdk.rm_inverse_kinematics_params_t
        self._peripheral_params_type = sdk.rm_peripheral_read_write_params_t
        self._lock = threading.RLock()
        self._stop_lock = threading.Lock()
        self._stop_diagnostics = threading.local()
        self._retry_stop_receive_timeout = bool(retry_stop_receive_timeout)
        self._closed = False
        if int(idle_health_read_attempts) not in (1, 2, 3):
            raise ValueError("idle_health_read_attempts must be 1, 2 or 3")
        self._idle_health_read_attempts = int(idle_health_read_attempts)
        self._normalize_kinematics = bool(normalize_kinematics)
        self._kinematic_frames = None
        self._last_health_check_monotonic = 0.0
        # Full health polling is several synchronous SDK round trips.  Keep it
        # out of an active CANFD stream; feedback/send return codes remain
        # checked on every cycle, and the next idle feedback read performs any
        # overdue full health check before a new motion session can start.
        self._streaming_motion_active = False
        self.ip = str(ip)
        self.port = int(port)
        self.robot = (
            sdk.RoboticArm(sdk.rm_thread_mode_e.RM_TRIPLE_MODE_E)
            if bool(initialize_sdk)
            else sdk.RoboticArm()
        )
        connected = False
        try:
            self.handle = self.robot.rm_create_robot_arm(self.ip, self.port)
            if self.handle.id < 0:
                raise ArmError(f"cannot connect to RM65 at {self.ip}:{self.port}")
            connected = True
            self._verify_model_and_health()
            if configure_safety:
                self._configure_safety(collision_level, enable_self_collision)
            elif verify_motion_safety:
                self._verify_safety(
                    collision_level,
                    enable_self_collision,
                    verify_self_collision=verify_self_collision,
                )
        except Exception:
            if connected:
                try:
                    self.robot.rm_delete_robot_arm()
                except Exception:
                    pass
            self._closed = True
            raise

    @staticmethod
    def _arm_error_is_nonzero(error_state) -> bool:
        if not isinstance(error_state, dict):
            return bool(error_state)
        return any(
            str(value).strip() not in ("", "0") for value in error_state.get("err", [])
        )

    def _verify_model_and_health(self) -> None:
        status, info = self.robot.rm_get_arm_software_info()
        if status != 0:
            raise ArmError(f"rm_get_arm_software_info failed: {status}")
        model = str(info.get("product_version", "unknown"))
        if not model.startswith("RM65"):
            raise ArmError(f"connected model is {model!r}, expected RM65*")
        self.check_health()

    @staticmethod
    def _check_query_status(operation: str, status: int) -> None:
        if status == 0:
            return
        if status == -2:
            raise ArmFeedbackTimeout(f"{operation} feedback receive timeout: status={status}")
        if status in (-1, -3):
            detail = "send failed" if status == -1 else "response decode failed"
            raise ArmCommunicationError(
                f"{operation} communication {detail}: status={status}; "
                "returned state is unavailable; check RM65 network/power connection"
            )
        raise ArmError(f"{operation} query failed: status={status}")

    def check_health(self) -> None:
        mode_status, run_mode = self.robot.rm_get_arm_run_mode()
        self._check_query_status("rm_get_arm_run_mode", mode_status)
        if int(run_mode) != 1:
            raise ArmError(
                "RM65 is not in real-hardware mode: "
                f"status={mode_status}, run_mode={run_mode} (required=1)"
            )
        power_status, power = self.robot.rm_get_arm_power_state()
        self._check_query_status("rm_get_arm_power_state", power_status)
        if int(power) != 1:
            raise ArmError(
                f"RM65 power check failed: status={power_status}, power={power}"
            )
        controller = self.robot.rm_get_controller_state()
        self._check_query_status("rm_get_controller_state", controller.get("return_code"))
        if int(controller.get("system_error", -1)) != 0:
            raise ArmError(f"controller system error: {controller}")
        all_status, state = self.robot.rm_get_arm_all_state()
        self._check_query_status("rm_get_arm_all_state", all_status)
        enabled = [int(value) for value in state.get("joint_en_flag", [])[:6]]
        errors = [int(value) for value in state.get("joint_err_code", [])[:6]]
        if len(enabled) != 6 or not all(enabled):
            raise ArmError(f"not all RM65 joints are enabled: {enabled}")
        if len(errors) != 6 or any(errors):
            raise ArmError(f"RM65 joint errors: {errors}")
        if self._arm_error_is_nonzero(state.get("err", {})):
            raise ArmError(f"RM65 arm error: {state.get('err')}")
        self._last_health_check_monotonic = time.monotonic()

    def set_streaming_motion_active(self, active: bool) -> None:
        """Defer slow periodic health queries while CANFD targets are streaming."""
        with self._lock:
            self._streaming_motion_active = bool(active)

    def _configure_safety(self, collision_level: int, self_collision: bool) -> None:
        if not 1 <= int(collision_level) <= 8:
            raise ValueError("collision_level must be in [1, 8]")
        status = self.robot.rm_set_collision_state(int(collision_level))
        if status != 0:
            raise ArmError(f"rm_set_collision_state failed: {status}")
        status = self.robot.rm_set_self_collision_enable(bool(self_collision))
        if status != 0:
            raise ArmError(f"rm_set_self_collision_enable failed: {status}")
        for _ in range(10):
            try:
                self._verify_safety(collision_level, self_collision)
                return
            except ArmError:
                pass
            time.sleep(0.1)
        raise ArmError("RM65 collision-safety settings could not be verified")

    def _verify_safety(
        self,
        collision_level: int,
        self_collision: bool,
        *,
        verify_self_collision: bool = True,
    ) -> None:
        if not 1 <= int(collision_level) <= 8:
            raise ValueError("collision_level must be in [1, 8]")
        level_status, level = self.robot.rm_get_collision_stage()
        self_status, enabled = self.robot.rm_get_self_collision_enable()
        if level_status != 0 or int(level) != int(collision_level):
            raise ArmError(
                "RM65 collision level mismatch: "
                f"status={level_status}, actual={level}, expected={collision_level}"
            )
        if verify_self_collision and (
            self_status != 0 or bool(enabled) != bool(self_collision)
        ):
            raise ArmError(
                "RM65 self-collision mismatch: "
                f"status={self_status}, actual={enabled}, expected={self_collision}"
            )

    def read_frames(self) -> tuple[FrameSpec, FrameSpec]:
        with self._lock:
            work_status, work = self.robot.rm_get_current_work_frame()
            tool_status, tool = self.robot.rm_get_current_tool_frame()
        if work_status != 0 or not isinstance(work, dict):
            raise ArmError(f"cannot read current work frame: {work_status}, {work!r}")
        if tool_status != 0 or not isinstance(tool, dict):
            raise ArmError(f"cannot read current tool frame: {tool_status}, {tool!r}")
        frames = (
            FrameSpec(str(work.get("name", "")), work.get("pose", [])),
            FrameSpec(str(tool.get("name", "")), tool.get("pose", [])),
        )
        self._kinematic_frames = tuple(pose6_to_transform(f.pose_m_rad) for f in frames)
        return frames

    def _require_kinematic_frames(self):
        if self._kinematic_frames is None:
            raise ArmError("read_frames must succeed before normalized FK/IK")
        return self._kinematic_frames

    def _check_idle_health(self) -> None:
        # Retry only a receive timeout in read-only idle health queries. A real
        # power, joint, enable or communication fault still fails immediately.
        deadline = time.monotonic() + 1.0
        attempts = self._idle_health_read_attempts
        for attempt in range(1, attempts + 1):
            try:
                self.check_health()
                return
            except ArmFeedbackTimeout as exc:
                if attempt >= attempts or time.monotonic() + 0.05 >= deadline:
                    raise
                print(f"[RM65 HEALTH RETRY] {exc}; attempt={attempt + 1}/{attempts}", flush=True)
                time.sleep(0.05)

    def read_joint_positions(self) -> np.ndarray:
        with self._lock:
            if (
                not getattr(self, "_streaming_motion_active", False)
                and time.monotonic() - self._last_health_check_monotonic >= 2.0
            ):
                self._check_idle_health()
            status, q_deg = self.robot.rm_get_joint_degree()
        if status != 0:
            error = ArmFeedbackTimeout if status == -2 else ArmError
            raise error(f"rm_get_joint_degree failed: {status}")
        return np.deg2rad(finite_vector(q_deg[:6], 6, "RM65 joint feedback"))

    def read_commissioning_feedback(self) -> dict:
        """Read two joint-state APIs plus motor currents for physical bring-up."""
        with self._lock:
            degree_status, degree_values = self.robot.rm_get_joint_degree()
            state_status, state = self.robot.rm_get_current_arm_state()
            all_status, all_state = self.robot.rm_get_arm_all_state()
        if degree_status != 0:
            raise ArmError(f"rm_get_joint_degree failed: {degree_status}")
        if state_status != 0 or not isinstance(state, dict):
            raise ArmError(
                f"rm_get_current_arm_state failed: {state_status}, {state!r}"
            )
        if all_status != 0 or not isinstance(all_state, dict):
            raise ArmError(f"rm_get_arm_all_state failed: {all_status}, {all_state!r}")
        direct_deg = finite_vector(degree_values[:6], 6, "direct joint feedback")
        state_deg = finite_vector(
            state.get("joint", [])[:6], 6, "arm-state joint feedback"
        )
        mismatch_deg = float(np.max(np.abs(direct_deg - state_deg)))
        if mismatch_deg > 0.1:
            raise ArmError(
                "RM65 joint feedback APIs disagree: "
                f"max_error={mismatch_deg:.3f} deg"
            )
        currents = finite_vector(
            all_state.get("joint_current", [])[:6],
            6,
            "RM65 joint-current feedback",
        )
        return {
            "q_rad": np.deg2rad(direct_deg),
            "cross_feedback_error_deg": mismatch_deg,
            "joint_current": currents,
        }

    def forward_pose(self, q_rad) -> np.ndarray:
        q_rad = finite_vector(q_rad, 6, "FK joint input")
        with self._lock:
            result = self.robot.rm_algo_forward_kinematics(
                np.rad2deg(q_rad).tolist(),
                1,
            )
        pose = finite_vector(result[:6], 6, "RM65 FK pose")
        if self._normalize_kinematics:
            pose = transform_to_pose6(base_flange_from_work_tcp(
                pose6_to_transform(pose), *self._require_kinematic_frames()
            ))
        return pose

    def inverse_kinematics(self, seed_q_rad, flange_pose_m_rad) -> np.ndarray:
        seed = finite_vector(seed_q_rad, 6, "IK seed")
        pose = finite_vector(flange_pose_m_rad, 6, "IK flange target")
        if self._normalize_kinematics:
            pose = transform_to_pose6(work_tcp_from_base_flange(
                pose6_to_transform(pose), *self._require_kinematic_frames()
            ))
        q_in_deg = np.zeros(7, dtype=float)
        q_in_deg[:6] = np.rad2deg(seed)
        params = self._inverse_params_type(
            q_in=q_in_deg.tolist(),
            q_pose=pose.tolist(),
            flag=1,
        )
        with self._lock:
            status, solution_deg = self.robot.rm_algo_inverse_kinematics(params)
        if status != 0:
            error_type = InverseKinematicsError if status == 1 else ArmError
            raise error_type(f"RM65 IK failed: status={status}, target={pose.tolist()}")
        return np.deg2rad(finite_vector(solution_deg[:6], 6, "RM65 IK solution"))

    def send_joint_target(
        self,
        q_rad,
        *,
        cancel_event: threading.Event,
    ) -> int:
        q_rad = finite_vector(q_rad, 6, "joint command")
        if cancel_event.is_set():
            raise ArmError("joint command cancelled before transmission")
        with self._lock:
            if cancel_event.is_set():
                raise ArmError("joint command cancelled before SDK entry")
            status = self.robot.rm_movej_canfd(
                np.rad2deg(q_rad).tolist(),
                follow=False,
                expand=0,
                trajectory_mode=0,
                radio=0,
            )
        if cancel_event.is_set():
            stop_status = self.slow_stop()
            raise ArmError(
                "joint command cancelled during SDK call; "
                f"post-command slow-stop status={stop_status}"
            )
        if status != 0:
            raise ArmError(f"rm_movej_canfd failed: {status}")
        return int(status)

    def _start_joint_move(
        self, q_rad, *, speed_percent: int, maximum_speed: int
    ) -> int:
        q_rad = finite_vector(q_rad, 6, "planned joint target")
        speed_percent = int(speed_percent)
        if not 1 <= speed_percent <= int(maximum_speed):
            raise ValueError(f"speed_percent must be in [1, {maximum_speed}]")
        with self._lock:
            status = self.robot.rm_movej(
                np.rad2deg(q_rad).tolist(),
                v=speed_percent,
                r=0,
                connect=0,
                block=0,
            )
        if status != 0:
            raise ArmError(f"rm_movej failed: {status}")
        return int(status)

    def start_planned_joint_move(self, q_rad, *, speed_percent: int = 1) -> int:
        """Start the bounded commissioning MoveJ used by the endpoint test."""
        return self._start_joint_move(
            q_rad,
            speed_percent=speed_percent,
            maximum_speed=3,
        )

    def start_home_joint_move(self, q_rad, *, speed_percent: int = 10) -> int:
        """Start a nonblocking MoveJ to the startup pose captured by this process."""
        return self._start_joint_move(
            q_rad,
            speed_percent=speed_percent,
            maximum_speed=10,
        )

    def configure_tool_modbus(
        self,
        *,
        baudrate: int,
        timeout_units: int,
        tool_voltage: int,
    ) -> None:
        """Configure the RM65 tool port for the integrated gripper."""
        if int(tool_voltage) not in (0, 24):
            raise ValueError("tool_voltage must be 0 or 24")
        with self._lock:
            if int(tool_voltage) == 24:
                status = self.robot.rm_set_tool_voltage(3)
                if status != 0:
                    raise ArmError(f"rm_set_tool_voltage failed: {status}")
            status = self.robot.rm_set_modbus_mode(
                1,
                int(baudrate),
                int(timeout_units),
            )
        if status != 0:
            raise ArmError(f"rm_set_modbus_mode failed: {status}")

    def write_tool_single_register(
        self,
        *,
        device_id: int,
        address: int,
        value: int,
    ) -> None:
        value = int(value)
        if not 0 <= value <= 0xFFFF:
            raise ValueError("tool register value must be in [0, 65535]")
        params = self._peripheral_params_type(1, int(address), int(device_id))
        with self._lock:
            status = self.robot.rm_write_single_register(params, value)
        if status != 0:
            raise ToolRegisterWriteError(int(address), int(status))

    def write_tool_registers(
        self,
        *,
        device_id: int,
        address: int,
        values,
    ) -> None:
        values = [int(value) for value in values]
        if not values:
            raise ValueError("tool register values may not be empty")
        if any(value < 0 or value > 0xFFFF for value in values):
            raise ValueError("tool register values must be in [0, 65535]")
        data_bytes = []
        for value in values:
            data_bytes.extend(((value >> 8) & 0xFF, value & 0xFF))
        params = self._peripheral_params_type(
            1,
            int(address),
            int(device_id),
            len(values),
        )
        with self._lock:
            status = self.robot.rm_write_registers(params, data_bytes)
        if status != 0:
            raise ArmError(
                "rm_write_registers "
                f"address=0x{int(address):04X} failed: {status}"
            )

    def read_tool_register(self, *, device_id: int, address: int) -> int:
        params = self._peripheral_params_type(1, int(address), int(device_id), None)
        with self._lock:
            result = self.robot.rm_read_holding_registers(params)
        if not isinstance(result, tuple) or len(result) != 2 or int(result[0]) != 0:
            raise ArmError(
                f"rm_read_holding_registers address=0x{int(address):04X} failed: {result!r}"
            )
        return int(result[1])

    def close_tool_modbus(self) -> None:
        with self._lock:
            status = self.robot.rm_close_modbus_mode(1)
        if status != 0:
            raise ArmError(f"rm_close_modbus_mode failed: {status}")

    def slow_stop(self) -> int:
        if not self._retry_stop_receive_timeout:
            # Preserve other callers' commissioned behavior.
            return int(self.robot.rm_set_arm_slow_stop())
        # Serialize stop callers, but NEVER wait behind ordinary SDK I/O before
        # the first stop. An in-flight joint query must not delay the watchdog.
        # The controller cancels its session before entering this method, and
        # send_joint_target checks cancellation before/after SDK transmission.
        with self._stop_lock:
            details = {"started_monotonic": time.monotonic(), "attempts": []}
            self._stop_diagnostics.value = details
            io_locked = self._lock.acquire(blocking=False)
            details["initial_overlapped_io"] = not io_locked
            started = time.monotonic()
            try:
                status = int(self.robot.rm_set_arm_slow_stop())
            finally:
                if io_locked:
                    self._lock.release()
            details["attempts"].append({"status": status, "duration_s": time.monotonic() - started})
            if status != -2:
                return status

            # -2 means the response was missing, not proof that the stop was
            # rejected. Reissue only this idempotent STOP, once, after outstanding
            # I/O drains. Never clear errors, enable motion, or infer success from
            # encoder stillness alone. A second missing ACK remains a hard fault.
            print("[RM65 STOP RETRY] 停止应答超时，等待当前通信结束后重发一次停止指令。", flush=True)
            wait_started = time.monotonic()
            io_locked = self._lock.acquire(timeout=0.25)
            details["retry_lock_wait_s"] = time.monotonic() - wait_started
            if not io_locked:
                details["retry_skipped"] = "io_still_busy"
                print("[RM65 STOP UNCONFIRMED] 通信仍阻塞，停止状态未确认。", flush=True)
                return status
            started = time.monotonic()
            try:
                status = int(self.robot.rm_set_arm_slow_stop())
            finally:
                self._lock.release()
            details["attempts"].append({"status": status, "duration_s": time.monotonic() - started})
            print(f"[RM65 STOP {'ACK RECOVERED' if status == 0 else 'UNCONFIRMED'}] "
                  f"retry_status={status}", flush=True)
            return status

    def last_stop_diagnostics(self) -> dict:
        # Per-thread details prevent a concurrent post-cancellation/shutdown stop
        # from replacing the watchdog stop's result before it is logged.
        details = getattr(self._stop_diagnostics, "value", {})
        return {**details, "attempts": [dict(item) for item in details.get("attempts", [])]}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            status = self.robot.rm_delete_robot_arm()
            if status != 0:
                raise ArmError(f"rm_delete_robot_arm failed: {status}")
            self._closed = True

    @staticmethod
    def destroy_sdk() -> int:
        """Release the process-wide RealMan SDK after all arm handles close."""
        sdk = importlib.import_module("Robotic_Arm.rm_robot_interface")
        return int(sdk.RoboticArm.rm_destroy())
