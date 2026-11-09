"""Per-controller gestures and DH-Robotics AG gripper support."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional


class AccessoryError(RuntimeError):
    pass


class SafeButtonGesture:
    """Classify a face-button release as SHORT or LONG while clutches are free."""

    def __init__(self, long_hold_s: float = 1.0):
        if float(long_hold_s) <= 0.0:
            raise ValueError("long_hold_s must be positive")
        self.long_hold_s = float(long_hold_s)
        self._neutral_seen = False
        self._pressed_since: Optional[float] = None

    def update(
        self,
        *,
        grip: float,
        trigger: float,
        pressed: bool,
        now: Optional[float] = None,
    ) -> Optional[str]:
        now = time.monotonic() if now is None else float(now)
        if float(grip) >= 0.1 or float(trigger) >= 0.1:
            self._neutral_seen = False
            self._pressed_since = None
            return None
        if bool(pressed):
            if self._neutral_seen and self._pressed_since is None:
                self._pressed_since = now
            return None
        if self._pressed_since is None:
            self._neutral_seen = True
            return None
        duration = now - self._pressed_since
        self._pressed_since = None
        self._neutral_seen = True
        return "LONG" if duration >= self.long_hold_s else "SHORT"


class GripperTriggerLatch:
    """Emit one TOGGLE action for each standalone Trigger press."""

    def __init__(
        self,
        *,
        debounce_s: float = 0.3,
        activation_threshold: float = 0.8,
        release_threshold: float = 0.1,
    ):
        if float(debounce_s) < 0.0:
            raise ValueError("debounce_s may not be negative")
        self.debounce_s = float(debounce_s)
        self.activation_threshold = float(activation_threshold)
        self.release_threshold = float(release_threshold)
        self._neutral_seen = False
        self._gesture_active = False
        self._last_action_time = float("-inf")

    def update(
        self,
        *,
        grip: float,
        trigger: float,
        now: Optional[float] = None,
    ) -> Optional[str]:
        now = time.monotonic() if now is None else float(now)
        trigger = float(trigger)
        if float(grip) >= 0.1:
            self._neutral_seen = False
            self._gesture_active = False
            return None
        if trigger <= self.release_threshold:
            self._neutral_seen = True
            self._gesture_active = False
            return None
        if trigger >= self.activation_threshold:
            if self._neutral_seen and not self._gesture_active:
                self._gesture_active = True
                self._neutral_seen = False
                if now - self._last_action_time < self.debounce_s:
                    return None
                self._last_action_time = now
                return "TOGGLE"
            return None
        return None


class GripperChordLatch:
    """Hold Grip (first) and press Trigger (second); rearm on Trigger release."""

    def __init__(self, *, debounce_s=0.3):
        if float(debounce_s) < 0.:
            raise ValueError("debounce_s may not be negative")
        self.debounce_s = float(debounce_s)
        self._neutral_seen = False
        self._last_action_time = float("-inf")

    def reset(self):
        self._neutral_seen = False

    def update(self, *, first, second, enabled=True, now=None):
        now = time.monotonic() if now is None else float(now)
        if not enabled:
            self.reset()
            return None
        if float(second) <= .1:
            self._neutral_seen = True
        elif float(first) >= .8 and float(second) >= .8 and self._neutral_seen:
            self._neutral_seen = False
            if now - self._last_action_time >= self.debounce_s:
                self._last_action_time = now
                return "TOGGLE"
        return None


class AGGripper:
    """Checked DH-Robotics AG protocol routed through the RM65 tool bus."""

    PROTOCOL_NAME = "AG"
    DEFAULT_BAUDRATE = 115200

    REG_INITIALIZE = 0x0100
    REG_FORCE = 0x0101
    REG_TARGET_POSITION = 0x0103
    REG_INITIALIZE_STATE = 0x0200
    REG_GRIP_STATE = 0x0201
    REG_POSITION = 0x0202

    STATE_MOVING = 0
    STATE_REACHED = 1
    STATE_GRIPPED = 2
    STATE_DROPPED = 3

    POSITION_CLOSED = 0
    POSITION_OPEN = 1000

    def __init__(
        self,
        arm,
        *,
        device_id: int = 1,
        force_percent: int = 30,
        timeout_units: int = 2,
        tool_voltage: int = 0,
    ):
        if not 1 <= int(device_id) <= 247:
            raise ValueError("device_id must be in [1, 247]")
        if not 20 <= int(force_percent) <= 100:
            raise ValueError("AG force_percent must be in [20, 100]")
        if int(timeout_units) <= 0:
            raise ValueError("timeout_units must be positive")
        if int(tool_voltage) not in (0, 24):
            raise ValueError("tool_voltage must be 0 or 24")
        self.arm = arm
        self.device_id = int(device_id)
        self.force_percent = int(force_percent)
        self.timeout_units = int(timeout_units)
        self.tool_voltage = int(tool_voltage)
        self._configured = False
        self._initialized = False
        self._last_target_open: Optional[bool] = None
        self._command_sequence = 0

    def _write(self, address: int, value: int) -> None:
        try:
            self.arm.write_tool_single_register(
                device_id=self.device_id,
                address=int(address),
                value=int(value),
            )
        except Exception as exc:
            raise AccessoryError(
                f"AG write 0x{int(address):04X} failed: {exc}"
            ) from exc

    def _read(self, address: int) -> int:
        try:
            return int(
                self.arm.read_tool_register(
                    device_id=self.device_id,
                    address=int(address),
                )
            )
        except Exception as exc:
            raise AccessoryError(
                f"AG read 0x{int(address):04X} failed: {exc}"
            ) from exc

    @staticmethod
    def _validate_feedback(position: int, grip_state: int) -> None:
        if not 0 <= int(position) <= 1000:
            raise AccessoryError(
                f"AG returned invalid position={position}"
            )
        if int(grip_state) not in (
            AGGripper.STATE_MOVING,
            AGGripper.STATE_REACHED,
            AGGripper.STATE_GRIPPED,
            AGGripper.STATE_DROPPED,
        ):
            raise AccessoryError(
                f"AG returned invalid grip_state={grip_state}"
            )

    def _read_initialization_register(self, address: int, deadline: float) -> int:
        """Retry reads only during startup; never replay writes or fake feedback."""
        for attempt in range(1, 4):
            try:
                value = self._read(address)
            except AccessoryError as exc:
                remaining = deadline - time.monotonic()
                retry = attempt < 3 and remaining > 0.1
                stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                print(
                    f"[{stamp}][AG INIT READ {'RETRY' if retry else 'FAILED'}] "
                    f"device={self.device_id} register=0x{address:04X} "
                    f"attempt={attempt}/3 error={exc}", flush=True,
                )
                if not retry:
                    raise AccessoryError(
                        f"AG startup feedback unavailable at 0x{address:04X} "
                        f"after {attempt} read attempt(s); check 24V supply, "
                        f"RS485 wiring and other bus users. Last error: {exc}"
                    ) from exc
                time.sleep(0.1)
            else:
                if attempt > 1:
                    stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                    print(
                        f"[{stamp}][AG INIT READ RECOVERED] device={self.device_id} "
                        f"register=0x{address:04X} attempt={attempt}/3 value={value}",
                        flush=True,
                    )
                return value

    def _check_startup_communication(self, timeout_s: float) -> None:
        """Confirm feedback before initialization; reopen the bus at most once."""
        deadline = time.monotonic() + min(float(timeout_s), 3.0)
        for connection in range(2):
            try:
                state = self._read_initialization_register(
                    self.REG_INITIALIZE_STATE, deadline
                )
            except AccessoryError as exc:
                if connection == 1 or deadline - time.monotonic() <= 0.3:
                    raise AccessoryError(
                        "AG startup communication unavailable before initialization "
                        f"(port=1, device={self.device_id}, baud={self.DEFAULT_BAUDRATE}); "
                        "夹爪未回复，未发送初始化或开合指令。"
                        "请检查夹爪指示灯、电源线、RS485 接头及设备 ID/波特率。"
                        f" Last error: {exc}"
                    ) from exc
                print("[AG STARTUP RECONNECT] 夹爪状态读取失败，重建一次末端 Modbus 连接。", flush=True)
                self.arm.close_tool_modbus()
                self._configured = False
                time.sleep(0.2)
                self.arm.configure_tool_modbus(
                    baudrate=self.DEFAULT_BAUDRATE,
                    timeout_units=self.timeout_units,
                    tool_voltage=0,
                )
                self._configured = True
                time.sleep(0.1)
                continue
            if state not in (0, 1, 2):
                raise AccessoryError(f"AG returned invalid initialization state={state}")
            print(f"[AG STARTUP READY] device={self.device_id} initialize_state={state}", flush=True)
            return

    def _write_startup_setting(self, address: int, value: int, deadline: float) -> None:
        """Retry the SAME absolute startup setting; never replay INIT or toggle."""
        if address not in (self.REG_FORCE, self.REG_TARGET_POSITION):
            raise ValueError("startup write recovery only supports force/absolute position")
        for attempt in range(1, 4):
            if attempt > 1 and time.monotonic() >= deadline:
                raise AccessoryError(f"AG startup write recovery timed out at 0x{address:04X}")
            try:
                self._write(address, value)
                return
            except AccessoryError as exc:
                status = getattr(exc.__cause__, "status", None)
                if status not in (-1, -2, -3, 1) or attempt >= 3 or time.monotonic() + 0.1 >= deadline:
                    raise
                print(f"[AG INIT WRITE RETRY] register=0x{address:04X} value={value} "
                      f"attempt={attempt + 1}/3 error={exc}", flush=True)
                time.sleep(0.1)
                state = self._read_initialization_register(self.REG_INITIALIZE_STATE, deadline)
                if state != 1:
                    raise AccessoryError(f"AG initialization lost during startup write: state={state}")

    def initialize(self, timeout_s: float = 8.0, *, verify_communication: bool = False,
                   retry_startup_writes: bool = False) -> dict:
        if float(timeout_s) <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.arm.configure_tool_modbus(
            baudrate=self.DEFAULT_BAUDRATE,
            timeout_units=self.timeout_units,
            tool_voltage=self.tool_voltage,
        )
        self._configured = True
        if verify_communication:
            self._check_startup_communication(timeout_s)
        self._write(self.REG_INITIALIZE, 1)

        deadline = time.monotonic() + float(timeout_s)
        initialize_state = None
        while time.monotonic() < deadline:
            initialize_state = self._read_initialization_register(
                self.REG_INITIALIZE_STATE, deadline
            )
            if initialize_state == 1:
                break
            if initialize_state not in (0, 2):
                raise AccessoryError(
                    "AG returned invalid initialization state="
                    f"{initialize_state}"
                )
            time.sleep(0.1)
        else:
            raise AccessoryError(
                f"AG initialization timed out after {float(timeout_s):.1f} s; "
                f"last_state={initialize_state}"
            )

        if retry_startup_writes:
            self._write_startup_setting(self.REG_FORCE, self.force_percent, deadline)
        else:
            self._write(self.REG_FORCE, self.force_percent)
        position = self._read_initialization_register(self.REG_POSITION, deadline)
        grip_state = self._read_initialization_register(self.REG_GRIP_STATE, deadline)
        self._validate_feedback(position, grip_state)
        self._last_target_open = position >= 500
        self._initialized = True

        # Match the verified AG implementation in the outer hardware project:
        # finish initialization with a deterministic closed target so the next
        # standalone Trigger toggle always starts by opening.
        try:
            if retry_startup_writes:
                self._write_startup_setting(self.REG_TARGET_POSITION, self.POSITION_CLOSED, deadline)
                self._last_target_open = False
                # Return measured post-command feedback, not the earlier open
                # position and not an assumed closed position.
                position = self._read_initialization_register(self.REG_POSITION, deadline)
                grip_state = self._read_initialization_register(self.REG_GRIP_STATE, deadline)
                self._validate_feedback(position, grip_state)
            else:
                self.close()
        except Exception:
            self._initialized = False
            raise
        return {
            "position": position,
            "grip_state": grip_state,
            "force_percent": self.force_percent,
            "target_open": False,
            "target_position": self.POSITION_CLOSED,
            "startup_action": "CLOSE",
        }

    def get_motion_status(self) -> dict:
        position = self._read(self.REG_POSITION)
        grip_state = self._read(self.REG_GRIP_STATE)
        self._validate_feedback(position, grip_state)
        return {
            "position": position,
            "grip_state": grip_state,
            "force_percent": self.force_percent,
        }

    def set_position(self, position: int) -> None:
        if not self._initialized:
            raise AccessoryError("AG must be initialized before movement")
        if not 0 <= int(position) <= 1000:
            raise ValueError("AG position must be in [0, 1000]")
        self._write(self.REG_TARGET_POSITION, int(position))

    def open(self) -> None:
        self.set_position(self.POSITION_OPEN)
        self._last_target_open = True

    def close(self) -> None:
        self.set_position(self.POSITION_CLOSED)
        self._last_target_open = False

    def command_open_state(self, target_open: bool) -> dict:
        if not self._initialized:
            raise AccessoryError("AG must be initialized before movement")
        target_open = bool(target_open)
        target_position = self.POSITION_OPEN if target_open else self.POSITION_CLOSED
        started = time.monotonic()
        self.set_position(target_position)
        completed = time.monotonic()
        self._last_target_open = target_open
        self._command_sequence += 1
        return {
            "command_id": self._command_sequence,
            "target_open": target_open,
            "target_position": target_position,
            "force_percent": self.force_percent,
            "target_register": self.REG_TARGET_POSITION,
            "command_dispatched": True,
            "write_duration_ms": (completed - started) * 1000.0,
        }

    def shutdown(self) -> None:
        if not self._configured:
            return
        try:
            self.arm.close_tool_modbus()
        finally:
            self._configured = False
            self._initialized = False


class Mini9CGripper:
    """MINI-9C gripper using the verified 0x9Cxx Modbus register protocol."""

    PROTOCOL_NAME = "MINI-9C"
    DEFAULT_BAUDRATE = 115200

    REG_TARGET_WIDTH = 0x9C40
    REG_FORCE = 0x9C41
    REG_POSITION = 0x9C45
    REG_TORQUE = 0x9C46
    REG_MOTION_DONE = 0x9C47
    REG_EXECUTE = 0x9C48
    REG_AUTO_EXECUTE = 0x9C9A

    AUTO_EXECUTE_DISABLE_PERSISTENT = 2
    AUTO_EXECUTE_DISABLE_RUNTIME = 1

    STATE_MOVING = 0
    STATE_REACHED = 1
    POSITION_CLOSED = 0
    POSITION_OPEN = 100

    def __init__(
        self,
        arm,
        *,
        device_id: int = 1,
        force_percent: int = 30,
        timeout_units: int = 2,
        tool_voltage: int = 0,
    ):
        if not 1 <= int(device_id) <= 247:
            raise ValueError("device_id must be in [1, 247]")
        if not 0 <= int(force_percent) <= 100:
            raise ValueError("MINI-9C force_percent must be in [0, 100]")
        if int(timeout_units) <= 0:
            raise ValueError("timeout_units must be positive")
        if int(tool_voltage) not in (0, 24):
            raise ValueError("tool_voltage must be 0 or 24")
        self.arm = arm
        self.device_id = int(device_id)
        self.force_percent = int(force_percent)
        self.timeout_units = int(timeout_units)
        self.tool_voltage = int(tool_voltage)
        self._configured = False
        self._initialized = False
        self._last_target_open: Optional[bool] = None
        self._command_sequence = 0

    def _write(self, address: int, value: int) -> None:
        try:
            self.arm.write_tool_registers(
                device_id=self.device_id,
                address=int(address),
                values=[int(value)],
            )
        except Exception as exc:
            raise AccessoryError(
                f"MINI-9C write 0x{int(address):04X} failed: {exc}"
            ) from exc

    def _read(self, address: int) -> int:
        try:
            return int(
                self.arm.read_tool_register(
                    device_id=self.device_id,
                    address=int(address),
                )
            )
        except Exception as exc:
            raise AccessoryError(
                f"MINI-9C read 0x{int(address):04X} failed: {exc}"
            ) from exc

    @staticmethod
    def _validate_feedback(position: int, torque: int, done: int) -> None:
        if not 0 <= int(position) <= 100:
            raise AccessoryError(f"MINI-9C returned invalid position={position}")
        if not 0 <= int(torque) <= 100:
            raise AccessoryError(f"MINI-9C returned invalid torque={torque}")
        if int(done) not in (0, 1):
            raise AccessoryError(f"MINI-9C returned invalid motion_done={done}")

    def initialize(self, timeout_s: float = 8.0) -> dict:
        if float(timeout_s) <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.arm.configure_tool_modbus(
            baudrate=self.DEFAULT_BAUDRATE,
            timeout_units=self.timeout_units,
            tool_voltage=self.tool_voltage,
        )
        self._configured = True
        persistent_disabled = False
        try:
            self._write(
                self.REG_AUTO_EXECUTE,
                self.AUTO_EXECUTE_DISABLE_PERSISTENT,
            )
            persistent_disabled = True
        except AccessoryError as persistent_error:
            print(
                "[GRIPPER COMPAT] MINI-9C persistent auto-run disable "
                f"unsupported ({persistent_error}); using runtime-only disable.",
                flush=True,
            )
        try:
            self._write(
                self.REG_AUTO_EXECUTE,
                self.AUTO_EXECUTE_DISABLE_RUNTIME,
            )
        except AccessoryError as runtime_error:
            raise AccessoryError(
                "MINI-9C cannot enter strict manual mode; refusing control "
                f"because automatic travel may remain enabled ({runtime_error})"
            ) from runtime_error
        self._write(self.REG_FORCE, self.force_percent)
        position = self._read(self.REG_POSITION)
        torque = self._read(self.REG_TORQUE)
        done = self._read(self.REG_MOTION_DONE)
        self._validate_feedback(position, torque, done)
        self._last_target_open = position >= 50
        self._initialized = True
        return {
            "position": position,
            "grip_state": self.STATE_REACHED if done else self.STATE_MOVING,
            "motion_done": done,
            "torque": torque,
            "force_percent": self.force_percent,
            "target_position": position,
            "target_open": self._last_target_open,
            "startup_action": "PRESERVE",
            "auto_execute_mode": (
                "PERSISTENT+RUNTIME" if persistent_disabled else "RUNTIME_ONLY"
            ),
        }

    def set_position(self, position: int) -> None:
        if not self._initialized:
            raise AccessoryError("MINI-9C must be initialized before movement")
        if not 0 <= int(position) <= 100:
            raise ValueError("MINI-9C position must be in [0, 100]")
        self._write(
            self.REG_AUTO_EXECUTE,
            self.AUTO_EXECUTE_DISABLE_RUNTIME,
        )
        self._write(self.REG_TARGET_WIDTH, int(position))
        self._write(self.REG_EXECUTE, 1)

    def command_open_state(self, target_open: bool) -> dict:
        target_open = bool(target_open)
        target_position = self.POSITION_OPEN if target_open else self.POSITION_CLOSED
        started = time.monotonic()
        self.set_position(target_position)
        completed = time.monotonic()
        self._last_target_open = target_open
        self._command_sequence += 1
        return {
            "command_id": self._command_sequence,
            "target_open": target_open,
            "target_position": target_position,
            "position_max": self.POSITION_OPEN,
            "force_percent": self.force_percent,
            "target_register": self.REG_TARGET_WIDTH,
            "command_dispatched": True,
            "write_duration_ms": (completed - started) * 1000.0,
        }

    def shutdown(self) -> None:
        if not self._configured:
            return
        try:
            self.arm.close_tool_modbus()
        finally:
            self._configured = False
            self._initialized = False
