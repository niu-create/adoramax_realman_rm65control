"""Safe local job backend for the six-stage XR/RM65 debug console."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .trajectory_stream import TRAJECTORY_PREFIX
from .transforms import pose6_to_transform, validate_rotation
from .xr_diagnostics import axis_capture_minimum_raw_distance_mm


class DebugUiError(ValueError):
    """User-facing request validation error."""


@dataclass(frozen=True)
class ActionSpec:
    action: str
    step: int
    title: str
    report_glob: str | None = None
    marks_step: bool = True


ACTION_SPECS = {
    "xr_motion": ActionSpec(
        "xr_motion", 1, "原始 XR 健康检查", "*_xr_motion_report.json"
    ),
    "xr_stationary": ActionSpec(
        "xr_stationary",
        1,
        "XR 静止 10 秒",
        "*_xr_stationary_report.json",
        marks_step=False,
    ),
    "live_input": ActionSpec(
        "live_input", 2, "正式输入链路镜像", "*_xr_live_input_mirror_report.json"
    ),
    "axis_x": ActionSpec(
        "axis_x", 3, "采集 X+", "*_xr_axis_Xplus.json", marks_step=False
    ),
    "axis_y": ActionSpec(
        "axis_y", 3, "采集 Y+", "*_xr_axis_Yplus.json", marks_step=False
    ),
    "axis_z": ActionSpec(
        "axis_z", 3, "采集 Z+", "*_xr_axis_Zplus.json", marks_step=False
    ),
    "axis_solve": ActionSpec(
        "axis_solve", 3, "拟合三轴旋转", "*_xr_rotation_candidate.json"
    ),
    "shadow_ik": ActionSpec(
        "shadow_ik", 4, "Shadow IK", "*_shadow_live_xr_ik_report.json"
    ),
    "stationary_stop": ActionSpec(
        "stationary_stop", 5, "静止与断流停止", "*_stationary_stop_report.json"
    ),
    "physical_preview": ActionSpec(
        "physical_preview", 6, "实机单轴预览", "live_xr_*.jsonl", marks_step=False
    ),
    "physical_live": ActionSpec(
        "physical_live", 6, "小距离实机运动", "live_xr_*.jsonl"
    ),
}


STEP_TITLES = {
    1: "原始 XR 健康检查",
    2: "正式输入链路镜像",
    3: "三轴方向与标定",
    4: "Shadow IK",
    5: "静止与断流停止",
    6: "小距离实机运动",
}


def _number(
    payload: dict[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = payload.get(key, default)
    if isinstance(raw, bool):
        raise DebugUiError(f"{key} 必须是数字")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise DebugUiError(f"{key} 必须是数字") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise DebugUiError(f"{key} 必须在 {minimum:g}～{maximum:g} 之间")
    return value


def _fmt_number(value: float) -> str:
    return f"{float(value):g}"


def _required_previous_step(action: str) -> int:
    spec = ACTION_SPECS[action]
    return spec.step - 1


def _physical_parameters(payload: dict[str, Any]) -> tuple[str, float]:
    axis = str(payload.get("axis", "X+"))
    if axis not in {"X+", "Y+", "Z+"}:
        raise DebugUiError("axis 只允许 X+、Y+ 或 Z+")
    maximum = 90 if axis == "X+" else 30
    distance = _number(payload, "distance_mm", 10, 1, maximum)
    return axis, distance


def _latest_axis_results(root: Path) -> list[Path]:
    result = []
    for axis in ("Xplus", "Yplus", "Zplus"):
        matches = sorted(
            (root / "xr_debug_logs").glob(f"*_xr_axis_{axis}.json"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        if not matches:
            raise DebugUiError(f"缺少 {axis.replace('plus', '+')} 采集结果")
        result.append(matches[-1])
    return result


def build_command(
    root: Path,
    action: str,
    payload: dict[str, Any],
    *,
    passed_steps: set[int],
    passed_actions: set[str],
    approved_physical_preview: dict[str, Any] | None = None,
    python_executable: str | None = None,
) -> list[str]:
    """Build one strictly whitelisted argv list; no shell text is accepted."""
    if action not in ACTION_SPECS:
        raise DebugUiError("未知调试动作")
    required = _required_previous_step(action)
    if required and required not in passed_steps:
        raise DebugUiError(f"第 {required} 关尚未通过，不能启动此动作")
    python = python_executable or sys.executable

    if action == "xr_motion":
        duration = _number(payload, "duration_s", 20, 5, 120)
        displacement = _number(payload, "min_displacement_mm", 30, 10, 500)
        return [
            python,
            "-u",
            str(root / "debug_xr_capture.py"),
            "motion",
            "--duration-s",
            _fmt_number(duration),
            "--min-travel-mm",
            _fmt_number(displacement),
            "--countdown-s",
            "0",
        ]
    if action == "xr_stationary":
        duration = _number(payload, "stationary_duration_s", 10, 5, 60)
        jitter = _number(payload, "max_p95_jitter_mm", 3, 0.2, 20)
        drift = _number(payload, "max_drift_mm", 5, 0.5, 50)
        return [
            python,
            "-u",
            str(root / "debug_xr_capture.py"),
            "stationary",
            "--duration-s",
            _fmt_number(duration),
            "--max-p95-jitter-mm",
            _fmt_number(jitter),
            "--max-drift-mm",
            _fmt_number(drift),
        ]
    if action == "live_input":
        duration = _number(payload, "duration_s", 30, 10, 180)
        displacement = _number(payload, "min_displacement_mm", 30, 10, 500)
        return [
            python,
            "-u",
            str(root / "debug_live_xr_input.py"),
            "--duration-s",
            _fmt_number(duration),
            "--min-displacement-mm",
            _fmt_number(displacement),
            "--countdown-s",
            "0",
        ]
    if action in {"axis_x", "axis_y", "axis_z"}:
        distance = _number(payload, "expected_distance_mm", 90, 20, 500)
        axis = {"axis_x": "X+", "axis_y": "Y+", "axis_z": "Z+"}[action]
        return [
            python,
            "-u",
            str(root / "debug_xr_capture.py"),
            "axis",
            "--axis",
            axis,
            "--expected-distance-mm",
            _fmt_number(distance),
        ]
    if action == "axis_solve":
        missing = {"axis_x", "axis_y", "axis_z"} - passed_actions
        if missing:
            readable = ", ".join(sorted(item[-1].upper() + "+" for item in missing))
            raise DebugUiError(f"请先完成三轴采集：{readable}")
        results = _latest_axis_results(root)
        return [
            python,
            "-u",
            str(root / "debug_xr_capture.py"),
            "solve",
            *(str(path) for path in results),
        ]
    if action == "shadow_ik":
        duration = _number(payload, "duration_s", 30, 10, 180)
        displacement = _number(payload, "min_displacement_mm", 30, 10, 100)
        return [
            python,
            "-u",
            str(root / "shadow_live_xr_ik.py"),
            "--duration-s",
            _fmt_number(duration),
            "--min-displacement-mm",
            _fmt_number(displacement),
            "--countdown-s",
            "5",
        ]
    if action == "stationary_stop":
        hold_s = _number(payload, "hold_s", 3, 1, 10)
        confirmation = str(payload.get("confirmation", ""))
        expected = "STATIONARY STOP"
        if confirmation != expected:
            raise DebugUiError(f"确认口令必须完整输入：{expected}")
        return [
            python,
            "-u",
            str(root / "debug_stationary_stop.py"),
            "--hold-s",
            _fmt_number(hold_s),
            "--countdown-s",
            "5",
            "--execute",
            "--confirm",
            expected,
        ]
    if action in {"physical_preview", "physical_live"}:
        if action == "physical_live" and "physical_preview" not in passed_actions:
            raise DebugUiError("请先完成第 6 关的只读端点预览")
        axis, distance = _physical_parameters(payload)
        if action == "physical_live":
            approved = approved_physical_preview or {}
            approved_axis = str(approved.get("axis", ""))
            try:
                approved_distance = float(approved.get("distance_mm"))
            except (TypeError, ValueError):
                approved_distance = float("nan")
            if approved_axis != axis or not math.isclose(
                approved_distance, distance, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise DebugUiError(
                    "实机轴和距离必须与刚通过的只读预览完全一致；"
                    "参数变化后请重新预览"
                )
        command = [
            python,
            "-u",
            str(root / "commission_live_xr_axis.py"),
            axis,
            "--distance-mm",
            _fmt_number(distance),
        ]
        if action == "physical_preview":
            return command
        expected = f"LIVE XR {axis} {distance:.3f}MM"
        if str(payload.get("confirmation", "")) != expected:
            raise DebugUiError(f"确认口令必须完整输入：{expected}")
        return [*command, "--execute", "--confirm", expected]
    raise AssertionError(action)


def validate_action_result(action: str, report_path: Path | None) -> tuple[bool, str]:
    """Apply UI-level acceptance checks after a process exits successfully."""
    if report_path is None or report_path.suffix != ".json":
        return True, "进程正常结束"
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"结果文件无法读取：{exc}"
    if action == "xr_stationary":
        jitter = payload.get("jitter") or {}
        p95_raw = jitter.get("radial_p95_mm")
        drift_raw = jitter.get("start_to_end_drift_mm")
        p95 = float("nan") if p95_raw is None else float(p95_raw)
        drift = float("nan") if drift_raw is None else float(drift_raw)
        message = f"静止 P95={p95:.3f} mm，首尾漂移={drift:.3f} mm"
        return bool(payload.get("passed")), message
    if action == "stationary_stop":
        threshold = float(payload.get("motion_start_threshold_mm", float("nan")))
        displacement = float(payload.get("max_xr_displacement_mm", float("nan")))
        latency = payload.get(
            "last_xr_advance_to_slow_stop_call_s",
            payload.get("disconnect_to_slow_stop_s"),
        )
        if bool(payload.get("passed")):
            latency_ms = float(latency) * 1000.0
            return True, (
                f"静止漂浮={displacement:.3f} mm，"
                f"末帧 XR 至慢停指令={latency_ms:.1f} ms"
            )
        error = str(payload.get("error") or "")
        if "stationary-stop gate blocked a joint target" in error:
            limit_text = "静止死区" if not math.isfinite(threshold) else f"{threshold:.1f} mm 静止死区"
            return False, f"手柄漂浮超过{limit_text}，运动目标已被硬门拦截"
        if error:
            return False, error
        failed_checks = [
            name for name, passed in (payload.get("checks") or {}).items() if not passed
        ]
        detail = "、".join(failed_checks) if failed_checks else "报告未通过"
        return False, f"第五阶段失败：{detail}"
    if "passed" in payload and not bool(payload["passed"]):
        return False, "报告中的 passed=false"
    if action == "axis_solve":
        rms = float(payload.get("rms_angular_error_deg", float("inf")))
        determinant = float(payload.get("determinant", float("nan")))
        if not math.isfinite(rms) or rms > 8.0:
            return False, f"三轴角度 RMS {rms:.3f}°，超过 8°"
        if not math.isfinite(determinant) or abs(determinant - 1.0) > 0.01:
            return False, f"旋转矩阵行列式 {determinant:.6f}，不是右手系"
        scale = float(payload.get("candidate_position_scale", float("nan")))
        deviation = float(payload.get("maximum_scale_deviation_pct", float("inf")))
        if not math.isfinite(scale) or not math.isfinite(deviation):
            return False, "三轴结果缺少尺度一致性数据"
        if deviation > 15.0:
            return False, f"三轴尺度离散={deviation:.1f}%，超过 15%"
        if not bool(payload.get("candidate_applied_to_config")):
            rotation_error = float(
                payload.get("configured_rotation_error_deg", float("inf"))
            )
            return False, (
                f"候选尚未应用：尺度={scale:.3f}，"
                f"旋转差={rotation_error:.3f}°"
            )
        return True, f"RMS={rms:.3f}°，det={determinant:.6f}，尺度={scale:.3f}"
    if action in {"axis_x", "axis_y", "axis_z"}:
        distance = float(payload.get("raw_distance_mm", 0.0))
        expected = float(payload.get("expected_distance_mm", 0.0))
        minimum = axis_capture_minimum_raw_distance_mm(expected)
        scale = float(payload.get("position_scale_candidate", float("nan")))
        passed = distance >= minimum and math.isfinite(scale) and scale > 0.0
        return (
            passed,
            f"尺子={expected:.1f} mm，SDK={distance:.3f} mm，候选尺度={scale:.3f}",
        )
    return True, "报告验收通过"


def subprocess_failure_message(output: str, return_code: int) -> str:
    if "RM65 IK failed" in output:
        return (
            "官方 RM65 IK 无解：当前姿态下该轴/距离不可达；"
            "请选择其他正轴或调整机械臂起始姿态后重新预览"
        )
    match = re.search(
        r"main-axis request exceeded bound: actual=([0-9.]+) mm, limit=([0-9.]+) mm",
        output,
    )
    if match:
        return (
            f"手柄主轴位移 {match.group(1)} mm 超过 {match.group(2)} mm 安全边界；"
            "移动过快，机械臂尚未来得及跟随。请至少用 3 秒缓慢到达目标，"
            "停住 1 秒，再用 3 秒缓慢返回 A 点"
        )
    if "cross-axis request exceeded bound" in output:
        return "手柄横向偏移超过安全边界；请沿所选单轴重新测试"
    if "reverse request exceeded bound" in output:
        return "手柄移动到了所选轴的反方向并超过安全边界"
    return f"进程退出码 {return_code}"


class DebugJobManager:
    """Run at most one child process and retain its observable state."""

    MAX_OUTPUT_CHARS = 500_000

    def __init__(
        self,
        root: Path,
        state_path: Path | None = None,
        *,
        trajectory_rotation_work_xr=None,
    ):
        self.root = root.resolve()
        self.state_path = state_path or self.root / ".debug_ui_state.json"
        self._trajectory_calibration_rms_deg: float | None = None
        self._trajectory_rotation_work_xr = self._resolve_trajectory_rotation(
            trajectory_rotation_work_xr
        )
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._job: dict[str, Any] | None = None
        self._output = ""
        self._trajectory: list[dict[str, Any]] = []
        self._trajectory_origin_m: list[float] | None = None
        self._trajectory_previous_m: list[float] | None = None
        self._trajectory_min_m: list[float] | None = None
        self._trajectory_max_m: list[float] | None = None
        self._trajectory_first_host_ns: int | None = None
        self._trajectory_smoothed_m: np.ndarray | None = None
        self._trajectory_last_source_ns: int | None = None
        self._trajectory_invalid_count = 0
        self._trajectory_metrics = self._empty_trajectory_metrics()
        self._state = self._load_state()

    def _resolve_trajectory_rotation(self, supplied) -> np.ndarray:
        if supplied is not None:
            return validate_rotation(supplied)
        config_path = self.root / "config" / "rm65_192_168_1_19.json"
        if not config_path.is_file():
            # Unit tests and embedders may use a minimal temporary root.
            return np.eye(3, dtype=float)
        config = load_config(config_path)
        transform_base_work = pose6_to_transform(config.work_frame.pose_m_rad)
        self._trajectory_calibration_rms_deg = config.calibration_rms_error_deg
        return validate_rotation(
            transform_base_work[:3, :3].T @ config.rotation_base_xr
        )

    def _empty_trajectory_metrics(self) -> dict[str, Any]:
        return {
            "source": "official_PXREADeviceStateJson",
            "coordinate_frame": "416table",
            "calibration_rms_error_deg": self._trajectory_calibration_rms_deg,
            "sample_count": 0,
            "current_delta_mm": [0.0, 0.0, 0.0],
            "current_distance_mm": 0.0,
            "peak_displacement_mm": 0.0,
            "cumulative_path_mm": 0.0,
            "effective_path_mm": 0.0,
            "axis_span_mm": [0.0, 0.0, 0.0],
            "spatial_extent_mm": 0.0,
            "elapsed_s": 0.0,
            "smoothed_current_delta_mm": [0.0, 0.0, 0.0],
            "jitter_p95_mm": None,
            "jitter_rms_mm": None,
            "jitter_max_mm": None,
            "drift_mm": None,
            "invalid_sample_count": 0,
            "tracking_valid": True,
            "grip": None,
            "trigger": None,
            "phase": "idle",
        }

    def _reset_trajectory(self) -> None:
        self._trajectory = []
        self._trajectory_origin_m = None
        self._trajectory_previous_m = None
        self._trajectory_min_m = None
        self._trajectory_max_m = None
        self._trajectory_first_host_ns = None
        self._trajectory_smoothed_m = None
        self._trajectory_last_source_ns = None
        self._trajectory_invalid_count = 0
        self._trajectory_metrics = self._empty_trajectory_metrics()

    def _load_state(self) -> dict[str, Any]:
        default = {
            "passed_steps": [],
            "passed_actions": [],
            "physical_preview": None,
            "history": [],
        }
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default
        passed_steps = []
        for step in payload.get("passed_steps", []):
            try:
                number = int(step)
            except (TypeError, ValueError):
                continue
            if number in range(1, 7):
                passed_steps.append(number)
        passed_actions = [
            action for action in payload.get("passed_actions", []) if action in ACTION_SPECS
        ]
        preview = payload.get("physical_preview")
        if not isinstance(preview, dict) or "physical_preview" not in passed_actions:
            preview = None
        else:
            try:
                preview_axis, preview_distance = _physical_parameters(preview)
                preview = {"axis": preview_axis, "distance_mm": preview_distance}
            except DebugUiError:
                preview = None
                passed_actions = [
                    action for action in passed_actions if action != "physical_preview"
                ]
        return {
            "passed_steps": passed_steps,
            "passed_actions": passed_actions,
            "physical_preview": preview,
            "history": list(payload.get("history", []))[-50:],
        }

    def _persist(self) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def public_state(self) -> dict[str, Any]:
        with self._lock:
            passed_steps = set(self._state["passed_steps"])
            active_step = None if self._job is None else self._job["step"]
            steps = []
            for number, title in STEP_TITLES.items():
                if number in passed_steps:
                    status = "passed"
                elif active_step == number and self._job and self._job["status"] == "running":
                    status = "running"
                elif number == 1 or number - 1 in passed_steps:
                    status = "ready"
                else:
                    status = "locked"
                steps.append({"number": number, "title": title, "status": status})
            return {
                "steps": steps,
                "passed_steps": sorted(passed_steps),
                "passed_actions": sorted(set(self._state["passed_actions"])),
                "physical_preview": self._state.get("physical_preview"),
                "job": None if self._job is None else dict(self._job),
                "history": list(reversed(self._state["history"][-10:])),
            }

    def output(self, offset: int) -> dict[str, Any]:
        with self._lock:
            offset = max(0, min(int(offset), len(self._output)))
            return {
                "offset": len(self._output),
                "chunk": self._output[offset:],
                "job": None if self._job is None else dict(self._job),
            }

    def trajectory(self, offset: int) -> dict[str, Any]:
        with self._lock:
            self._update_stationary_metrics_locked()
            offset = max(0, min(int(offset), len(self._trajectory)))
            return {
                "offset": len(self._trajectory),
                "points": list(self._trajectory[offset:]),
                "metrics": dict(self._trajectory_metrics),
            }

    def _update_stationary_metrics_locked(self) -> None:
        if (
            not self._trajectory
            or self._trajectory_metrics.get("phase") != "stationary_raw"
        ):
            return
        positions = np.asarray(
            [point["xyz_mm"] for point in self._trajectory], dtype=float
        )
        center = np.median(positions, axis=0)
        radial_mm = np.linalg.norm(positions - center, axis=1)
        source_ns = np.asarray(
            [int(point.get("timestamp_ns") or 0) for point in self._trajectory],
            dtype=np.int64,
        )
        if len(positions) == 1:
            drift_mm = 0.0
        elif np.all(source_ns > 0):
            start = positions[source_ns <= source_ns[0] + 1_000_000_000]
            end = positions[source_ns >= source_ns[-1] - 1_000_000_000]
            drift_mm = float(np.linalg.norm(np.median(end, axis=0) - np.median(start, axis=0)))
        else:
            window = max(1, len(positions) // 10)
            drift_mm = float(
                np.linalg.norm(
                    np.median(positions[-window:], axis=0)
                    - np.median(positions[:window], axis=0)
                )
            )
        self._trajectory_metrics.update(
            {
                "jitter_p95_mm": float(np.percentile(radial_mm, 95.0)),
                "jitter_rms_mm": float(np.sqrt(np.mean(radial_mm**2))),
                "jitter_max_mm": float(np.max(radial_mm)),
                "drift_mm": drift_mm,
            }
        )

    def start(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise DebugUiError("已有调试任务正在运行")
            passed_steps = set(self._state["passed_steps"])
            passed_actions = set(self._state["passed_actions"])
            command = build_command(
                self.root,
                action,
                payload,
                passed_steps=passed_steps,
                passed_actions=passed_actions,
                approved_physical_preview=self._state.get("physical_preview"),
            )
            physical_parameters = None
            if action in {"physical_preview", "physical_live"}:
                axis, distance = _physical_parameters(payload)
                physical_parameters = {"axis": axis, "distance_mm": distance}
            if action == "physical_preview":
                # A new preview attempt revokes the previous approval until
                # this exact axis and distance complete successfully.
                actions = set(self._state["passed_actions"])
                actions.discard("physical_preview")
                self._state["passed_actions"] = sorted(actions)
                self._state["physical_preview"] = None
                self._persist()
            spec = ACTION_SPECS[action]
            started_wall_ns = time.time_ns()
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=self.root,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=False,
                start_new_session=True,
            )
            self._process = process
            self._output = ""
            self._reset_trajectory()
            self._job = {
                "action": action,
                "step": spec.step,
                "title": spec.title,
                "status": "running",
                "pid": process.pid,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "return_code": None,
                "outcome": None,
                "message": (
                    "正在连接 RM65；请戴好头显并先完全松开 Grip"
                    if action == "stationary_stop"
                    else "任务已启动"
                ),
                "report_path": None,
                # Axis capture enables Enter only after the child has printed
                # an A/B prompt.  This prevents queued double-clicks from
                # accidentally accepting the next point.
                "can_send_enter": False,
                "enter_phase": None,
                "last_prompt_rejected": False,
                "parameters": physical_parameters,
            }
            thread = threading.Thread(
                target=self._read_process,
                args=(process, action, started_wall_ns),
                name="xr-debug-ui-job-reader",
                daemon=True,
            )
            thread.start()
            return dict(self._job)

    def _find_report(self, action: str, started_wall_ns: int) -> Path | None:
        pattern = ACTION_SPECS[action].report_glob
        if pattern is None:
            return None
        folders = [self.root / "xr_debug_logs", self.root / "logs" / "live_xr"]
        candidates = []
        for folder in folders:
            if folder.exists():
                candidates.extend(folder.glob(pattern))
        fresh = [
            path
            for path in candidates
            if path.is_file() and path.stat().st_mtime_ns >= started_wall_ns - 1_000_000_000
        ]
        return max(fresh, key=lambda path: path.stat().st_mtime_ns, default=None)

    def _append_output(self, text: str) -> None:
        self._output += text
        if len(self._output) > self.MAX_OUTPUT_CHARS:
            removed = len(self._output) - self.MAX_OUTPUT_CHARS
            self._output = f"[较早输出已截去 {removed} 字符]\n" + self._output[-self.MAX_OUTPUT_CHARS :]

    def _update_axis_prompt_locked(self, action: str, line: str) -> None:
        if action not in {"axis_x", "axis_y", "axis_z"} or self._job is None:
            return
        if "[稳定窗口拒绝]" in line:
            phase = self._job.get("enter_phase") or "当前"
            self._job["can_send_enter"] = False
            self._job["last_prompt_rejected"] = True
            self._job["message"] = f"{phase} 点不够稳定，正在准备重试"
            return
        if "[B点位移过小]" in line:
            self._job["can_send_enter"] = False
            self._job["enter_phase"] = "B"
            self._job["last_prompt_rejected"] = True
            detail = line[line.index("[B点位移过小]") :].strip()
            self._job["message"] = detail
            return
        if "[A点已采集]" in line:
            self._job["can_send_enter"] = False
            self._job["enter_phase"] = "B"
            self._job["last_prompt_rejected"] = False
            self._job["message"] = "A 点已采集；请移动到 B 点并保持稳定"
            return
        for marker, phase in (("[A点]", "A"), ("[B点]", "B")):
            if marker not in line:
                continue
            retry = bool(self._job.get("last_prompt_rejected"))
            self._job["can_send_enter"] = True
            self._job["enter_phase"] = phase
            self._job["last_prompt_rejected"] = False
            prefix = "上次采集未通过；" if retry else ""
            self._job["message"] = (
                f"{prefix}保持 {phase} 点不动，然后点击“采集 {phase} 点”"
            )
            return

    def _update_job_progress_locked(self, action: str, line: str) -> None:
        if (
            action != "stationary_stop"
            or self._job is None
            or self._job.get("action") != action
        ):
            return
        progress_markers = (
            ("[STATIONARY PREVIEW]", "正在连接 RM65 并核对安全状态"),
            ("[准备]", "准备中：戴好头显、唤醒手柄，并保持 Grip 松开"),
            ("[操作]", "XR 已就绪：现在按住 Grip，保持手柄静止"),
            ("[SESSION]", "Grip 起点已记录：保持手柄静止，不要松开"),
            ("[STATIONARY PASS]", "静止保持通过：正在模拟 XR 断流并验证慢停"),
            ("[REPORT]", "测试已结束：正在验收报告"),
        )
        if "[倒计时]" in line:
            remaining = line.split("[倒计时]", 1)[1].strip()
            self._job["message"] = f"{remaining} 秒后开始读取 XR；请保持 Grip 松开"
            return
        if "[STATIONARY FAIL]" in line:
            detail = line.split("[STATIONARY FAIL]", 1)[1].strip()
            self._job["message"] = f"启动失败：{detail}"
            return
        for marker, message in progress_markers:
            if marker in line:
                self._job["message"] = message
                return

    def _ingest_trajectory_line(self, payload_text: str) -> None:
        try:
            payload = json.loads(payload_text)
            position = [float(value) for value in payload["position_m"]]
            if len(position) != 3 or not all(math.isfinite(value) for value in position):
                raise ValueError("invalid position")
            grip = payload.get("grip")
            trigger = payload.get("trigger")
            if grip is not None:
                grip = float(grip)
                if not math.isfinite(grip):
                    raise ValueError("invalid grip")
            if trigger is not None:
                trigger = float(trigger)
                if not math.isfinite(trigger):
                    raise ValueError("invalid trigger")
            host_ns = int(payload.get("received_monotonic_ns") or 0)
            source_ns = int(payload.get("timestamp_ns") or 0)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._append_output(f"[TRAJECTORY EVENT ERROR] {exc}\n")
            return

        position_array = np.asarray(position, dtype=float)
        if np.linalg.norm(position_array) <= 1.0e-4:
            self._trajectory_invalid_count += 1
            self._trajectory_metrics.update(
                {
                    "invalid_sample_count": self._trajectory_invalid_count,
                    "tracking_valid": False,
                    "phase": "tracking_invalid",
                }
            )
            if self._trajectory_invalid_count == 1:
                self._append_output(
                    "[XR TRACKING] 收到全零位置，3D 轨迹已忽略该无效样本。\n"
                )
            return

        if self._trajectory_origin_m is None:
            self._trajectory_origin_m = position.copy()
            self._trajectory_previous_m = position.copy()
            self._trajectory_min_m = [0.0, 0.0, 0.0]
            self._trajectory_max_m = [0.0, 0.0, 0.0]
            self._trajectory_first_host_ns = host_ns if host_ns > 0 else None

        assert self._trajectory_previous_m is not None
        assert self._trajectory_min_m is not None
        assert self._trajectory_max_m is not None
        raw_delta_m = np.asarray(
            [
                position[index] - self._trajectory_origin_m[index]
                for index in range(3)
            ],
            dtype=float,
        )
        delta_m = self._trajectory_rotation_work_xr @ raw_delta_m
        if self._trajectory_smoothed_m is None:
            self._trajectory_smoothed_m = delta_m.copy()
        else:
            dt_s = 1.0 / 90.0
            if source_ns > 0 and self._trajectory_last_source_ns is not None:
                dt_s = min(
                    0.1,
                    max(0.001, (source_ns - self._trajectory_last_source_ns) / 1.0e9),
                )
            alpha = 1.0 - math.exp(-2.0 * math.pi * 3.0 * dt_s)
            self._trajectory_smoothed_m += alpha * (
                delta_m - self._trajectory_smoothed_m
            )
        if source_ns > 0:
            self._trajectory_last_source_ns = source_ns
        raw_step_m = np.asarray(position) - np.asarray(self._trajectory_previous_m)
        step_m = float(
            np.linalg.norm(self._trajectory_rotation_work_xr @ raw_step_m)
        )
        distance_mm = float(np.linalg.norm(delta_m) * 1000.0)
        for index in range(3):
            self._trajectory_min_m[index] = min(
                self._trajectory_min_m[index], float(delta_m[index])
            )
            self._trajectory_max_m[index] = max(
                self._trajectory_max_m[index], float(delta_m[index])
            )
        spans_mm = [
            (self._trajectory_max_m[index] - self._trajectory_min_m[index]) * 1000.0
            for index in range(3)
        ]
        elapsed_s = 0.0
        if self._trajectory_first_host_ns is not None and host_ns > 0:
            elapsed_s = max(0.0, (host_ns - self._trajectory_first_host_ns) / 1.0e9)
        phase = str(payload.get("phase", "stream"))
        point = {
            "xyz_mm": [float(value) * 1000.0 for value in delta_m],
            "smoothed_xyz_mm": [
                float(value) * 1000.0 for value in self._trajectory_smoothed_m
            ],
            "raw_xr_xyz_mm": [float(value) * 1000.0 for value in raw_delta_m],
            "grip": grip,
            "trigger": trigger,
            "timestamp_ns": payload.get("timestamp_ns"),
            "sequence": payload.get("sequence"),
            "phase": phase,
        }
        self._trajectory.append(point)
        self._trajectory_previous_m = position.copy()
        self._trajectory_metrics.update(
            {
                "sample_count": len(self._trajectory),
                "current_delta_mm": point["xyz_mm"],
                "smoothed_current_delta_mm": point["smoothed_xyz_mm"],
                "current_distance_mm": distance_mm,
                "peak_displacement_mm": max(
                    float(self._trajectory_metrics["peak_displacement_mm"]),
                    distance_mm,
                ),
                "cumulative_path_mm": (
                    float(self._trajectory_metrics["cumulative_path_mm"])
                    + step_m * 1000.0
                ),
                # Stage 5 intentionally asks the operator to hold still. Keep
                # raw cumulative distance for diagnostics, but do not present
                # normal hand float as effective motion.
                "effective_path_mm": (
                    float(self._trajectory_metrics["effective_path_mm"])
                    + (0.0 if phase == "stationary_hold" else step_m * 1000.0)
                ),
                "axis_span_mm": spans_mm,
                "spatial_extent_mm": math.sqrt(
                    sum(value * value for value in spans_mm)
                ),
                "elapsed_s": elapsed_s,
                "invalid_sample_count": self._trajectory_invalid_count,
                "tracking_valid": True,
                "grip": grip,
                "trigger": trigger,
                "phase": point["phase"],
            }
        )

    def _read_process(
        self,
        process: subprocess.Popen[str],
        action: str,
        started_wall_ns: int,
    ) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            with self._lock:
                if line.startswith(TRAJECTORY_PREFIX):
                    self._ingest_trajectory_line(line[len(TRAJECTORY_PREFIX) :])
                else:
                    self._update_axis_prompt_locked(action, line)
                    self._update_job_progress_locked(action, line)
                    self._append_output(line)
        return_code = process.wait()
        report = self._find_report(action, started_wall_ns)
        passed = return_code == 0
        if passed:
            passed, message = validate_action_result(action, report)
        else:
            if report is not None and report.suffix == ".json":
                _, message = validate_action_result(action, report)
            else:
                message = subprocess_failure_message(self._output, return_code)
        with self._lock:
            if self._job is None or self._job["action"] != action:
                return
            stopped = bool(self._job.get("stop_requested"))
            outcome = "stopped" if stopped else ("passed" if passed else "failed")
            if stopped:
                message = "任务已由操作者停止"
            parameters = self._job.get("parameters")
            if passed and action == "physical_preview" and isinstance(parameters, dict):
                message = (
                    f"只读预览通过：{parameters['axis']} / "
                    f"{float(parameters['distance_mm']):g} mm；"
                    "保持参数不变并输入页面口令后再执行实机测试"
                )
            self._job.update(
                {
                    "status": "finished",
                    "return_code": return_code,
                    "outcome": outcome,
                    "message": message,
                    "report_path": (
                        None if report is None else str(report.relative_to(self.root))
                    ),
                    "can_send_enter": False,
                    "enter_phase": None,
                }
            )
            spec = ACTION_SPECS[action]
            if passed and not stopped:
                actions = set(self._state["passed_actions"])
                actions.add(action)
                self._state["passed_actions"] = sorted(actions)
                if action == "physical_preview" and isinstance(parameters, dict):
                    self._state["physical_preview"] = dict(parameters)
                if spec.marks_step:
                    steps = set(self._state["passed_steps"])
                    steps.add(spec.step)
                    self._state["passed_steps"] = sorted(steps)
            elif spec.marks_step:
                self._invalidate_from(spec.step)
            elif action in {"axis_x", "axis_y", "axis_z"}:
                self._state["passed_actions"] = [
                    current
                    for current in self._state["passed_actions"]
                    if current != action
                ]
                self._invalidate_from(3)
            entry = {
                "action": action,
                "title": spec.title,
                "step": spec.step,
                "outcome": outcome,
                "message": message,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "report_path": self._job["report_path"],
                "parameters": parameters,
            }
            self._state["history"].append(entry)
            self._state["history"] = self._state["history"][-50:]
            self._persist()
            self._process = None

    def _invalidate_from(self, step: int) -> None:
        self._state["passed_steps"] = [
            current for current in self._state["passed_steps"] if current < step
        ]
        self._state["passed_actions"] = [
            action
            for action in self._state["passed_actions"]
            if ACTION_SPECS[action].step < step
        ]
        if step <= 6:
            self._state["physical_preview"] = None

    def send_enter(self) -> None:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                raise DebugUiError("当前没有等待输入的任务")
            if self._job is None or not self._job.get("can_send_enter"):
                raise DebugUiError("当前没有可采集的 A/B 点，请等待页面提示")
            phase = self._job.get("enter_phase") or "当前"
            self._job["can_send_enter"] = False
            self._job["message"] = f"正在采集 {phase} 点的稳定窗口，请保持不动"
            assert self._process.stdin is not None
            self._process.stdin.write("\n")
            self._process.stdin.flush()

    def stop(self) -> None:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                raise DebugUiError("当前没有运行中的任务")
            assert self._job is not None
            self._job["stop_requested"] = True
            self._job["message"] = "正在进行安全中断"
            # Let Python unwind every script's finally blocks. This matters in
            # the physical stage, whose cleanup sends the RM65 slow-stop.
            self._process.send_signal(signal.SIGINT)

    def close(self) -> None:
        """Safely interrupt an active child when the UI server exits."""
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        self.stop()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            # Do not escalate to SIGKILL: the child keeps ownership of its
            # cleanup and can still complete a slow-stop.
            pass

    def reset(self, confirmation: str) -> None:
        with self._lock:
            if confirmation != "RESET SIX STEPS":
                raise DebugUiError("重置口令不正确")
            if self._process is not None and self._process.poll() is None:
                raise DebugUiError("任务运行时不能重置")
            self._state = {
                "passed_steps": [],
                "passed_actions": [],
                "physical_preview": None,
                "history": [],
            }
            self._job = None
            self._output = ""
            self._reset_trajectory()
            self._persist()


_REPORT_PATH_PATTERN = re.compile(r"^[A-Za-z0-9_+./-]+$")


def resolve_report_path(root: Path, relative: str) -> Path:
    if not relative or not _REPORT_PATH_PATTERN.fullmatch(relative):
        raise DebugUiError("报告路径非法")
    candidate = (root / relative).resolve()
    allowed_roots = [(root / "xr_debug_logs").resolve(), (root / "logs").resolve()]
    if not any(candidate == base or base in candidate.parents for base in allowed_roots):
        raise DebugUiError("报告不在允许目录中")
    if candidate.suffix not in {".json", ".jsonl"} or not candidate.is_file():
        raise DebugUiError("报告不存在或格式不允许")
    return candidate
