import time

import numpy as np

from .frame_builder import CollectionFrameBuilder
from .models import SourceBundle
from .recovery import CheckpointError


class DualRM65CollectionRuntime:
    """Observe a running dual controller and manage 30 Hz episodes."""

    def __init__(
        self,
        *,
        controller,
        cameras,
        gripper_sampler,
        episodes,
        audit,
        frame_builder=None,
        dataset_fps: float = 30.0,
        audit_hz: float = 50.0,
        max_episodes=None,
        heartbeat=None,
        max_dataset_gap_factor: float = 3.0,
        max_recorded_joint_step_deg: float = 10.0,
        camera_wait_timeout_s: float = 0.25,
        grip_warmup_s: float = 0.0,
        wait_for_grip_follow: bool = False,
        discard_invalid_on_save: bool = False,
        reject_on_cadence_violation: bool = True,
        require_fresh_camera_frames: bool = True,
        joystick_threshold: float = 0.65,
        joystick_release_threshold: float = 0.35,
    ):
        if dataset_fps <= 0 or audit_hz <= 0:
            raise ValueError("dataset_fps and audit_hz must be positive")
        if max_dataset_gap_factor <= 1.0:
            raise ValueError("max_dataset_gap_factor must exceed 1")
        self.controller = controller
        self.cameras = dict(cameras)
        self.gripper_sampler = gripper_sampler
        self.episodes = episodes
        self.audit = audit
        self.frame_builder = frame_builder or CollectionFrameBuilder()
        self.dataset_fps = float(dataset_fps)
        self.audit_hz = float(audit_hz)
        if max_episodes is not None and int(max_episodes) <= 0:
            raise ValueError("max_episodes must be positive")
        self.max_episodes = None if max_episodes is None else int(max_episodes)
        self.saved_episodes = 0
        self.stop_requested = False
        self.heartbeat = heartbeat or (lambda: None)
        self.max_dataset_gap_s = (
            float(max_dataset_gap_factor) / self.dataset_fps
        )
        if float(max_recorded_joint_step_deg) <= 0.0:
            raise ValueError("max_recorded_joint_step_deg must be positive")
        self.max_recorded_joint_step_rad = np.deg2rad(
            float(max_recorded_joint_step_deg)
        )
        if float(camera_wait_timeout_s) <= 0.0:
            raise ValueError("camera_wait_timeout_s must be positive")
        self.camera_wait_timeout_s = float(camera_wait_timeout_s)
        self._last_camera_timestamps = {}
        self._xr_recovery_failure_baseline = {}
        self._xr_pending_arms = set()
        self._last_recording_sample_time = None
        self._last_recording_observation_state = None
        self._last_recording_action = None
        if not np.isfinite(grip_warmup_s) or grip_warmup_s < 0:
            raise ValueError("grip_warmup_s must be finite and nonnegative")
        self.grip_warmup_s = float(grip_warmup_s)
        self.wait_for_grip_follow = bool(wait_for_grip_follow or self.grip_warmup_s)
        # Starting a recording after Grip positioning can straddle one or two
        # scheduler ticks. Ignore only that startup settle window; strict
        # cadence checks remain enabled for the rest of the episode.
        self._cadence_warmup_samples = 2 if self.wait_for_grip_follow else 0
        self._recording_sample_count = 0
        self._start_pending = False
        self._grip_warmup_started_at = None
        self._warmup_display = None
        self.discard_invalid_on_save = bool(discard_invalid_on_save)
        self.reject_on_cadence_violation = bool(reject_on_cadence_violation)
        self.require_fresh_camera_frames = bool(require_fresh_camera_frames)
        self._last_rejection = None
        self._last_save_result = "尚未保存"
        self._last_save_message = ""
        if not 0.0 < joystick_release_threshold < joystick_threshold <= 1.0:
            raise ValueError("joystick thresholds must satisfy 0 < release < threshold <= 1")
        self.joystick_threshold = float(joystick_threshold)
        self.joystick_release_threshold = float(joystick_release_threshold)
        self._joystick_action_latched = None

    def _poll_joystick_action(self):
        """Return one edge-triggered action from the right PICO thumbstick."""
        controllers = getattr(self.controller, "controllers", {})
        right = controllers.get("right_arm")
        get_status = getattr(right, "get_teleop_status", None)
        if not callable(get_status):
            return None
        status = get_status("right_arm") or {}
        if not status.get("stream_fresh", False):
            return None
        axis = status.get("axis_xy", (0.0, 0.0))
        try:
            x, y = float(axis[0]), float(axis[1])
        except (TypeError, ValueError, IndexError):
            return None
        if max(abs(x), abs(y)) <= self.joystick_release_threshold:
            self._joystick_action_latched = None
            return None
        action = None
        # PICO logs from this collector show a forward push as axisY=+1.
        if y >= self.joystick_threshold and abs(y) >= abs(x):
            action = "ENTER"  # forward: prepare/start
        elif x >= self.joystick_threshold and abs(x) > abs(y):
            action = "RIGHT"  # right: save
        elif x <= -self.joystick_threshold and abs(x) > abs(y):
            action = "LEFT"   # left: discard
        if action is None or self._joystick_action_latched is not None:
            return None
        self._joystick_action_latched = action
        return action

    @property
    def operator_notice(self):
        if self.episodes.recording:
            return getattr(self.episodes, "invalid_reason", None) or ""
        return self._last_rejection or ""

    @property
    def operator_display(self):
        """Compact status shared by the desktop preview and VR camera view."""
        state = self.operator_state
        if (self.stop_requested and self.max_episodes is not None
                and self.saved_episodes >= self.max_episodes):
            state = "FINISHED"
        instructions = {
            "READY": "松开 Grip；回车或右手摇杆向前准备",
            "WAIT GRIP": "已准备；按住 Grip 定位，完成后自动录制",
            "WAIT FOLLOW": "正在同步右臂，等待进入跟随并自动录制",
            "RECORDING": "录制中；右键/摇杆向右保存，左键/摇杆向左作废",
            "INVALID": "本条数据异常；保存键将剔除本条",
            "REJECTED": "本条未保存；回车或摇杆向前重新准备",
            "REC WARN": "录制有质量警告，请检查原因",
            "FINISHED": "本次采集已完成，保存数据已计入数据集",
        }
        if state.startswith("POSITIONING"):
            instruction = "正在定位；保持 Grip，定位期间不记录"
        else:
            instruction = instructions.get(state, "请查看采集终端状态")
        notice = self.operator_notice
        if notice:
            validity = "无效"
        elif state in ("RECORDING", "REC WARN"):
            validity = "待校验"
        elif self._last_save_result == "保存成功":
            validity = "有效"
        elif self._last_save_result == "已作废":
            validity = "已作废"
        else:
            validity = "待采集"
        return {
            "state": state,
            "saved": self.saved_episodes,
            "target": self.max_episodes,
            "instruction": instruction,
            "error": self._quality_reason(notice) if notice else "",
            "validity": validity,
            "save_result": self._last_save_result,
            "save_message": self._last_save_message,
        }

    @staticmethod
    def _quality_reason(reason):
        categories = (
            ("cadence", "采样间隔过大，数据不连续"),
            ("joint continuity", "相邻帧关节变化超过阈值"),
            ("camera", "相机数据缺失、过期或不新鲜"),
            ("stale", "机械臂或夹爪反馈过期"),
            ("teleop session stopped", "Grip 仍按住时机械臂控制会话已停止"),
            ("Home", "录制中发生回 Home 动作"),
            ("frame_write_failed", "采集数据写入失败"),
        )
        for keyword, description in categories:
            if keyword in str(reason):
                return f"{description}；详情：{reason}"
        return f"数据校验未通过；详情：{reason}"

    @property
    def operator_state(self):
        if self.episodes.recording:
            return (
                ("REC WARN" if getattr(self.episodes, "preserve_invalid", False) else "INVALID")
                if getattr(self.episodes, "invalid_reason", None) is not None
                else "RECORDING"
            )
        if not self._start_pending:
            return "REJECTED" if self._last_rejection else "READY"
        if self._grip_warmup_started_at is None:
            return "WAIT GRIP"
        remaining = self.grip_warmup_s - (
            time.monotonic() - self._grip_warmup_started_at
        )
        return f"POSITIONING {int(np.ceil(remaining))}" if remaining > 0 else "WAIT FOLLOW"

    def _cancel_pending_start(self, reason):
        if self._start_pending:
            self.audit.write({"event": "episode_prepare_cancelled", "reason": reason})
        self._start_pending = False
        self._grip_warmup_started_at = None
        self._warmup_display = None

    def _camera_health_errors(self):
        errors = []
        for camera in self.cameras.values():
            health_error = getattr(camera, "health_error", None)
            if callable(health_error):
                error = health_error(max_age_s=0.5)
                if error:
                    errors.append(error)
        return errors

    def advance_grip_warmup(self, now=None):
        """Wait without blocking teleoperation, polling, preview or the watchdog."""
        if not self._start_pending or self.stop_requested:
            return
        now = time.monotonic() if now is None else float(now)
        statuses = []
        for name, controller in self.controller.controllers.items():
            get_status = getattr(controller, "get_teleop_status", None)
            statuses.append(get_status(name) if callable(get_status) else {})
        held = bool(statuses) and all(
            s.get("grip_pressed" if self._grip_warmup_started_at is None else "grip_held", False)
            and s.get("stream_fresh", False)
            and not s.get("home_active", False) for s in statuses
        )
        if not held:
            if self._grip_warmup_started_at is not None:
                self.audit.write({"event": "episode_positioning_reset", "time": now,
                                  "reason": "grip_released_or_xr_unavailable"})
                print("EPISODE WAIT GRIP: 定位已暂停；按住 Grip 重新定位，完成后自动录制。", flush=True)
            self._grip_warmup_started_at = None
            self._warmup_display = None
            return
        if self._grip_warmup_started_at is None:
            self._grip_warmup_started_at = now
            self.audit.write({"event": "episode_positioning_start", "time": now,
                              "minimum_wait_s": self.grip_warmup_s})
        remaining = self.grip_warmup_s - (now - self._grip_warmup_started_at)
        if remaining > 0:
            seconds = int(np.ceil(remaining))
            if self._warmup_display != seconds:
                print(f"EPISODE POSITIONING: {seconds} 秒；保持 Grip，定位中不记录。", flush=True)
                self._warmup_display = seconds
            return
        if not all(s.get("follow_ready", False) for s in statuses):
            if self._warmup_display != "follow":
                print("EPISODE WAIT FOLLOW: 正在定位，完成同步或重新定位后自动录制；尚未记录。", flush=True)
                self._warmup_display = "follow"
            return
        camera_errors = self._camera_health_errors()
        if camera_errors:
            message = "; ".join(camera_errors)
            self.audit.write({"event": "episode_start_blocked", "error": message})
            self._cancel_pending_start("camera_unhealthy")
            print(f"EPISODE START BLOCKED: {message}; press Enter to prepare again", flush=True)
            return
        # Request camera frames captured after this boundary, not buffered
        # frames from positioning. Do not read/synchronize SDK feedback while
        # the control thread is following the hand.
        self._last_camera_timestamps = {name: now for name in self.cameras}
        self._reset_recording_history()
        self.episodes.start()
        self.audit.write({"event": "episode_start", "time": now,
                          "start_condition": "grip_follow_ready",
                          "grip_warmup_s": self.grip_warmup_s,
                          "actual_warmup_s": now - self._grip_warmup_started_at})
        self._start_pending = False
        self._grip_warmup_started_at = None
        self._warmup_display = None
        print("EPISODE RECORDING: 定位结束，开始记录，可以执行抓取。", flush=True)

    def _reset_recording_history(self):
        self._xr_pending_arms.clear()
        self._xr_recovery_failure_baseline = {}
        for name, controller in getattr(self.controller, "controllers", {}).items():
            get_status = getattr(controller, "get_teleop_status", None)
            if callable(get_status):
                status = get_status(name) or {}
                self._xr_recovery_failure_baseline[name] = status.get("xr_recovery_failures", 0)
        self._last_recording_sample_time = None
        self._last_recording_observation_state = None
        self._last_recording_action = None
        self._recording_sample_count = 0

    @staticmethod
    def _recorded_joint_values(value):
        array = np.asarray(value, dtype=float)
        if array.ndim != 1:
            raise ValueError("recorded state/action must be one-dimensional")
        if array.size >= 7 and array.size % 7 == 0:
            return array.reshape(-1, 7)[:, :6].reshape(-1)
        return array[:6]

    def print_operator_status(self) -> None:
        state = self.operator_state
        target = self.max_episodes or "unlimited"
        gates = []
        for side in ("left", "right"):
            arm_name = f"{side}_arm"
            controller = self.controller.controllers.get(arm_name)
            get_status = getattr(controller, "get_teleop_status", None)
            if not callable(get_status):
                continue
            status = get_status(arm_name)
            reason = status.get("reason", "unknown")
            reason_text = {
                "no_xr_packet": "等待XR数据，请连接PICO并启动遥操作",
                "xr_stale": "XR输入过期，等待恢复",
                "waiting_xr_recovery": "等待XR稳定后重新接管",
                "release_required": "请完全松开Grip后重新按下",
                "robot_feedback_timeout": "机械臂反馈超时，请松开Grip等待通信恢复",
            }.get(reason, reason)
            gates.append(
                f"{side}:{reason_text} "
                f"grip={float(status.get('grip', 0.0)):.3f}"
            )
        gate_text = " | " + " | ".join(gates) if gates else ""
        enter_action = "prepare" if self.wait_for_grip_follow else "start"
        joystick_help = (
            "PICO右手摇杆前/右/左=准备/保存/作废, "
            if "right_arm" in self.controller.controllers else ""
        )
        print(
            f"[COLLECTION {state}] saved={self.saved_episodes}/{target} | "
            f"Enter={enter_action}, Right Arrow/S=save, Left Arrow=discard, "
            f"{joystick_help}Esc=exit"
            f"{gate_text}",
            flush=True,
        )
        if self.operator_notice:
            print(f"[数据质量提示] {self._quality_reason(self.operator_notice)}", flush=True)

    def handle_key(self, key) -> None:
        if key is None:
            return
        if key == "ENTER":
            if self._start_pending:
                print("EPISODE ALREADY PREPARED: 保持 Grip，定位完成后自动录制。", flush=True)
                return
            if not self.episodes.recording:
                home_arms = []
                active_arms = []
                for name, controller in self.controller.controllers.items():
                    get_status = getattr(controller, "get_teleop_status", None)
                    if not callable(get_status):
                        continue
                    status = get_status(name)
                    if status.get("home_active"):
                        home_arms.append(name)
                    if status.get("active") or (
                        self.wait_for_grip_follow and status.get("grip_pressed")
                    ):
                        active_arms.append(name)
                if home_arms:
                    message = (
                        "Home motion is active for "
                        + ", ".join(home_arms)
                        + "; wait for Home to finish before recording"
                    )
                    self.audit.write(
                        {"event": "episode_start_blocked", "error": message}
                    )
                    print(
                        f"EPISODE START BLOCKED: {message}", flush=True
                    )
                    return
                if active_arms:
                    message = (
                        "Grip is active for "
                        + ", ".join(active_arms)
                        + "; release Grip before recording"
                    )
                    self.audit.write(
                        {"event": "episode_start_blocked", "error": message}
                    )
                    print(f"EPISODE START BLOCKED: {message}", flush=True)
                    return
                arm_sync = {}
                try:
                    for name, controller in self.controller.controllers.items():
                        prepare_arm = getattr(
                            controller, "prepare_collection_episode", None
                        )
                        if callable(prepare_arm):
                            arm_sync[name] = prepare_arm()
                except Exception as exc:
                    message = f"arm encoder sync failed: {exc}"
                    self.audit.write(
                        {"event": "episode_start_blocked", "error": message}
                    )
                    print(f"EPISODE START BLOCKED: {message}", flush=True)
                    return
                prepare_episode = getattr(
                    self.gripper_sampler, "prepare_episode", None
                )
                if callable(prepare_episode):
                    try:
                        prepare_episode()
                    except Exception as exc:
                        message = f"gripper feedback sync failed: {exc}"
                        self.audit.write(
                            {"event": "episode_start_blocked", "error": message}
                        )
                        print(f"EPISODE START BLOCKED: {message}", flush=True)
                        return
                camera_errors = self._camera_health_errors()
                if camera_errors:
                    message = "; ".join(camera_errors)
                    self.audit.write(
                        {"event": "episode_start_blocked", "error": message}
                    )
                    print(
                        f"EPISODE START BLOCKED: {message}", flush=True
                    )
                    return
                self._reset_recording_history()
                self._last_rejection = None
                self._last_save_result = "等待保存"
                self._last_save_message = ""
                if self.wait_for_grip_follow:
                    self._start_pending = True
                    self._grip_warmup_started_at = None
                    self._warmup_display = None
                    self.audit.write({"event": "episode_prepared", "arm_sync": arm_sync,
                                      "start_condition": "grip_follow_ready",
                                      "grip_warmup_s": self.grip_warmup_s})
                    print("EPISODE WAIT GRIP: 已准备；按住 Grip，定位完成后自动录制。", flush=True)
                    return
                self.episodes.start()
                self.audit.write(
                    {"event": "episode_start", "arm_sync": arm_sync}
                )
                print("EPISODE RECORDING", flush=True)
            else:
                print("EPISODE ALREADY RECORDING", flush=True)
            return
        if key == "RIGHT":
            self.audit.write({"event": "episode_save_requested", "state": self.operator_state})
            if self._start_pending:
                print("EPISODE NOT RECORDING: 定位尚未完成，没有数据可保存。", flush=True)
                return
            if self.episodes.recording:
                # Recheck at Save, including between sampling ticks. A pending
                # reconnect must finish before this episode can be accepted.
                control_errors = self._recording_control_errors()
                if control_errors and getattr(self.episodes, "invalid_reason", None) is None:
                    reason = "teleop session stopped: " + "; ".join(control_errors)
                    self.episodes.invalidate(reason)
                    self.audit.write({"event": "episode_control_session_stopped", "error": reason})
                invalid_reason = getattr(self.episodes, "invalid_reason", None)
                if invalid_reason is None and self._xr_pending_arms:
                    print("EPISODE WAIT XR: 输入短暂中断，等待恢复后再按保存；正在继续记录。", flush=True)
                    self.audit.write({"event": "episode_save_wait_xr", "arms": sorted(self._xr_pending_arms)})
                    return
                if invalid_reason is not None and self.discard_invalid_on_save:
                    # Reject the WHOLE episode before any checkpoint/encoding.
                    self._last_save_result = "未保存（无效数据）"
                    self._last_save_message = self._quality_reason(invalid_reason)
                    print("EPISODE REJECTING: 本条数据有问题，不保存，正在清理本条临时数据。", flush=True)
                    self.episodes.discard("invalid_on_save")
                    self._last_rejection = str(invalid_reason)
                    self._reset_recording_history()
                    self.audit.write({"event": "episode_rejected", "error": str(invalid_reason),
                                      "saved_episodes": self.saved_episodes})
                    print("\aEPISODE REJECTED: 数据有问题，本条已整条剔除，未保存，也不保留异常备份。\n"
                          f"原因：{self._quality_reason(invalid_reason)}\n"
                          "请按 Enter 准备重新采集。", flush=True)
                    return
                if invalid_reason is not None and not getattr(self.episodes, "preserve_invalid", False):
                    self._last_save_result = "保存阻止（无效数据）"
                    self._last_save_message = self._quality_reason(invalid_reason)
                    self.audit.write(
                        {
                            "event": "episode_save_blocked",
                            "error": str(invalid_reason),
                        }
                    )
                    print(
                        "EPISODE SAVE BLOCKED: current episode is invalid; "
                        "press Left Arrow to discard it. "
                        f"reason={invalid_reason}",
                        flush=True,
                    )
                    return
                if getattr(self.episodes, "frame_count", None) == 0:
                    self._last_save_result = "未保存（无数据帧）"
                    self._last_save_message = "尚未收到可记录的完整帧"
                    print("EPISODE EMPTY: 尚未收到可记录的完整帧；继续等待或按左键取消。", flush=True)
                    return
                print("EPISODE SAVING: 已收到保存键，正在保存，请勿退出。", flush=True)
                self._last_save_result = "保存中"
                try:
                    result = self.episodes.accept()
                except CheckpointError as exc:
                    self._last_save_result = "保存失败，可重试"
                    self._last_save_message = str(exc)
                    self.audit.write({"event": "episode_checkpoint_failed", "error": str(exc)})
                    print(f"EPISODE SAVE RETRY: {exc}", flush=True)
                    return
                except Exception as exc:
                    self._last_save_result = "保存失败"
                    self._last_save_message = str(exc)
                    path = getattr(getattr(self.episodes, "sink", None), "recovery_path", None)
                    self.audit.write({"event": "episode_save_failed", "error": str(exc),
                                      "recovery_path": str(path) if path else None})
                    print(f"EPISODE SAVE FAILED: {exc}; 原始备份目录={path}。停止本次采集，避免写入半成品数据集。", flush=True)
                    raise
                if result == "archived":
                    self._last_save_result = "异常已归档，未计入"
                    self._last_save_message = self._quality_reason(invalid_reason)
                    self.audit.write({"event": "episode_raw_saved", "quality_issue": invalid_reason})
                    print("EPISODE RAW SAVED: 异常记录已备份，待离线检查；不计入合格条数。按 Enter 可开始下一条。", flush=True)
                    self._reset_recording_history()
                    return
                self.saved_episodes += 1
                self._last_save_result = "保存成功"
                self._last_save_message = f"第 {self.saved_episodes} 条已写入数据集"
                reached_target = (
                    self.max_episodes is not None
                    and self.saved_episodes >= self.max_episodes
                )
                self.audit.write(
                    {
                        "event": "episode_accept",
                        "saved_episodes": self.saved_episodes,
                        "target_episodes": self.max_episodes,
                        "target_reached": reached_target,
                    }
                )
                print(
                    f"EPISODE SAVED {self.saved_episodes}/"
                    f"{self.max_episodes or 'unlimited'}",
                    flush=True,
                )
                if reached_target:
                    self.stop_requested = True
                self._reset_recording_history()
            else:
                print(
                    "EPISODE NOT RECORDING: 当前没有正在录制的数据可保存；"
                    "请先按 Enter 准备，并等待 EPISODE RECORDING。",
                    flush=True,
                )
            return
        if key == "LEFT":
            if self._start_pending:
                self._cancel_pending_start("operator")
                print("EPISODE PREPARATION CANCELLED", flush=True)
                return
            if self.episodes.recording:
                self.episodes.discard("operator")
                self._last_save_result = "已作废"
                self._last_save_message = "操作者取消本条数据"
                self.audit.write({"event": "episode_discard"})
                print("EPISODE DISCARDED", flush=True)
                self._reset_recording_history()
            return
        if key == "ESC":
            self._cancel_pending_start("exit")
            print("COLLECTION EXIT REQUESTED", flush=True)
            self.stop_requested = True
            return

    def _arm_snapshots(self):
        arms = {}
        for name, controller in self.controller.controllers.items():
            snapshot_reader = getattr(controller, "get_collection_snapshot", None)
            if snapshot_reader is None:
                snapshot_reader = controller.rm65.get_collection_snapshot
            arms[name] = snapshot_reader()
        return arms

    def _camera_snapshots(self):
        snapshots = {}
        candidate_timestamps = {}
        for name, camera in self.cameras.items():
            previous = self._last_camera_timestamps.get(
                name, float("-inf")
            )
            snapshot_after = getattr(camera, "snapshot_after", None)
            if callable(snapshot_after) and self.require_fresh_camera_frames:
                sample = snapshot_after(
                    previous,
                    timeout_s=self.camera_wait_timeout_s,
                )
            else:
                # Test doubles and legacy camera adapters remain compatible.
                # The hardware collection path intentionally uses the latest
                # cached frame so a transient UVC read stall does not discard
                # an otherwise usable episode.
                sample = camera.snapshot()
            timestamp = float(sample.monotonic_time)
            if (callable(snapshot_after) and self.require_fresh_camera_frames
                    and timestamp <= previous):
                raise RuntimeError(
                    f"{name} camera returned a reused frame: "
                    f"current={timestamp:.9f}, previous={previous:.9f}"
                )
            snapshots[name] = sample
            candidate_timestamps[name] = timestamp
        # Commit only after every camera supplied a fresh frame.  If one
        # camera times out, the successful camera may be retried next cycle
        # without allowing a partial source bundle into the dataset.
        self._last_camera_timestamps.update(candidate_timestamps)
        return snapshots

    def _sources(self) -> SourceBundle:
        return SourceBundle(
            arms=self._arm_snapshots(),
            grippers=self.gripper_sampler.snapshots(),
            cameras=self._camera_snapshots(),
        )

    def _recording_control_errors(self):
        """Return control-session faults that make a recording unsafe to keep.

        A controller can stop an operator session after an IK or tracking
        rejection while the operator is still holding Grip.  In that state
        the runtime would otherwise keep sampling the held robot position and
        could save a misleading episode after the user presses Save.
        """
        errors = []
        self._xr_pending_arms.clear()
        controllers = getattr(self.controller, "controllers", {})
        for name, controller in controllers.items():
            get_status = getattr(controller, "get_teleop_status", None)
            if not callable(get_status):
                continue
            status = get_status(name) or {}
            # Only controllers explicitly advertising bounded XR recovery get
            # grace. Camera, robot-feedback, cadence and joint checks still run.
            # The counter catches timeouts even if input recovered between samples.
            if status.get("xr_recovery_failures", 0) > self._xr_recovery_failure_baseline.get(name, 0):
                errors.append(f"{name} teleop session interrupted: reason=xr_recovery_failed")
                continue
            if status.get("xr_recovery_pending"):
                elapsed = float(status.get("xr_recovery_elapsed_s", float("inf")))
                if elapsed < 1.0 and status.get("reason") == "waiting_xr_recovery":
                    self._xr_pending_arms.add(name)
                    continue
                errors.append(f"{name} teleop session interrupted: reason=xr_recovery_timeout")
                continue
            if (status.get("reason") == "robot_feedback_timeout" or (status.get("grip_held")
                    and not status.get("active")
                    and status.get("reason") not in (None, "grip_released"))):
                errors.append(
                    f"{name} teleop session interrupted: "
                    f"reason={status.get('reason')}"
                )
        return errors

    def sample_once(self, sample_time=None):
        try:
            sources = self._sources()
        except Exception as exc:
            reason = f"fresh camera frame unavailable: {exc}"
            if (
                self.episodes.recording
                and getattr(self.episodes, "invalid_reason", None) is None
            ):
                self.episodes.invalidate(reason)
                print(f"EPISODE INVALID: {reason}", flush=True)
            self.audit.write(
                {"event": "dataset_camera_freshness_violation", "error": reason}
            )
            raise
        if self.episodes.recording:
            control_errors = self._recording_control_errors()
            if control_errors and getattr(self.episodes, "invalid_reason", None) is None:
                reason = "teleop session stopped: " + "; ".join(control_errors)
                self.episodes.invalidate(reason)
                self.audit.write({"event": "episode_control_session_stopped", "error": reason})
                print(f"EPISODE INVALID: {reason}", flush=True)
            home_arms = [
                name
                for name, snapshot in sources.arms.items()
                if snapshot.get("home_active")
            ]
            if home_arms and getattr(
                self.episodes, "invalid_reason", None
            ) is None:
                reason = (
                    "Home motion detected during recording: "
                    + ", ".join(home_arms)
                )
                self.episodes.invalidate(reason)
                self.audit.write(
                    {"event": "episode_home_violation", "error": reason}
                )
                print(f"EPISODE INVALID: {reason}", flush=True)
        sample_time = (
            time.monotonic() if sample_time is None else float(sample_time)
        )
        try:
            frame = self.frame_builder.build(sample_time, sources)
        except Exception as exc:
            if self.episodes.recording:
                self.episodes.invalidate(str(exc))
            self.audit.write(
                {"event": "sample_invalid", "time": sample_time, "error": str(exc)}
            )
            raise
        sample_interval_s = None
        if (
            self.episodes.recording
            and self._last_recording_sample_time is not None
        ):
            sample_interval_s = (
                sample_time - self._last_recording_sample_time
            )
        self.audit.write(
            {
                "event": "dataset_sample",
                "time": sample_time,
                "observation_state": frame.observation_state.tolist(),
                "action": frame.action.tolist(),
                "source_ages_ms": dict(frame.source_ages_ms),
                "source_timestamps": dict(frame.source_timestamps),
                "arm_feedback_read_times": {
                    name: snapshot.get("feedback_read_time")
                    for name, snapshot in sources.arms.items()
                },
                "sample_interval_ms": (
                    None
                    if sample_interval_s is None
                    else sample_interval_s * 1000.0
                ),
            }
        )
        if self.episodes.recording:
            continuity_violation = None
            for field_name, current, previous in (
                (
                    "observation.state",
                    frame.observation_state,
                    self._last_recording_observation_state,
                ),
                ("action", frame.action, self._last_recording_action),
            ):
                if previous is None:
                    continue
                joint_delta = np.abs(
                    self._recorded_joint_values(current)
                    - self._recorded_joint_values(previous)
                )
                joint_index = int(np.argmax(joint_delta))
                if joint_delta[joint_index] > self.max_recorded_joint_step_rad:
                    continuity_violation = {
                        "field": field_name,
                        "joint": joint_index + 1,
                        "delta_deg": float(np.rad2deg(joint_delta[joint_index])),
                    }
                    break
            if continuity_violation is not None:
                reason = (
                    "recorded joint continuity violation: "
                    f"field={continuity_violation['field']}, "
                    f"joint=J{continuity_violation['joint']}, "
                    f"delta_deg={continuity_violation['delta_deg']:.3f}, "
                    f"limit_deg={np.rad2deg(self.max_recorded_joint_step_rad):.3f}"
                )
                self.episodes.invalidate(reason)
                self.audit.write(
                    {"event": "dataset_joint_jump", "error": reason}
                )
                print(f"EPISODE INVALID: {reason}", flush=True)
            self._last_recording_observation_state = np.asarray(
                frame.observation_state, dtype=float
            ).copy()
            self._last_recording_action = np.asarray(
                frame.action, dtype=float
            ).copy()
            self._last_recording_sample_time = sample_time
            self._recording_sample_count += 1
            if (
                sample_interval_s is not None
                and self._recording_sample_count > self._cadence_warmup_samples
                and self.reject_on_cadence_violation
                and sample_interval_s > self.max_dataset_gap_s
            ):
                reason = (
                    "dataset cadence violation: "
                    f"gap_ms={sample_interval_s * 1000.0:.3f}, "
                    f"limit_ms={self.max_dataset_gap_s * 1000.0:.3f}"
                )
                self.episodes.invalidate(reason)
                self.audit.write(
                    {"event": "dataset_cadence_violation", "error": reason}
                )
                print(f"EPISODE INVALID: {reason}", flush=True)
            if (getattr(self.episodes, "invalid_reason", None) is None
                    or getattr(self.episodes, "preserve_invalid", False)):
                try:
                    self.episodes.append(frame)
                except Exception as exc:
                    self.episodes.invalidate(f"frame_write_failed: {exc}")
                    # A partial sink write is not a mere camera sampling error.
                    # Stop and checkpoint completed frames; do not keep writing
                    # into a potentially inconsistent LeRobot buffer.
                    if (getattr(self.episodes, "preserve_invalid", False)
                            or self.discard_invalid_on_save):
                        self.stop_requested = True
                        print(f"COLLECTION WRITE FAILED: {exc}; 停止录制，按当前数据质量策略清理本条。", flush=True)
                    raise
        return frame

    def audit_once(self, sample_time=None) -> None:
        sample_time = time.monotonic() if sample_time is None else float(sample_time)
        arms = self._arm_snapshots()
        arm_payload = {}
        for name, snapshot in arms.items():
            controller = self.controller.controllers[name]
            get_status = getattr(controller, "get_teleop_status", None)
            teleop = get_status(name) if callable(get_status) else None
            arm_payload[name] = {
                **snapshot,
                "feedback_positions": (
                    None if snapshot["feedback_positions"] is None
                    else snapshot["feedback_positions"].tolist()
                ),
                "sent_positions": (
                    None if snapshot["sent_positions"] is None
                    else snapshot["sent_positions"].tolist()
                ),
                "action_positions": (
                    None
                    if snapshot.get("action_positions") is None
                    else snapshot["action_positions"].tolist()
                ),
                "action_source": snapshot.get("action_source"),
                "teleop": teleop,
            }
        self.audit.write(
            {
                "event": "control_audit",
                "time": sample_time,
                "arms": arm_payload,
                "grippers": self.gripper_sampler.snapshots(),
            }
        )

    def run(self, keyboard) -> None:
        dataset_period = 1.0 / self.dataset_fps
        audit_period = 1.0 / self.audit_hz
        dataset_deadline = time.monotonic()
        audit_deadline = dataset_deadline
        status_deadline = dataset_deadline
        self.print_operator_status()
        while not self.stop_requested:
            raise_if_failed = getattr(self.controller, "raise_if_failed", None)
            if callable(raise_if_failed):
                try:
                    raise_if_failed()
                except Exception as exc:
                    self.audit.write({
                        "event": "collection_controller_failure",
                        "time": time.monotonic(),
                        "error": f"{type(exc).__name__}: {exc}",
                        "saved_episodes": self.saved_episodes,
                        "recording": self.episodes.recording,
                    })
                    print(
                        f"COLLECTION STOPPED: {exc}. "
                        f"已保存 {self.saved_episodes} 条保留在数据目录；"
                        "请检查机械臂连接和供电后重新启动。",
                        flush=True,
                    )
                    if self.episodes.recording:
                        if getattr(self.episodes, "preserve_invalid", False):
                            self.episodes.invalidate("controller_failure")
                        else:
                            self.episodes.discard("controller_failure")
                    self._cancel_pending_start("controller_failure")
                    raise
            self.heartbeat()
            now = time.monotonic()
            timeout = max(
                0.0,
                min(dataset_deadline, audit_deadline) - now,
            )
            self.handle_key(self._poll_joystick_action())
            if self.stop_requested:
                break
            key = keyboard.poll(timeout)
            self.handle_key(key)
            if self.stop_requested:
                break
            self.advance_grip_warmup()
            now = time.monotonic()
            if now >= audit_deadline:
                self.audit_once(now)
                audit_deadline += audit_period
                if audit_deadline < now - audit_period:
                    audit_deadline = now + audit_period
            if now >= dataset_deadline:
                try:
                    self.sample_once()
                except Exception:
                    pass
                dataset_deadline += dataset_period
                if dataset_deadline < now - dataset_period:
                    dataset_deadline = now + dataset_period
            if now >= status_deadline:
                self.print_operator_status()
                status_deadline = now + 5.0

    def close(self) -> None:
        self._cancel_pending_start("exit")
        self.episodes.close()
