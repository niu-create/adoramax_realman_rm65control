#!/usr/bin/env python3
"""Collect commissioned left-RM65 demonstrations with top/left UVC cameras."""

import argparse
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LEFT_DIMENSIONS = [
    "left_q1",
    "left_q2",
    "left_q3",
    "left_q4",
    "left_q5",
    "left_q6",
    "left_gripper",
]
TOP_CAMERA_SERIAL = "CP1Z842000EP"
LEFT_WRIST_CAMERA_SERIAL = "CP1Z842000C3"
DATASET_FPS = 30


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--top-camera", default="auto")
    parser.add_argument("--left-wrist-camera", default="auto")
    parser.add_argument("--right-wrist-camera", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--no-camera-preview", action="store_true",
                        help="disable the head/left-wrist camera preview window")
    parser.add_argument(
        "--vr-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stream the cached head camera to XRoboToolkit Remote Vision (default: enabled)",
    )
    parser.add_argument("--vr-control-port", type=int, default=13579)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument(
        "--video-crf",
        type=int,
        default=18,
        help="AV1 constant-rate factor; lower values preserve more frame detail",
    )
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--arm-response", choices=("original", "faster", "responsive"),
                        default="responsive", help="bounded elapsed-time translation response")
    parser.add_argument("--tracking-fault-recovery", choices=("auto", "release"), default="auto",
                        help="auto: recover XR pauses within 1 second; release: require a new Grip press")
    parser.add_argument(
        "--forward-reach-gain", type=float, default=2.0,
        help="same as right: forward displacement 1..2; backward remains 1:1",
    )
    parser.add_argument("--right-reach-gain", type=float, default=1.5)
    parser.add_argument("--vertical-reach-gain", type=float, default=1.5)
    parser.add_argument("--wrist-control", choices=("auto", "pose", "auto-ik"), default="auto")
    parser.add_argument("--ray-j4-control", action="store_true")
    parser.add_argument("--ray-j4-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--tracking-frame", choices=("head_relative", "world"), default="head_relative")
    parser.add_argument("--grip-sync", choices=("reanchor", "persistent"), default="reanchor")
    parser.add_argument("--grip-sync-speed-deg-s", type=float, default=90.0)
    parser.add_argument("--position-fault-recovery", choices=("release", "reanchor"), default="release")
    parser.add_argument("--position-gain", type=float, default=None,
                        help="obsolete: omit this option to use the right-arm directional gains")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _device(value):
    text = str(value)
    if text.isdigit():
        return int(text)
    if text.startswith("/dev/video") and text[10:].isdigit():
        return int(text[10:])
    return text


def _udev_properties(device):
    result = subprocess.run(
        ["udevadm", "info", "-q", "property", "-n", str(device)],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def _resolve_camera(selector, expected_serial, role):
    candidates = sorted(
        Path("/dev").glob("video*"), key=lambda path: int(path.name[5:])
    )
    if str(selector).lower() != "auto":
        candidates = [
            Path(
                str(selector)
                if str(selector).startswith("/dev/")
                else f"/dev/video{selector}"
            )
        ]
    matches = []
    for device in candidates:
        try:
            properties = _udev_properties(device)
        except (OSError, subprocess.CalledProcessError):
            continue
        serial = properties.get("ID_SERIAL_SHORT", "") or properties.get(
            "ID_SERIAL", ""
        )
        capabilities = properties.get("ID_V4L_CAPABILITIES", "")
        path = properties.get("ID_PATH", "")
        identity_matches = serial == expected_serial or serial.endswith(expected_serial)
        is_capture_node = ":capture:" in capabilities and path.endswith(":1.4")
        if identity_matches and is_capture_node:
            matches.append(str(device))
    if len(matches) != 1:
        raise RuntimeError(
            f"{role} camera identity check failed: expected serial={expected_serial}, "
            f"selector={selector!r}, matches={matches}"
        )
    return matches[0]


def _validate(args):
    if not 1 <= getattr(args, "vr_control_port", 13579) <= 65535:
        raise ValueError("vr-control-port must be in [1, 65535]")
    cameras = (args.top_camera, args.left_wrist_camera)
    if len(set(cameras)) != 2 and "auto" not in {
        str(value).lower() for value in cameras
    }:
        raise ValueError("top and left-wrist selectors must identify different devices")
    if not args.task.strip() or not args.repo_id.strip():
        raise ValueError("repo-id and task may not be empty")
    if min(args.camera_width, args.camera_height, args.camera_fps) <= 0:
        raise ValueError("camera dimensions and fps must be positive")
    if args.image_writer_threads <= 0:
        raise ValueError("image-writer-threads must be positive")
    if not 0 <= args.video_crf <= 63:
        raise ValueError("video-crf must be between 0 and 63")
    if args.num_episodes is not None and args.num_episodes <= 0:
        raise ValueError("num-episodes must be positive")
    if not 1.0 <= args.forward_reach_gain <= 2.0:
        raise ValueError("forward-reach-gain must be in [1.0, 2.0]")
    if not 1.0 <= args.right_reach_gain <= 1.5 or not 1.0 <= args.vertical_reach_gain <= 1.5:
        raise ValueError("right/vertical-reach-gain must be in [1.0, 1.5]")
    if args.position_gain is not None:
        raise ValueError("已切换右臂控制逻辑，请删除旧参数 --position-gain；默认前伸2倍、向右及上下1.5倍")
    if not 1.0 <= args.grip_sync_speed_deg_s <= 180.0:
        raise ValueError("grip-sync-speed-deg-s must be in [1, 180]")
    if args.grip_sync == "persistent" and args.tracking_frame != "head_relative":
        raise ValueError("persistent Grip sync requires head_relative tracking")
    if args.ray_j4_control and args.wrist_control == "pose":
        raise ValueError("ray-j4-control requires auto or auto-ik wrist control")
    root = Path(args.dataset_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"dataset root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    return root


def _print_plan(args, root):
    print("Commissioned left RM65 LeRobot collection preflight")
    print(f"mode={'DRY_RUN' if args.dry_run else 'REAL_HARDWARE'}")
    print(f"dataset_root={root}")
    print(f"repo_id={args.repo_id}")
    print("arm_ip=192.168.1.18")
    print("teleop_source=shared_current_right_controller; left_installation_adapter")
    print("left_mapping=right_observed_directions_transferred_through_416table; normalized_FK_IK")
    print(f"arm_response={args.arm_response}; health_checks=before_and_after_Grip")
    print(f"tracking_fault_recovery={args.tracking_fault_recovery}")
    print("gripper_control=left_Grip+Trigger_toggle; release_Trigger_to_rearm")
    print(f"forward_reach_gain={args.forward_reach_gain}; right_reach_gain={args.right_reach_gain}; vertical_reach_gain={args.vertical_reach_gain}; backward_and_left=1.0")
    print(f"wrist_control={args.wrist_control}; ray_j4_control={args.ray_j4_control}; tracking_frame={args.tracking_frame}; grip_sync={args.grip_sync}")
    print("fault_handling=shared_right_controller")
    print("observation.state: float32[7]")
    print("action: float32[7]")
    print("order=left_q1..q6,left_gripper")
    print("units=joints_rad,gripper_0_to_100")
    print("action_source=successful_physical_command")
    print("camera_keys=top,left_wrist")
    print(f"top_camera={args.top_camera} expected_serial={TOP_CAMERA_SERIAL}")
    print(
        f"left_wrist_camera={args.left_wrist_camera} "
        f"expected_serial={LEFT_WRIST_CAMERA_SERIAL}"
    )
    print("right_wrist_camera=DISABLED")
    print(f"camera_fps={args.camera_fps}")
    print(
        f"vr_head_camera={'ENABLED' if getattr(args, 'vr_preview', True) else 'DISABLED'} "
        f"control_port={getattr(args, 'vr_control_port', 13579)} source=top"
    )
    print(f"image_writer_threads={args.image_writer_threads}")
    print(f"video_codec=libsvtav1 video_crf={args.video_crf}")
    print(f"dataset_fps={DATASET_FPS}")
    print(f"num_episodes={args.num_episodes or 'unlimited'}")
    print("episode_start=Enter_prepare_then_Grip_follow_ready; countdown=DISABLED")
    print("quality_policy=discard_invalid_on_save; backup_valid_before_encoding")
    print(f"xr_recovery={args.tracking_fault_recovery}; window_s=1.0; motion_stop_timeout_s=0.2")
    if args.dry_run:
        print("motion_commands=0 (dry-run)")
    else:
        print("motion_commands=LEFT_ARM_ENABLED_AFTER_PREFLIGHT")


def _wait_cameras(cameras, timeout_s=8):
    deadline = time.monotonic() + timeout_s
    errors = []
    while time.monotonic() < deadline:
        errors = [
            error
            for camera in cameras.values()
            if (error := camera.health_error(max_age_s=0.5))
        ]
        if not errors:
            return
        time.sleep(0.05)
    raise RuntimeError("camera preflight failed: " + "; ".join(errors))


def _instructions(count, wrist_control="auto", ray_j4_control=False):
    print("\n========== 单左臂 Episode 采集 ==========", flush=True)
    print("Enter：准备 | 按住左 Grip 进入跟随后自动录制 | → / S：保存 | ←：作废/取消准备 | Esc：退出", flush=True)
    print("先松开 Grip 再按 Enter；看到 EPISODE RECORDING 后执行任务。", flush=True)
    print("INVALID 表示本条异常；按保存键将剔除本条，不计入合格条数。", flush=True)
    print("遥操作：按住左 Grip 平移；保持 Grip，按 Trigger 切换夹爪开/合。", flush=True)
    print("每次松开 Trigger 后再按才切换；单按 Trigger 不动作。", flush=True)
    if wrist_control == "auto":
        joints = "J4/J5/J6" if ray_j4_control else "J5/J6"
        print(f"沿用右臂自动模式：转动手柄切换到 {joints}；明显平移后切回。转腕时末端位置可能改变。", flush=True)
    elif wrist_control == "auto-ik":
        print("沿用右臂 auto-ik：J5 单独转动，J4/J6 姿态意图通过整臂 IK；明显平移后切回。", flush=True)
    else:
        print("沿用右臂 pose：手柄位置和朝向同时控制末端。", flush=True)
    print("XR 长时间断流后先松开再按 Grip。", flush=True)
    print(
        f"目标：{count or '不限数量'} 条；仅记录顶部和左腕两路相机，右腕相机已停用。",
        flush=True,
    )
    print("=========================================\n", flush=True)


def _configure_lerobot_video_crf(video_crf, *, dataset_module=None, base_encoder=None):
    """Apply the requested AV1 quality to LeRobot's episode encoder."""
    if dataset_module is None:
        from lerobot.datasets import lerobot_dataset as dataset_module
    if base_encoder is None:
        from lerobot.datasets.video_utils import encode_video_frames as base_encoder
    video_crf = int(video_crf)

    def encode_rm65_video(*args, **kwargs):
        kwargs["crf"] = video_crf
        return base_encoder(*args, **kwargs)

    encode_rm65_video._rm65_video_crf = video_crf
    dataset_module.encode_video_frames = encode_rm65_video
    return encode_rm65_video


def _run_real(args, root):
    from xrobotoolkit_teleop.collection.audit import JsonlAuditWriter
    from xrobotoolkit_teleop.collection.cameras import OpenCVCameraReader
    from xrobotoolkit_teleop.collection.recovery import RecoverableLeRobotDatasetSink
    from xrobotoolkit_teleop.collection.episode import EpisodeManager
    from xrobotoolkit_teleop.collection.keyboard import TerminalKeyboard
    from xrobotoolkit_teleop.collection.left_frame_builder import LeftArmFrameBuilder
    from xrobotoolkit_teleop.collection.preview import CameraPreviewKeyboard
    from xrobotoolkit_teleop.collection.runtime import DualRM65CollectionRuntime
    from xrobotoolkit_teleop.collection.supervisor import SupervisorNotifier
    from xrobotoolkit_teleop.collection.vr_video import VRHeadCameraStreamer
    from xrobotoolkit_teleop.hardware.commissioned_left_rm65_collection_controller import (
        CommissionedLeftRM65CollectionController,
    )
    from xrobotoolkit_teleop.hardware.right_rm65_collection_controller import (
        CachedGripperSampler,
    )

    class TwoCameraLeRobotDatasetSink(RecoverableLeRobotDatasetSink):
        CAMERA_KEYS = (
            "observation.images.top",
            "observation.images.left_wrist",
        )
        KEEP_DISCARDED_BACKUP = False

        def checkpoint(self, reasons):
            with notifier.saving():
                return super().checkpoint(reasons)

        def accept_episode(self, task):
            print("EPISODE SAVING: 正在编码顶部和左腕视频，请等待 EPISODE SAVED。", flush=True)
            with notifier.saving():
                return super().accept_episode(task)

    notifier = SupervisorNotifier.from_environment()
    cameras = {}
    controller = None
    controller_thread = None
    sampler = None
    runtime = None
    episodes = None
    audit = None
    vr_stream = None
    preview = None
    sink = None
    errors = []
    try:
        top_camera = _resolve_camera(args.top_camera, TOP_CAMERA_SERIAL, "top")
        left_wrist_camera = _resolve_camera(
            args.left_wrist_camera,
            LEFT_WRIST_CAMERA_SERIAL,
            "left_wrist",
        )
        if top_camera == left_wrist_camera:
            raise RuntimeError("top and left-wrist cameras resolved to the same node")
        if args.right_wrist_camera:
            print(
                f"[CAMERA NOTICE] --right-wrist-camera={args.right_wrist_camera} "
                "ignored; right wrist capture is disabled."
            )
        print(f"[CAMERA VERIFIED] top={top_camera} serial={TOP_CAMERA_SERIAL}")
        print(
            f"[CAMERA VERIFIED] left_wrist={left_wrist_camera} "
            f"serial={LEFT_WRIST_CAMERA_SERIAL}"
        )
        with TerminalKeyboard() as keyboard:
            # Register each reader immediately so partial construction is cleaned up.
            for role, device in (("top", top_camera), ("left_wrist", left_wrist_camera)):
                cameras[role] = OpenCVCameraReader(
                    role,
                    _device(device),
                    width=args.camera_width,
                    height=args.camera_height,
                    fps=args.camera_fps,
                )
            for camera in cameras.values():
                camera.start()
            _wait_cameras(cameras)
            if not args.no_camera_preview:
                if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
                    # Initialize Qt before LeRobot/PyAV to avoid window startup deadlock.
                    preview = CameraPreviewKeyboard(
                        keyboard, cameras, lambda: bool(episodes and episodes.recording),
                        status=lambda: runtime.operator_state if runtime else "READY",
                        notice=lambda: runtime.operator_notice if runtime else "",
                        camera_roles=(("top", "HEAD"), ("left_wrist", "LEFT WRIST")),
                    )
                else:
                    print("[CAMERA PREVIEW] 无桌面显示，使用终端按键。", flush=True)
            if args.vr_preview:
                vr_stream = VRHeadCameraStreamer(
                    cameras["top"],
                    control_port=args.vr_control_port,
                )
                vr_stream.start()
            _configure_lerobot_video_crf(args.video_crf)
            sink = TwoCameraLeRobotDatasetSink.create(
                root=root,
                repo_id=args.repo_id,
                fps=DATASET_FPS,
                image_height=args.camera_height,
                image_width=args.camera_width,
                dimension_names=LEFT_DIMENSIONS,
                image_writer_threads=args.image_writer_threads,
            )
            audit = JsonlAuditWriter(
                root / "audit" / f"session-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
            )
            episodes = EpisodeManager(sink, task=args.task, backup_valid=True)
            controller = CommissionedLeftRM65CollectionController(
                log_dir=root / "teleop_logs",
                forward_reach_gain=args.forward_reach_gain,
                arm_response=args.arm_response,
                resume_tracking_faults=args.tracking_fault_recovery == "auto",
                right_reach_gain=args.right_reach_gain,
                vertical_reach_gain=args.vertical_reach_gain,
                wrist_control=args.wrist_control,
                ray_j4_control=args.ray_j4_control,
                ray_j4_sign=args.ray_j4_sign,
                tracking_frame=args.tracking_frame,
                grip_sync=args.grip_sync,
                grip_sync_speed_deg_s=args.grip_sync_speed_deg_s,
                position_fault_recovery=args.position_fault_recovery,
            )
            controller.start_xr_collection()

            def control():
                try:
                    controller.run()
                except BaseException as exc:
                    errors.append(exc)

            controller_thread = threading.Thread(
                target=control, name="commissioned-left-rm65-control"
            )
            controller_thread.start()
            sampler = CachedGripperSampler(controller.grippers)
            sampler.start()
            runtime = DualRM65CollectionRuntime(
                controller=controller,
                cameras=cameras,
                gripper_sampler=sampler,
                episodes=episodes,
                audit=audit,
                frame_builder=LeftArmFrameBuilder(),
                dataset_fps=DATASET_FPS,
                max_episodes=args.num_episodes,
                heartbeat=notifier.heartbeat,
                wait_for_grip_follow=True,
                discard_invalid_on_save=True,
            )
            notifier.ready()
            _instructions(args.num_episodes, args.wrist_control, args.ray_j4_control)
            runtime.run(preview if preview is not None else keyboard)
            print(f"Collection finished: saved_episodes={runtime.saved_episodes}")
    finally:
        # Stop control before disk cleanup; every resource gets a cleanup attempt.
        # Preserve the original failure when a secondary close operation also fails.
        primary_error = sys.exc_info()[1]
        cleanup_errors = []

        def cleanup(label, action):
            try:
                action()
            except Exception as exc:
                cleanup_errors.append(exc)
                print(f"[CLEANUP ERROR] {label}: {exc}", file=sys.stderr, flush=True)

        if controller is not None:
            cleanup("request_stop", controller.request_stop)
        if controller_thread is not None and controller_thread.ident is not None:
            cleanup("control_thread", lambda: controller_thread.join(timeout=8))
        control_alive = controller_thread is not None and controller_thread.is_alive()
        if control_alive:
            cleanup_errors.append(RuntimeError("left control thread did not stop within 8 seconds"))
        elif controller is not None:
            cleanup("controller", controller.close)
        if runtime is not None:
            cleanup("runtime", runtime.close)
        elif episodes is not None:
            cleanup("episodes", episodes.close)
        for label, resource in (("sampler", sampler), ("preview", preview),
                                ("vr_stream", vr_stream), *cameras.items(), ("audit", audit)):
            if resource is not None:
                cleanup(label, resource.close)
        if sink is not None:
            cleanup("image_writer", sink.dataset.stop_image_writer)
        if primary_error is None and not errors and cleanup_errors:
            raise RuntimeError("left collection cleanup failed: " + "; ".join(map(str, cleanup_errors))) from cleanup_errors[0]
    if errors:
        raise errors[0]
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = _validate(args)
    _print_plan(args, root)
    return 0 if args.dry_run else _run_real(args, root)


if __name__ == "__main__":
    raise SystemExit(main())
