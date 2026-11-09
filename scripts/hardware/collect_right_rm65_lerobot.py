#!/usr/bin/env python3
"""Run right-RM65 teleoperation, optionally collecting verified camera data."""
import argparse
import os
import subprocess
import threading
import time
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RIGHT_DIMENSIONS = ["right_q1", "right_q2", "right_q3", "right_q4", "right_q5", "right_q6", "right_gripper"]
TOP_CAMERA_SERIAL = "CP1Z842000EP"
RIGHT_WRIST_CAMERA_SERIAL = "CP1Z842000W8"

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teleop-only", action="store_true",
                        help="test the arm without cameras or dataset recording")
    parser.add_argument("--log-dir", type=Path,
                        help="diagnostic log directory for --teleop-only; defaults under outputs/teleop_test")
    parser.add_argument("--dataset-root")
    parser.add_argument("--repo-id")
    parser.add_argument("--task")
    parser.add_argument("--top-camera", default="auto")
    parser.add_argument("--left-wrist-camera", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--right-wrist-camera", default="auto")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--no-camera-preview", action="store_true",
                        help="disable the head/right-wrist camera preview window")
    parser.add_argument(
        "--vr-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stream the cached head camera to XRoboToolkit Remote Vision (default: enabled)",
    )
    parser.add_argument(
        "--vr-control-port",
        type=int,
        default=13579,
        help="XRoboToolkit Remote Vision control port",
    )
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument(
        "--video-crf", type=int, default=18,
        help="AV1 constant-rate factor; lower values preserve more frame detail",
    )
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument(
        "--arm-response", choices=("original", "faster", "responsive"), default="responsive",
        help="responsive keeps the original wrist input gain and improves bounded command pacing; faster/original restore previous profiles (all within 180 deg/s)",
    )
    parser.add_argument(
        "--forward-reach-gain", type=float, default=2.0,
        help="forward displacement from the reference, 1..2 (default 2); backward stays 1:1",
    )
    parser.add_argument("--right-reach-gain", type=float, default=1.5,
                        help="rightward displacement from the reference, 1..1.5; left stays 1:1")
    parser.add_argument("--vertical-reach-gain", type=float, default=1.5,
                        help="up/down displacement gain, 1..1.5")
    parser.add_argument("--grip-sync-speed-deg-s", type=float, default=90.0,
                        help="persistent position reacquisition speed cap, 1..180 deg/s")
    parser.add_argument("--position-fault-recovery", choices=("reanchor", "release"), default="release",
                        help="reanchor stops and resets hand/TCP reference after unreachable or stalled position targets; release requires a new Grip press")
    parser.add_argument(
        "--wrist-control", choices=("buttons", "auto", "pose", "auto-ik"), default="auto",
        help="auto-ik controls J5 directly and uses whole-arm IK for J4/J6 (twist reversed); pose follows all orientation axes continuously; auto controls wrist joints directly",
    )
    parser.add_argument(
        "--ray-j4-control", action="store_true",
        help="opt in to local ray left/right rotation controlling J4 in auto wrist mode",
    )
    parser.add_argument(
        "--ray-j4-sign", type=int, choices=(-1, 1), default=1,
        help="J4 direction for ray left/right rotation; reverse with -1 after a small physical test",
    )
    parser.add_argument(
        "--tracking-frame", choices=("head_relative", "world"), default="head_relative",
        help="head_relative reads controller pose relative to the headset; world uses the XR-world reference",
    )
    parser.add_argument(
        "--grip-sync", choices=("persistent", "reanchor"), default="reanchor",
        help="persistent remembers the first Grip hand/head/TCP mapping and tracks the moving hand during later synchronization; reanchor restores each-press zeroing",
    )
    parser.add_argument(
        "--absolute-control", choices=("off", "pose", "wrist"), default="off",
        help="absolute pose or fixed-J1-J3 wrist control; select camera calibration or first-Grip test registration",
    )
    parser.add_argument(
        "--head-camera-calibration", type=Path,
        help="validated 416table-to-head-camera calibration JSON",
    )
    parser.add_argument(
        "--uncalibrated-absolute-test", action="store_true",
        help="explicitly test first-Grip absolute registration without camera-frame targets",
    )
    parser.add_argument(
        "--grip-sync-orientation-relax-deg", type=float, default=0.0,
        help="opt-in right-arm Grip re-press recovery: preserve TCP XYZ and allow up to 10 deg tool-local Y orientation change only when exact IK fails",
    )
    parser.add_argument(
        "--tracking-fault-recovery", choices=("auto", "release"), default="release",
        help="auto holds and re-anchors after a short XR gap/jump while Grip stays held; release restores manual re-press recovery",
    )
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
        check=True, capture_output=True, text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)

def _resolve_camera(selector, expected_serial, role):
    candidates = sorted(Path("/dev").glob("video*"), key=lambda p: int(p.name[5:]))
    if str(selector).lower() != "auto":
        candidates = [Path(str(selector) if str(selector).startswith("/dev/") else f"/dev/video{selector}")]
    matches = []
    for device in candidates:
        try:
            properties = _udev_properties(device)
        except (OSError, subprocess.CalledProcessError):
            continue
        serial = properties.get("ID_SERIAL_SHORT", "") or properties.get("ID_SERIAL", "")
        capabilities = properties.get("ID_V4L_CAPABILITIES", "")
        path = properties.get("ID_PATH", "")
        identity_matches = serial == expected_serial or serial.endswith(expected_serial)
        is_capture_node = ":capture:" in capabilities and path.endswith(":1.4")
        if identity_matches and is_capture_node:
            matches.append(str(device))
    if len(matches) != 1:
        detected = []
        for device in candidates:
            try:
                properties = _udev_properties(device)
            except (OSError, subprocess.CalledProcessError):
                continue
            detected.append(
                f"{device}(serial={properties.get('ID_SERIAL_SHORT', '') or properties.get('ID_SERIAL', '')}, "
                f"path={properties.get('ID_PATH', '')}, caps={properties.get('ID_V4L_CAPABILITIES', '')})"
            )
        raise RuntimeError(
            f"{role} camera identity check failed: expected serial={expected_serial}, "
            f"selector={selector!r}, matches={matches}, detected={detected or 'no /dev/video nodes'}"
        )
    return matches[0]


def _resolve_camera_with_retry(selector, expected_serial, role, *, timeout_s=10.0):
    """Allow USB cameras a short time to re-enumerate after reconnect/startup."""
    deadline = time.monotonic() + float(timeout_s)
    last_error = None
    while True:
        try:
            return _resolve_camera(selector, expected_serial, role)
        except RuntimeError as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                raise
            print(
                f"[CAMERA WAIT] {role} 未检测到序列号 {expected_serial}；"
                "请确认 USB 相机已连接，等待重新枚举...",
                flush=True,
            )
            time.sleep(0.5)

def _validate(args):
    if not 1 <= getattr(args, "vr_control_port", 13579) <= 65535:
        raise ValueError("vr-control-port must be in [1, 65535]")
    if not args.teleop_only:
        if not all((args.dataset_root, args.repo_id, args.task)):
            raise ValueError("--dataset-root, --repo-id and --task are required unless --teleop-only is selected")
        cameras = (args.top_camera, args.right_wrist_camera)
        if len(set(cameras)) != 2 and "auto" not in {str(value).lower() for value in cameras}:
            raise ValueError("top and right-wrist selectors must identify different devices")
        if not args.task.strip() or not args.repo_id.strip(): raise ValueError("repo-id and task may not be empty")
        if min(args.camera_width, args.camera_height, args.camera_fps) <= 0: raise ValueError("camera dimensions and fps must be positive")
        if args.image_writer_threads <= 0: raise ValueError("image-writer-threads must be positive")
        if not 0 <= args.video_crf <= 63: raise ValueError("video-crf must be between 0 and 63")
        if args.num_episodes is not None and args.num_episodes <= 0: raise ValueError("num-episodes must be positive")
    if not 1.0 <= args.forward_reach_gain <= 2.0: raise ValueError("forward-reach-gain must be in [1.0, 2.0]")
    if not 1.0 <= args.right_reach_gain <= 1.5: raise ValueError("right-reach-gain must be in [1.0, 1.5]")
    if not 1.0 <= args.vertical_reach_gain <= 1.5: raise ValueError("vertical-reach-gain must be in [1.0, 1.5]")
    if not 1.0 <= args.grip_sync_speed_deg_s <= 180.0: raise ValueError("grip-sync-speed-deg-s must be in [1, 180]")
    if args.ray_j4_control and args.wrist_control not in ("auto", "auto-ik"): raise ValueError("ray-j4-control requires --wrist-control auto or auto-ik")
    if args.wrist_control in ("pose", "auto-ik") and (args.absolute_control != "off" or args.grip_sync_orientation_relax_deg):
        raise ValueError("whole-arm wrist control requires --absolute-control off and no orientation relaxation")
    if args.grip_sync == "persistent" and args.tracking_frame != "head_relative": raise ValueError("persistent Grip sync requires --tracking-frame head_relative")
    if not 0 <= args.grip_sync_orientation_relax_deg <= 10: raise ValueError("grip-sync-orientation-relax-deg must be in [0, 10]")
    if args.grip_sync_orientation_relax_deg and args.grip_sync != "persistent": raise ValueError("Grip sync orientation recovery requires --grip-sync persistent")
    if args.absolute_control != "off":
        if args.tracking_frame != "head_relative": raise ValueError("absolute control requires --tracking-frame head_relative")
        if args.uncalibrated_absolute_test == bool(args.head_camera_calibration):
            raise ValueError("choose exactly one of --head-camera-calibration or --uncalibrated-absolute-test")
        if args.head_camera_calibration:
            if not args.head_camera_calibration.is_file(): raise ValueError("head-camera-calibration file does not exist")
            from xrobotoolkit_teleop.vendor.xr_robot1_right_rm65.rm65_teleop.absolute_retarget import HeadCameraCalibration
            HeadCameraCalibration.load(args.head_camera_calibration)
    elif args.uncalibrated_absolute_test or args.head_camera_calibration:
        raise ValueError("absolute calibration options require --absolute-control pose or wrist")
    if args.teleop_only:
        log_dir = args.log_dir or (
            PROJECT_ROOT / "outputs" / "teleop_test" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        return Path(log_dir).expanduser().resolve()
    root = Path(args.dataset_root).expanduser().resolve()
    if root.exists(): raise FileExistsError(f"dataset root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    return root

def _print_plan(args, root):
    print("Verified right RM65 " + ("teleoperation preflight" if args.teleop_only else "LeRobot collection preflight"))
    print(f"mode={'DRY_RUN' if args.dry_run else 'REAL_HARDWARE'}")
    if args.teleop_only:
        print("recording=DISABLED cameras=DISABLED vr_preview=DISABLED")
        print(f"diagnostic_log_dir={root}")
    else:
        print(f"dataset_root={root}")
        print(f"repo_id={args.repo_id}")
    print("arm_ip=192.168.1.19")
    print(f"arm_response={args.arm_response} max_joint_speed_deg_s=180")
    if args.absolute_control == "off":
        print(f"forward_reach_gain={args.forward_reach_gain} right_reach_gain={args.right_reach_gain} vertical_reach_gain={args.vertical_reach_gain}; left/backward_gain=1.0; from_reference")
    else:
        print("absolute_position_gain=1.0; incremental forward-reach gain bypassed")
        print("absolute_joint_speed_cap_deg_s=60; absolute_feedback_lead_cap_deg=5; full_target_ik_before_follow")
    print("translation_timing=" + {
        "original": "original_fixed_90hz_step",
        "faster": "elapsed_capped_1.5_cycles",
        "responsive": "elapsed_capped_2_cycles",
    }[args.arm_response])
    if args.absolute_control == "off":
        if args.wrist_control == "pose":
            print("pose_orientation_gain=1.0; all_three_rotation_axes; whole_arm_ik=J1..J6; normalized_tcp_frames=True")
            print("pose_command_lead_cap_deg=4.5; pose_feedback_lead_cap_deg=5; pose_ik_fk_tolerance=1mm/1deg; orientation_deadband_deg=1")
        elif args.wrist_control == "auto-ik":
            print("whole_arm_ik=J4/J6_gestures; j5_control=direct_joint; j4_gain=1; j5_j6_gain=1.5; j6_sign=reversed_from_config")
            print("wrist_tcp_position=hold_for_J4_J6; J5_changes_tcp_position")
            print("wrist_ik_orientation_mapping=virtual_joint_fk; wrist_axis_switch=hand_intent")
            print("wrist_ik_feedback_lead_deg=J1-J3:5,J4-J6:15; margin_deg=0.5; fk_tolerance=1mm/1deg")
        else:
            print("wrist_input_gain=1.5 translation_gain=1.0")
        print(f"wrist_control={args.wrist_control}")
    else:
        print("wrist_input_gain=1.0; gesture switching bypassed")
    print(f"ray_j4_control={args.ray_j4_control} ray_j4_sign={args.ray_j4_sign}")
    print("gripper_control=Grip+Trigger; keep_grip_held; toggle_once_then_release_trigger")
    print(f"tracking_frame={args.tracking_frame}")
    print(f"grip_sync={args.grip_sync}")
    print("control_baseline=morning_30cm_directional_reanchor")
    print(f"grip_sync_speed_deg_s={args.grip_sync_speed_deg_s} grip_sync_max_distance_m=0.30 (persistent position mode)")
    print(f"position_fault_recovery={args.position_fault_recovery} (persistent position mode)")
    print(f"grip_sync_orientation_relax_deg={args.grip_sync_orientation_relax_deg}")
    print(f"absolute_control={args.absolute_control} head_camera_calibration={args.head_camera_calibration}")
    if args.uncalibrated_absolute_test:
        print("absolute_registration=first_grip_uncalibrated; camera_frame_target=unavailable")
    print(f"tracking_fault_recovery={args.tracking_fault_recovery}")
    print("teleop_source=XR_Robot1/rm65_xr_teleop_clean (frozen copy)")
    if args.teleop_only:
        print("motion_commands=0 (dry-run)" if args.dry_run else "motion_commands=RIGHT_ARM_ENABLED_AFTER_PREFLIGHT")
        return
    print("observation.state: float32[7]")
    print("action: float32[7]")
    print("order=right_q1..q6,right_gripper")
    print("units=joints_rad,gripper_0_to_100")
    print("action_source=successful_physical_command")
    print("camera_keys=top,right_wrist")
    print(f"top_camera={args.top_camera} expected_serial={TOP_CAMERA_SERIAL}")
    print(f"right_wrist_camera={args.right_wrist_camera} expected_serial={RIGHT_WRIST_CAMERA_SERIAL}")
    print("left_wrist_camera=DISABLED")
    print(f"camera_fps={args.camera_fps}")
    print(
        f"vr_head_camera={'ENABLED' if getattr(args, 'vr_preview', True) else 'DISABLED'} "
        f"control_port={getattr(args, 'vr_control_port', 13579)} source=top"
    )
    print(f"image_writer_threads={args.image_writer_threads}")
    print(f"video_codec=libsvtav1 video_crf={args.video_crf}")
    print(f"dataset_fps=30\nnum_episodes={args.num_episodes or 'unlimited'}")
    print("recording_start=Enter_prepare_then_Grip_follow_ready; countdown=DISABLED; positioning_frames=excluded")
    print("motion_commands=0 (dry-run)" if args.dry_run else "motion_commands=RIGHT_ARM_ENABLED_AFTER_PREFLIGHT")

def _wait_cameras(cameras, timeout_s=8):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        errors = [e for c in cameras.values() if (e := c.health_error(max_age_s=.5))]
        if not errors: return
        time.sleep(.05)
    raise RuntimeError("camera preflight failed: " + "; ".join(errors))

def _instructions(count, wrist_control="buttons", ray_j4_control=False, grip_sync="reanchor", absolute_control="off", uncalibrated_absolute_test=False, teleop_only=False, position_fault_recovery="release"):
    if teleop_only:
        print("\n========== 单右臂遥操作测试 ==========")
        print("相机和数据集采集已关闭；仅保存控制诊断日志。Esc 或 Ctrl+C 退出。")
    else:
        print("\n========== 单右臂 Episode 采集 ==========")
        print("Enter：准备 | PICO摇杆向前：准备/开始 | 按住 Grip：定位完成后自动录制，无倒计时 | → / S 或摇杆向右：保存 | ← 或摇杆向左：作废/取消准备 | Esc：退出")
        print("INVALID 表示本条数据有异常；按 → / S 将整条剔除，不保存异常数据，并显示原因。")
        print("先松开 Grip 再按 Enter；定位期间不记录，中途松开 Grip 就等待再次按住。")
        print("完成同步或重新定位、进入跟随后立即录制；看到 EPISODE RECORDING 后开始抓取。")
        print(f"目标：{count or '不限数量'} 条；仅记录顶部和右腕两路相机，左腕相机已停用。")
    print("夹爪：右手 Grip + Trigger 两键控制；保持 Grip 跟随时，按 Trigger 切换开/关，不用按 A。")
    print("Trigger 每次按下只切换一次；松开 Trigger 再按可反向切换，Grip 可以一直按住。")
    if absolute_control != "off":
        if uncalibrated_absolute_test:
            print("试验模式：首次 Grip 固定手柄与 TCP 的对应关系；未标定相机坐标系。")
        else:
            print("手柄相对头显的绝对位姿直接映射到相机坐标；首次 Grip 不归零。")
        print("松开 Grip 停止并保持，再按按同一绝对映射限速同步。")
        if absolute_control == "pose":
            print("按住 Grip 跟随手柄相对头显的位置和姿态；IK 无解时暂停，移回可达位置后自动恢复，绝对映射不变。")
        else:
            print("按住 Grip 由射线姿态控制 J4-J6，J1-J3 保持；TCP 数值容差 1 mm，姿态容差 1°。")
        print("同步距离过大或姿态不可达时保持，松开 Grip 即停止更新。")
        print("=========================================\n", flush=True)
        return
    if grip_sync == "persistent":
        print("首次按 Grip 保存手柄/头显/机械臂位置关系；重按先限速同步，看到 FOLLOW 后实时跟随。")
        print("同步中目标随当前手位更新，到位后转入跟随；松开 Grip 随时停止并保持。")
        if position_fault_recovery == "reanchor":
            print("目标超过 30 cm、不可达或跟随停滞时先停止，再按当前手位/末端位置自动重建对应关系；保持 Grip 即可续跟。")
        else:
            print("目标跨度超过 30 cm、不可达或同步超时会停止本次跟随；松开 Grip 后可重试。")
    if wrist_control == "auto-ik":
        print("按住 Grip：沿用原平移/转腕手势切换；抬压手腕直接控制 J5，其他五个关节保持原指令。")
        print("左右摆射线控制 J4 动作意图（需 --ray-j4-control），拧腕控制 J6，J6 方向已反向。")
        print("J4/J6 沿用整臂 IK 配合保持夹爪位置；J5 单独转动会改变夹爪位置，切回 IK 时就地接管。")
        print("J4 输入 1 倍，J5/J6 输入 1.5 倍。")
        print("明显平移后停止转腕、丢弃旧目标，停稳后从实际姿态恢复平移。")
        print("松开 Grip 后，长按 B 1 秒再释放回 Home。")
    elif wrist_control == "pose":
        print("按住 Grip：手柄位置和三轴朝向同时控制末端，由 J1-J6 整臂 IK 配合完成。")
        print("原地转手柄时保持末端位置目标；不再切换或选中单个腕部关节。")
        print("每次接管以当前手柄/末端朝向重新对齐，旋转比例 1:1；松开 Grip 停止。")
        print("重按时先按原位置映射同步，完成后重新对齐朝向并进入整臂跟随。")
        print("松开 Grip 后，长按 B 1 秒再释放回 Home。")
    elif wrist_control == "auto":
        joints = "J4/J5/J6" if ray_j4_control else "J5/J6"
        held = "J1-J3" if ray_j4_control else "J1-J4"
        print(f"按住 Grip：平移控制末端；转射线自动切到 {joints}；明显平移后切回。")
        print(f"切换时短暂减速并重置参考点；关节模式下 {held} 保持，末端位置可能随转动改变。")
        print("无需 Trigger 或 B 切换关节；松开 Grip 停止。")
        print("松开 Grip 后，长按 B 1 秒再释放回 Home。")
    else:
        print("按住 Grip 平移；Grip+Trigger 控制选中的 J5/J6；松开后短按 B 切换。")
    print("=========================================\n", flush=True)

def _configure_lerobot_video_crf(video_crf, *, dataset_module=None, base_encoder=None):
    """Apply the requested AV1 quality to LeRobot's episode encoder.

    LeRobot 0.3.3 calls a module-level ``encode_video_frames`` function without
    exposing its CRF through ``LeRobotDataset.create``.  Keep the override
    local to this collection process and retain every other encoder argument.
    """
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

def _make_controller(args, log_dir):
    from xrobotoolkit_teleop.hardware.right_rm65_collection_controller import VerifiedRightRM65CollectionController
    return VerifiedRightRM65CollectionController(
        log_dir=log_dir, arm_response=args.arm_response,
        forward_reach_gain=args.forward_reach_gain,
        right_reach_gain=args.right_reach_gain,
        vertical_reach_gain=args.vertical_reach_gain,
        grip_sync_speed_deg_s=args.grip_sync_speed_deg_s,
        position_fault_recovery=args.position_fault_recovery,
        wrist_control=args.wrist_control,
        ray_j4_control=args.ray_j4_control,
        ray_j4_sign=args.ray_j4_sign,
        tracking_frame=args.tracking_frame,
        grip_sync=args.grip_sync,
        absolute_control=args.absolute_control,
        head_camera_calibration=args.head_camera_calibration,
        allow_uncalibrated_absolute=args.uncalibrated_absolute_test,
        grip_sync_orientation_relax_deg=args.grip_sync_orientation_relax_deg,
        resume_tracking_faults=args.tracking_fault_recovery == "auto",
    )


def _run_teleop_only(args, log_dir):
    # Import no camera readers, dataset sink, video encoder or LeRobot here.
    from xrobotoolkit_teleop.collection.keyboard import TerminalKeyboard
    from xrobotoolkit_teleop.collection.supervisor import SupervisorNotifier

    notifier = SupervisorNotifier.from_environment()
    controller = monitor = None
    keyboard_errors = []
    try:
        with TerminalKeyboard() as keyboard:
            controller = _make_controller(args, log_dir)
            controller.start_xr_collection()

            def watch_keyboard():
                try:
                    while not controller.runtime.stop_event.is_set():
                        if keyboard.poll(0.05) == "ESC":
                            controller.request_stop()
                            return
                        notifier.heartbeat()
                except BaseException as exc:
                    keyboard_errors.append(exc)
                    controller.request_stop()

            _instructions(None, args.wrist_control, args.ray_j4_control,
                          args.grip_sync, args.absolute_control,
                          args.uncalibrated_absolute_test, teleop_only=True,
                          position_fault_recovery=args.position_fault_recovery)
            monitor = threading.Thread(target=watch_keyboard, name="right-teleop-keyboard", daemon=True)
            monitor.start()
            notifier.ready()
            controller.run()
            if keyboard_errors:
                raise RuntimeError("teleoperation keyboard failed") from keyboard_errors[0]
    except KeyboardInterrupt:
        print("\nStopping right-arm teleoperation...", flush=True)
        return 130
    finally:
        if controller is not None:
            controller.request_stop()
            if monitor is not None and monitor.ident is not None:
                monitor.join(timeout=1.0)
            try:
                # Runtime.run normally shuts down in finally; also cover a
                # reader/startup failure before entering that runtime.
                controller.controller.shutdown()
            finally:
                controller.close()
    return 0


def _run_real(args, root):
    from xrobotoolkit_teleop.collection.audit import JsonlAuditWriter
    from xrobotoolkit_teleop.collection.cameras import OpenCVCameraReader
    from xrobotoolkit_teleop.collection.recovery import RecoverableLeRobotDatasetSink
    from xrobotoolkit_teleop.collection.episode import EpisodeManager
    from xrobotoolkit_teleop.collection.frame_builder import RightArmFrameBuilder
    from xrobotoolkit_teleop.collection.keyboard import TerminalKeyboard
    from xrobotoolkit_teleop.collection.preview import CameraPreviewKeyboard
    from xrobotoolkit_teleop.collection.runtime import DualRM65CollectionRuntime
    from xrobotoolkit_teleop.collection.supervisor import SupervisorNotifier
    from xrobotoolkit_teleop.collection.vr_video import VRHeadCameraStreamer
    from xrobotoolkit_teleop.hardware.right_rm65_collection_controller import CachedGripperSampler

    class TwoCameraLeRobotDatasetSink(RecoverableLeRobotDatasetSink):
        CAMERA_KEYS = ("observation.images.top", "observation.images.right_wrist")
        KEEP_DISCARDED_BACKUP = False

        def checkpoint(self, reasons):
            with notifier.saving():
                return super().checkpoint(reasons)

        def accept_episode(self, task):
            print("EPISODE SAVING: 正在编码头部和右腕视频，请等待 EPISODE SAVED 后再开始下一条。", flush=True)
            with notifier.saving():
                super().accept_episode(task)

    class TwoCameraRightArmFrameBuilder(RightArmFrameBuilder):
        CAMERA_ROLES = ("top", "right_wrist")

    notifier = SupervisorNotifier.from_environment()
    cameras = {}; controller = sampler = runtime = episodes = audit = preview = sink = vr_stream = None; errors = []
    try:
        top_camera = _resolve_camera_with_retry(args.top_camera, TOP_CAMERA_SERIAL, "top")
        right_wrist_camera = _resolve_camera_with_retry(args.right_wrist_camera, RIGHT_WRIST_CAMERA_SERIAL, "right_wrist")
        if top_camera == right_wrist_camera:
            raise RuntimeError("top and right-wrist cameras resolved to the same node")
        if args.left_wrist_camera:
            print(f"[CAMERA NOTICE] --left-wrist-camera={args.left_wrist_camera} ignored; left wrist capture is disabled.")
        print(f"[CAMERA VERIFIED] top={top_camera} serial={TOP_CAMERA_SERIAL}")
        print(f"[CAMERA VERIFIED] right_wrist={right_wrist_camera} serial={RIGHT_WRIST_CAMERA_SERIAL}")
        with TerminalKeyboard() as keyboard:
            cameras = {role: OpenCVCameraReader(role, _device(device), width=args.camera_width, height=args.camera_height, fps=args.camera_fps)
                       for role, device in (("top", top_camera), ("right_wrist", right_wrist_camera))}
            for camera in cameras.values(): camera.start()
            _wait_cameras(cameras)
            if not args.no_camera_preview:
                if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
                    # Initialize Qt before LeRobot/PyAV: reversing this order can
                    # deadlock OpenCV window creation on the collection machine.
                    preview = CameraPreviewKeyboard(
                        keyboard, cameras, lambda: bool(episodes and episodes.recording),
                        status=lambda: runtime.operator_state if runtime else "READY",
                        notice=lambda: runtime.operator_notice if runtime else "")
                    print("[CAMERA PREVIEW] Head / right wrist preview enabled.", flush=True)
                else:
                    print("[CAMERA PREVIEW] No desktop display; cameras still record.", flush=True)
            if args.vr_preview:
                vr_stream = VRHeadCameraStreamer(
                    cameras["top"],
                    wrist_camera=cameras["right_wrist"],
                    display_status=lambda: runtime.operator_display if runtime else {
                        "state": "READY", "saved": 0, "target": args.num_episodes,
                        "instruction": "采集程序启动中，请等待", "error": "",
                    },
                    control_port=args.vr_control_port,
                )
                vr_stream.start()
            _configure_lerobot_video_crf(args.video_crf)
            sink = TwoCameraLeRobotDatasetSink.create(root=root, repo_id=args.repo_id, fps=30,
                image_height=args.camera_height, image_width=args.camera_width, dimension_names=RIGHT_DIMENSIONS,
                image_writer_threads=args.image_writer_threads)
            audit = JsonlAuditWriter(root / "audit" / f"session-{datetime.now():%Y%m%d-%H%M%S}.jsonl")
            episodes = EpisodeManager(sink, task=args.task, backup_valid=True)
            controller = _make_controller(args, root / "teleop_logs")
            controller.start_xr_collection()
            def control():
                try: controller.run()
                except BaseException as exc: errors.append(exc)
            thread = threading.Thread(target=control, name="verified-right-rm65-control")
            thread.start()
            sampler = CachedGripperSampler(controller.grippers); sampler.start()
            runtime = DualRM65CollectionRuntime(controller=controller, cameras=cameras,
                gripper_sampler=sampler, episodes=episodes, audit=audit,
                frame_builder=TwoCameraRightArmFrameBuilder(), dataset_fps=30,
                max_episodes=args.num_episodes, heartbeat=notifier.heartbeat,
                wait_for_grip_follow=True, discard_invalid_on_save=True,
                reject_on_cadence_violation=False,
                require_fresh_camera_frames=False)
            notifier.ready()
            _instructions(args.num_episodes, args.wrist_control, args.ray_j4_control, args.grip_sync, args.absolute_control, args.uncalibrated_absolute_test,
                          position_fault_recovery=args.position_fault_recovery)
            runtime.run(preview if preview is not None else keyboard)
            print(f"Collection finished: saved_episodes={runtime.saved_episodes}")
            if (vr_stream and runtime.max_episodes is not None
                    and runtime.saved_episodes >= runtime.max_episodes):
                controller.request_stop()
                time.sleep(3.0)  # Keep the final save receipt visible in the headset.
    finally:
        # Stop motion before potentially slow exit checkpointing. A checkpoint
        # error must not skip controller/camera cleanup.
        if controller: controller.request_stop()
        if 'thread' in locals(): thread.join(timeout=8)
        try:
            if runtime: runtime.close()
            elif episodes: episodes.close()
        finally:
            try:
                if controller: controller.close()
            finally:
                try:
                    if sampler: sampler.close()
                    if preview: preview.close()
                    if vr_stream: vr_stream.close()
                    for camera in cameras.values(): camera.close()
                    if audit: audit.close()
                finally:
                    if sink: sink.dataset.stop_image_writer()
    if errors: raise errors[0]
    return 0

def main(argv=None):
    args = build_parser().parse_args(argv); root = _validate(args); _print_plan(args, root)
    if args.dry_run:
        return 0
    return _run_teleop_only(args, root) if args.teleop_only else _run_real(args, root)

if __name__ == "__main__": raise SystemExit(main())
