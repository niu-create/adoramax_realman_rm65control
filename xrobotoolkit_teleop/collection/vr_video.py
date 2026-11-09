"""Low-latency head-camera streaming for XRoboToolkit Remote Vision.

The XRoboToolkit headset app opens a control connection to TCP 13579 and asks
the robot PC to connect back to its H.264 receiver on TCP 12345.  This module
implements that protocol around an existing collection camera reader so the
camera device is opened only once.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import struct
import threading
import time
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

import numpy as np


MAX_CONTROL_BODY = 2 * 1024 * 1024
MAX_VIDEO_WIDTH = 2560
MAX_VIDEO_HEIGHT = 1440
MAX_VIDEO_FPS = 30
DEFAULT_VIDEO_BITRATE = 4_000_000


@dataclass(frozen=True)
class CameraRequest:
    width: int
    height: int
    fps: int
    bitrate: int
    enable_mv_hevc: int
    render_mode: int
    port: int
    camera: str
    ip: str

    @property
    def stereo_layout(self) -> str:
        # XRoboToolkit's built-in ZED profile expects one left/right image in
        # each half.  A monocular head camera is duplicated into those halves.
        return "side_by_side" if self.camera.upper() == "ZED" else "mono"


def encode_control_message(command: str, data: bytes = b"") -> bytes:
    command_bytes = command.encode("utf-8")
    body = struct.pack("<I", len(command_bytes)) + command_bytes + struct.pack("<I", len(data)) + data
    return struct.pack(">I", len(body)) + body


def decode_control_body(body: bytes) -> tuple[str, bytes]:
    if len(body) < 8:
        raise ValueError("control body is too short")
    command_length = struct.unpack_from("<I", body, 0)[0]
    data_length_offset = 4 + command_length
    if command_length > 1024 or data_length_offset + 4 > len(body):
        raise ValueError("invalid control command length")
    data_length = struct.unpack_from("<I", body, data_length_offset)[0]
    data_offset = data_length_offset + 4
    if data_length > MAX_CONTROL_BODY or data_offset + data_length != len(body):
        raise ValueError("invalid control payload length")
    command = body[4:data_length_offset].decode("utf-8")
    return command, body[data_offset:]


def decode_camera_request(data: bytes) -> CameraRequest:
    if len(data) < 33 or data[:2] != b"\xca\xfe" or data[2] != 1:
        raise ValueError("invalid XRoboToolkit camera request header")
    values = struct.unpack_from("<7i", data, 3)
    offset = 31

    def read_string() -> str:
        nonlocal offset
        if offset >= len(data):
            raise ValueError("camera request string length is missing")
        length = data[offset]
        offset += 1
        if offset + length > len(data):
            raise ValueError("camera request string is truncated")
        value = data[offset : offset + length].decode("utf-8")
        offset += length
        return value

    camera = read_string()
    target_ip = read_string()
    if offset != len(data):
        raise ValueError("camera request contains trailing bytes")
    return CameraRequest(*values, camera=camera, ip=target_ip)


def _validated_request(request: CameraRequest, peer_ip: str) -> CameraRequest:
    if request.width <= 0 or request.height <= 0:
        raise ValueError("VR requested a non-positive video resolution")
    if request.width > MAX_VIDEO_WIDTH or request.height > MAX_VIDEO_HEIGHT:
        raise ValueError(
            f"VR resolution {request.width}x{request.height} exceeds " f"{MAX_VIDEO_WIDTH}x{MAX_VIDEO_HEIGHT}"
        )
    if request.width % 2 or request.height % 2:
        raise ValueError("VR video width and height must be even")
    if not 1 <= request.port <= 65535:
        raise ValueError("VR video port is out of range")
    try:
        target = ipaddress.ip_address(request.ip)
        peer = ipaddress.ip_address(peer_ip)
    except ValueError as exc:
        raise ValueError("VR supplied an invalid IPv4 address") from exc
    if target.version != 4 or peer.version != 4:
        raise ValueError("XRoboToolkit Remote Vision requires IPv4")
    # Do not let a control client turn the collector into a network relay.
    # XRoboToolkit reports its own local address, which must match its peer.
    if target != peer:
        raise ValueError(f"VR video target {target} does not match control peer {peer}")
    return CameraRequest(
        width=request.width,
        height=request.height,
        fps=min(max(1, request.fps), MAX_VIDEO_FPS),
        bitrate=min(max(250_000, request.bitrate or DEFAULT_VIDEO_BITRATE), 12_000_000),
        enable_mv_hevc=request.enable_mv_hevc,
        render_mode=request.render_mode,
        port=request.port,
        camera=request.camera,
        ip=str(target),
    )


def _letterbox(image: np.ndarray, width: int, height: int, *, cv_module=None) -> np.ndarray:
    if cv_module is None:
        import cv2 as cv_module
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(2, min(width, int(round(source_width * scale))))
    resized_height = max(2, min(height, int(round(source_height * scale))))
    resized_width -= resized_width % 2
    resized_height -= resized_height % 2
    resized = cv_module.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv_module.INTER_AREA if scale < 1 else cv_module.INTER_LINEAR,
    )
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    output[y : y + resized_height, x : x + resized_width] = resized
    return output


def prepare_vr_frame(
    image: np.ndarray,
    width: int,
    height: int,
    stereo_layout: str,
    *,
    cv_module=None,
) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("head camera frame must be an RGB uint8 image")
    if stereo_layout == "mono":
        return _letterbox(image, width, height, cv_module=cv_module)
    if stereo_layout != "side_by_side" or width % 2:
        raise ValueError(f"unsupported VR stereo layout: {stereo_layout}")
    eye = _letterbox(image, width // 2, height, cv_module=cv_module)
    return np.concatenate((eye, eye), axis=1)


def prepare_collection_vr_frame(
    head_image: np.ndarray,
    wrist_image: np.ndarray,
    width: int,
    height: int,
    stereo_layout: str,
    display: dict,
) -> np.ndarray:
    """Show both cameras and collector guidance in each headset eye."""
    from PIL import Image, ImageDraw

    if stereo_layout not in ("mono", "side_by_side"):
        raise ValueError(f"unsupported VR stereo layout: {stereo_layout}")
    eye_width = width // 2 if stereo_layout == "side_by_side" else width
    if eye_width < 2 or height < 2:
        raise ValueError("VR display is too small")
    band_height = max(96, min(height // 2, int(height * 0.38)))
    video_height = height - band_height
    gap = max(2, eye_width // 200)
    left_width = (eye_width - gap) // 2
    right_width = eye_width - gap - left_width
    eye = np.zeros((height, eye_width, 3), dtype=np.uint8)
    eye[:video_height, :left_width] = _letterbox(head_image, left_width, video_height)
    eye[:video_height, left_width + gap:] = _letterbox(wrist_image, right_width, video_height)

    state = str(display.get("state", "READY"))
    error = str(display.get("error", ""))
    validity = str(display.get("validity", "待采集"))
    save_result = str(display.get("save_result", "尚未保存"))
    background = (135, 20, 20) if error or validity == "无效" else (20, 32, 45)
    eye[video_height:] = background
    canvas = Image.fromarray(eye)
    draw = ImageDraw.Draw(canvas)
    font_size = max(14, min(eye_width // 28, height // 18, 28))
    font = _vr_status_font(font_size)
    line_height = font_size + 5
    margin = max(5, eye_width // 100)
    draw.text((margin, 3), "头部 HEAD", font=font, fill="white")
    draw.text((left_width + gap + margin, 3), "右腕 WRIST", font=font, fill="white")
    target = display.get("target") or "∞"
    lines = [
        f"状态：{state} | 数据：{validity}",
        f"保存结果：{save_result} | 已保存：{display.get('saved', 0)}/{target}",
    ]
    message = str(display.get("save_message", ""))
    if error:
        message = f"错误：{error}"
    if message:
        lines.append(message)
    lines.append(str(display.get("instruction", "")))
    y = video_height + 3
    for line in lines:
        if y + line_height > height:
            break
        if line.startswith("保存结果："):
            if "成功" in line:
                color = (100, 255, 120)
            elif "失败" in line or "未保存" in line:
                color = (255, 150, 150)
            elif "中" in line:
                color = (255, 220, 120)
            else:
                color = "white"
        elif line.startswith("错误："):
            color = (255, 235, 235)
        else:
            color = "white"
        remaining = line
        while remaining and y + line_height <= height:
            low, high = 1, len(remaining)
            while low < high:
                middle = (low + high + 1) // 2
                if draw.textbbox((0, 0), remaining[:middle], font=font)[2] <= eye_width - 2 * margin:
                    low = middle
                else:
                    high = middle - 1
            draw.text((margin, y), remaining[:low], font=font, fill=color)
            remaining = remaining[low:]
            y += line_height
    eye = np.asarray(canvas)
    return np.concatenate((eye, eye), axis=1) if stereo_layout == "side_by_side" else eye


@lru_cache(maxsize=16)
def _vr_status_font(size: int):
    from PIL import ImageFont

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    return ImageFont.truetype(str(font_path), size) if font_path.is_file() else ImageFont.load_default()


class H264FrameEncoder:
    """Encode RGB frames as Annex-B H.264 access units without frame delay."""

    def __init__(self, width: int, height: int, fps: int, bitrate: int):
        import av

        self._av = av
        self._codec = av.CodecContext.create("libx264", "w")
        self._codec.width = int(width)
        self._codec.height = int(height)
        self._codec.pix_fmt = "yuv420p"
        self._codec.time_base = Fraction(1, int(fps))
        self._codec.framerate = Fraction(int(fps), 1)
        self._codec.bit_rate = int(bitrate)
        self._codec.max_b_frames = 0
        self._codec.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            "x264-params": (f"keyint={fps}:min-keyint={fps}:scenecut=0:" "repeat-headers=1:annexb=1"),
        }
        self._codec.open()
        self._pts = 0

    @staticmethod
    def probe() -> None:
        import av

        av.CodecContext.create("libx264", "w")

    def encode(self, image: np.ndarray) -> list[bytes]:
        frame = self._av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = self._pts
        frame.time_base = self._codec.time_base
        self._pts += 1
        return [bytes(packet) for packet in self._codec.encode(frame)]

    def close(self) -> None:
        self._codec = None


class VRHeadCameraStreamer:
    """Serve cached head-camera frames to XRoboToolkit Remote Vision."""

    def __init__(
        self,
        camera,
        *,
        wrist_camera=None,
        display_status=None,
        control_host: str = "0.0.0.0",
        control_port: int = 13579,
        socket_module=socket,
        clock=time.monotonic,
    ):
        self.camera = camera
        self.wrist_camera = wrist_camera
        self.display_status = display_status
        self.control_host = str(control_host)
        self.control_port = int(control_port)
        self._socket = socket_module
        self._clock = clock
        self._stop = threading.Event()
        self._stream_stop = threading.Event()
        self._ready = threading.Event()
        self._server: Optional[socket.socket] = None
        self._control_client: Optional[socket.socket] = None
        self._control_thread: Optional[threading.Thread] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._startup_error: Optional[BaseException] = None
        self._stream_error: Optional[BaseException] = None
        self._audio_request_id = ""

    @property
    def stream_error(self) -> Optional[BaseException]:
        return self._stream_error

    def start(self, timeout_s: float = 3.0) -> None:
        if not 0 <= self.control_port <= 65535:
            raise ValueError("VR control port must be in [0, 65535]")
        H264FrameEncoder.probe()
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="xr-remote-vision-control",
            daemon=True,
        )
        self._control_thread.start()
        if not self._ready.wait(timeout_s):
            self.close()
            raise RuntimeError("VR Remote Vision control service startup timed out")
        if self._startup_error is not None:
            error = self._startup_error
            self.close()
            raise RuntimeError(
                f"VR Remote Vision could not listen on " f"{self.control_host}:{self.control_port}: {error}"
            ) from error
        print(
            f"[VR VIDEO] Remote Vision ready on " f"{self.control_host}:{self.control_port}; select ZEDMINI and Listen",
            flush=True,
        )

    def _control_loop(self) -> None:
        try:
            server = self._socket.socket(self._socket.AF_INET, self._socket.SOCK_STREAM)
            self._server = server
            server.setsockopt(self._socket.SOL_SOCKET, self._socket.SO_REUSEADDR, 1)
            server.bind((self.control_host, self.control_port))
            # Port zero is useful for isolated protocol tests.
            self.control_port = int(server.getsockname()[1])
            server.listen(1)
            server.settimeout(0.25)
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        while not self._stop.is_set():
            try:
                client, peer = server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            client.settimeout(0.25)
            self._control_client = client
            print(f"[VR VIDEO] headset control connected: {peer[0]}", flush=True)
            try:
                self._serve_control_client(client, peer[0])
            except (ConnectionError, OSError, ValueError) as exc:
                if not self._stop.is_set():
                    print(f"[VR VIDEO] control connection ended: {exc}", flush=True)
            finally:
                self._stop_stream()
                try:
                    client.close()
                except OSError:
                    pass
                self._control_client = None

    def _serve_control_client(self, client: socket.socket, peer_ip: str) -> None:
        next_ping = self._clock() + 1.0
        supports_heartbeat = False
        buffer = bytearray()
        while not self._stop.is_set():
            try:
                chunk = client.recv(65536)
                if not chunk:
                    return
                buffer.extend(chunk)
            except TimeoutError:
                pass
            while len(buffer) >= 4:
                body_length = struct.unpack_from(">I", buffer, 0)[0]
                if body_length <= 0 or body_length > MAX_CONTROL_BODY:
                    raise ValueError(f"invalid control frame length: {body_length}")
                if len(buffer) < 4 + body_length:
                    break
                body = bytes(buffer[4 : 4 + body_length])
                del buffer[: 4 + body_length]
                command, data = decode_control_body(body)
                if command == "AUDIO_SESSION":
                    supports_heartbeat = True
                    self._audio_request_id = data.decode("ascii", errors="ignore")[:64]
                elif command == "OPEN_CAMERA":
                    request = _validated_request(decode_camera_request(data), peer_ip)
                    self._start_stream(request)
                    if supports_heartbeat:
                        self._send_audio_config(client, request)
                elif command == "CLOSE_CAMERA":
                    self._stop_stream()
                elif command == "PONG":
                    pass
            now = self._clock()
            if supports_heartbeat and now >= next_ping:
                client.sendall(encode_control_message("PING", struct.pack(">d", now)))
                next_ping = now + 1.0

    def _send_audio_config(self, client: socket.socket, request: CameraRequest) -> None:
        payload = json.dumps(
            {
                "schema": "g1_wuji_audio_ports_v2",
                "audio_request_id": self._audio_request_id,
                "audio_stream_port": 0,
                "microphone_upload_port": 0,
                "sample_rate": 16000,
                "channels": 1,
                "sample_format": "s16le",
                "video_projection": "flat",
                "video_stereo_layout": request.stereo_layout,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        client.sendall(encode_control_message("AUDIO_CONFIG", payload))

    def _start_stream(self, request: CameraRequest) -> None:
        self._stop_stream()
        self._stream_error = None
        self._stream_stop.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(request,),
            name="xr-remote-vision-h264",
            daemon=True,
        )
        self._stream_thread.start()

    def _stream_loop(self, request: CameraRequest) -> None:
        print(
            f"[VR VIDEO] connecting video to {request.ip}:{request.port} "
            f"{request.width}x{request.height}@{request.fps} "
            f"layout={request.stereo_layout}",
            flush=True,
        )
        video_socket = None
        encoder = None
        try:
            # The headset starts its video listener just before OPEN_CAMERA.  On
            # slower headsets the listener can become ready a few hundred
            # milliseconds after the control request, so tolerate that race.
            connect_error = None
            for _attempt in range(10):
                if self._stop.is_set() or self._stream_stop.is_set():
                    return
                try:
                    video_socket = self._socket.create_connection((request.ip, request.port), timeout=1.0)
                    break
                except OSError as exc:
                    connect_error = exc
                    self._stream_stop.wait(0.2)
            if video_socket is None:
                raise ConnectionError(
                    f"headset video listener {request.ip}:{request.port} was not ready"
                ) from connect_error
            video_socket.setsockopt(self._socket.IPPROTO_TCP, self._socket.TCP_NODELAY, 1)
            video_socket.settimeout(2.0)
            encoder = H264FrameEncoder(
                request.width,
                request.height,
                request.fps,
                request.bitrate,
            )
            last_timestamp = -1.0
            next_frame = self._clock()
            while not self._stop.is_set() and not self._stream_stop.is_set():
                now = self._clock()
                if now < next_frame:
                    self._stream_stop.wait(min(next_frame - now, 0.02))
                    continue
                sample = self.camera.snapshot()
                if sample.monotonic_time <= last_timestamp:
                    self._stream_stop.wait(0.002)
                    continue
                last_timestamp = sample.monotonic_time
                if self.wrist_camera is not None:
                    wrist_sample = self.wrist_camera.snapshot()
                    display = self.display_status() if self.display_status else {}
                    image = prepare_collection_vr_frame(
                        sample.image, wrist_sample.image, request.width,
                        request.height, request.stereo_layout, display,
                    )
                else:
                    image = prepare_vr_frame(
                        sample.image, request.width, request.height,
                        request.stereo_layout,
                    )
                for packet in encoder.encode(image):
                    video_socket.sendall(struct.pack(">I", len(packet)) + packet)
                next_frame = max(next_frame + 1.0 / request.fps, self._clock())
            print("[VR VIDEO] video stream stopped", flush=True)
        except BaseException as exc:
            self._stream_error = exc
            if not self._stop.is_set() and not self._stream_stop.is_set():
                print(f"[VR VIDEO] video stream failed: {exc}", flush=True)
        finally:
            if encoder is not None:
                encoder.close()
            if video_socket is not None:
                try:
                    video_socket.close()
                except OSError:
                    pass

    def _stop_stream(self) -> None:
        self._stream_stop.set()
        thread = self._stream_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        self._stream_thread = None

    def close(self) -> None:
        self._stop.set()
        self._stop_stream()
        for sock in (self._control_client, self._server):
            if sock is not None:
                try:
                    sock.shutdown(self._socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
        thread = self._control_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        self._control_thread = None
