"""Absolute PICO-head to robot-TCP retargeting from a fixed registration.

No actuator calls occur here. All transforms map child coordinates into parent
coordinates. Camera-to-work calibration additionally enables camera-frame targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .transforms import (
    invert_transform,
    quaternion_xyzw_to_rotation,
    rotation_angle_rad,
    rotation_log,
    validate_rotation,
)


def _transform(value, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-9):
        raise ValueError(f"{name} has an invalid bottom row")
    validate_rotation(matrix[:3, :3])
    return matrix.copy()


def _pose7_transform(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("XR pose must contain seven finite values")
    result = np.eye(4)
    result[:3, :3] = quaternion_xyzw_to_rotation(pose[3:7])
    result[:3, 3] = pose[:3]
    return result


def head_controller_transform(sample) -> np.ndarray:
    """Return the controller/ray-origin pose in the PICO headset frame."""
    if sample.head_pose_xyz_xyzw is None:
        raise ValueError("PICO headset pose is missing")
    return invert_transform(_pose7_transform(sample.head_pose_xyz_xyzw)) @ (
        _pose7_transform(sample.pose_xyz_xyzw)
    )


def control_sample_head_controller_transform(sample) -> np.ndarray:
    """Undo the controller's calibrated-world-axis head translation encoding."""
    if sample.head_pose_xyz_xyzw is None:
        raise ValueError("PICO headset pose is missing")
    head_rotation_xr = quaternion_xyzw_to_rotation(
        sample.head_pose_xyz_xyzw[3:7]
    )
    result = np.eye(4)
    result[:3, :3] = quaternion_xyzw_to_rotation(sample.pose_xyz_xyzw[3:7])
    result[:3, 3] = head_rotation_xr.T @ sample.position
    return result


@dataclass(frozen=True)
class HeadCameraCalibration:
    transform_work_camera: np.ndarray
    camera_serial: str
    work_frame: str
    tool_frame: str
    rms_error_m: float
    max_error_m: float
    point_count: int
    lift_height_mm: float | None = None
    # Unity controller -> ray origin, expressed in Unity coordinates. Identity
    # applies when the transmitted controller pose is already the aim pose.
    transform_controller_ray: np.ndarray = field(default_factory=lambda: np.eye(4))
    # TCP axes relative to the ray axes AFTER conversion to right-handed axes.
    rotation_ray_tcp: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        object.__setattr__(
            self, "transform_work_camera",
            _transform(self.transform_work_camera, "transform_work_camera"),
        )
        object.__setattr__(self, "transform_controller_ray", _transform(
            self.transform_controller_ray, "transform_controller_ray"
        ))
        object.__setattr__(self, "rotation_ray_tcp", validate_rotation(self.rotation_ray_tcp))
        if self.camera_serial != "CP1Z842000EP":
            raise ValueError("calibration is not for the verified head camera")
        if self.work_frame != "416table" or self.tool_frame != "leboripper":
            raise ValueError("calibration work/tool frames do not match the right arm")
        if self.point_count < 6:
            raise ValueError("camera calibration requires at least six 3D pairs")
        if not np.isfinite(self.rms_error_m) or not 0 <= self.rms_error_m <= 0.01:
            raise ValueError("camera calibration RMS residual exceeds 10 mm")
        if not np.isfinite(self.max_error_m) or not 0 <= self.max_error_m <= 0.02:
            raise ValueError("camera calibration maximum residual exceeds 20 mm")

    @classmethod
    def load(cls, path: str | Path) -> "HeadCameraCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 2 or data.get("kinematics_frame") != "work_tcp":
            raise ValueError("head camera calibration requires schema 2 work_tcp points; recapture legacy points after the SDK frame fix")
        held_out_error = float(data.get("validation_max_error_m", float("nan")))
        if not np.isfinite(held_out_error) or held_out_error > 0.02 or held_out_error < 0:
            raise ValueError("head camera calibration requires held-out error <=20 mm")
        return cls(
            transform_work_camera=data["transform_work_camera"],
            camera_serial=data["camera_serial"],
            work_frame=data["work_frame"],
            tool_frame=data["tool_frame"],
            rms_error_m=float(data["rms_error_m"]),
            max_error_m=float(data["max_error_m"]),
            point_count=int(data["point_count"]),
            lift_height_mm=(
                None if data.get("lift_height_mm") is None
                else float(data["lift_height_mm"])
            ),
            transform_controller_ray=data["transform_controller_ray"],
            rotation_ray_tcp=data["rotation_ray_tcp"],
        )

    def as_dict(self) -> dict:
        return {
            "schema_version": 2,
            "kinematics_frame": "work_tcp",
            "transform_work_camera": self.transform_work_camera.tolist(),
            "camera_serial": self.camera_serial,
            "work_frame": self.work_frame,
            "tool_frame": self.tool_frame,
            "rms_error_m": self.rms_error_m,
            "max_error_m": self.max_error_m,
            "point_count": self.point_count,
            "lift_height_mm": self.lift_height_mm,
            "transform_controller_ray": self.transform_controller_ray.tolist(),
            "rotation_ray_tcp": self.rotation_ray_tcp.tolist(),
        }


def fit_head_camera_calibration(camera_points_m, work_points_m, *, lift_height_mm=None):
    """Fit T_work_camera from independent, noncoplanar 3D TCP observations."""
    camera = np.asarray(camera_points_m, dtype=float)
    work = np.asarray(work_points_m, dtype=float)
    if camera.ndim != 2 or camera.shape[1] != 3 or camera.shape != work.shape:
        raise ValueError("camera/work points must be matching Nx3 arrays")
    if len(camera) < 6 or not np.all(np.isfinite(camera)) or not np.all(np.isfinite(work)):
        raise ValueError("at least six finite 3D correspondences are required")
    camera_centered = camera - camera.mean(axis=0)
    work_centered = work - work.mean(axis=0)
    spread = np.linalg.svd(camera_centered, compute_uv=False)
    if spread[0] < 0.15 or spread[2] < 0.02:
        raise ValueError("calibration points must span at least 15 cm and vary in 3D")
    u, _, vt = np.linalg.svd(camera_centered.T @ work_centered)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = work.mean(axis=0) - rotation @ camera.mean(axis=0)
    estimated = (rotation @ camera.T).T + translation
    errors = np.linalg.norm(estimated - work, axis=1)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return HeadCameraCalibration(
        transform_work_camera=transform,
        camera_serial="CP1Z842000EP",
        work_frame="416table",
        tool_frame="leboripper",
        rms_error_m=float(np.sqrt(np.mean(errors**2))),
        max_error_m=float(np.max(errors)),
        point_count=len(camera),
        lift_height_mm=lift_height_mm,
    )


class CameraAbsolutePoseRetarget:
    """Direct head-local -> camera-local absolute mapping, independent of Grip.

    Unity: X right, Y up, Z forward (left handed). Optical camera: X right,
    Y down, Z forward (right handed). A basis change is a reflection, so
    rotation MUST be conjugated on both sides rather than left-multiplied.
    Headset and camera origins correspond; metres are mapped one-to-one.
    """

    def __init__(self, calibration: HeadCameraCalibration):
        self.calibration = calibration

    def target(self, sample):
        head_ray = (control_sample_head_controller_transform(sample)
                    @ self.calibration.transform_controller_ray)
        basis = np.diag([1.0, -1.0, 1.0])
        camera_tcp = np.eye(4)
        camera_tcp[:3, 3] = basis @ head_ray[:3, 3]
        camera_tcp[:3, :3] = (
            basis @ head_ray[:3, :3] @ basis @ self.calibration.rotation_ray_tcp
        )
        return self.calibration.transform_work_camera @ camera_tcp, camera_tcp


class AbsolutePoseRetarget:
    """Map a fixed headset/controller registration to absolute TCP targets."""

    def __init__(self, calibration: HeadCameraCalibration | None, rotation_work_xr):
        self.calibration = calibration
        self.rotation_work_xr = validate_rotation(rotation_work_xr)
        self.transform_work_virtual_head = None
        self.transform_controller_tcp = None

    def capture_origin(self, sample, transform_work_tcp) -> None:
        """One initial correspondence; no incremental reanchoring on Grip release."""
        tcp = _transform(transform_work_tcp, "transform_work_tcp")
        head_controller = control_sample_head_controller_transform(sample)
        head_rotation_xr = _pose7_transform(sample.head_pose_xyz_xyzw)[:3, :3]
        rotation_work_head = self.rotation_work_xr @ head_rotation_xr
        virtual_head = np.eye(4)
        virtual_head[:3, :3] = rotation_work_head
        virtual_head[:3, 3] = (
            tcp[:3, 3] - rotation_work_head @ head_controller[:3, 3]
        )
        controller_tcp = np.eye(4)
        controller_tcp[:3, :3] = (
            virtual_head[:3, :3] @ head_controller[:3, :3]
        ).T @ tcp[:3, :3]
        self.transform_work_virtual_head = virtual_head
        self.transform_controller_tcp = controller_tcp

    def target(self, sample) -> tuple[np.ndarray, np.ndarray | None]:
        if self.transform_work_virtual_head is None:
            raise ValueError("absolute pose origin has not been captured")
        target_work = (
            self.transform_work_virtual_head
            @ control_sample_head_controller_transform(sample)
            @ self.transform_controller_tcp
        )
        target_camera = None
        if self.calibration is not None:
            target_camera = (
                invert_transform(self.calibration.transform_work_camera) @ target_work
            )
        return target_work, target_camera


def solve_wrist_orientation(
    q_current,
    target_rotation_work_tcp,
    hold_position_work_m,
    fk_work_tcp,
    joint_soft_lower_rad,
    joint_soft_upper_rad,
    *,
    max_position_error_m: float = 0.001,
    max_orientation_error_deg: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Solve absolute J4-J6 angles while holding J1-J3 exactly fixed."""
    current = np.asarray(q_current, dtype=float)
    lower = np.asarray(joint_soft_lower_rad, dtype=float)
    upper = np.asarray(joint_soft_upper_rad, dtype=float)
    target_rotation = validate_rotation(target_rotation_work_tcp)
    hold_position = np.asarray(hold_position_work_m, dtype=float)
    if current.shape != (6,) or lower.shape != (6,) or upper.shape != (6,):
        raise ValueError("wrist IK requires six joint values and limits")
    if not all(np.all(np.isfinite(v)) for v in (current, lower, upper)) or np.any(lower >= upper):
        raise ValueError("wrist joints and limits must be finite with lower < upper")
    if not (np.isfinite(max_position_error_m) and max_position_error_m > 0
            and np.isfinite(max_orientation_error_deg) and max_orientation_error_deg > 0):
        raise ValueError("wrist IK tolerances must be finite and positive")
    if hold_position.shape != (3,) or not np.all(np.isfinite(hold_position)):
        raise ValueError("hold position must be finite XYZ")
    if np.any(current[:3] < lower[:3]) or np.any(current[:3] > upper[:3]):
        raise ValueError("fixed J1-J3 are outside joint soft limits")
    if np.any(current[3:] < lower[3:]) or np.any(current[3:] > upper[3:]):
        raise ValueError("current wrist joints are outside soft limits")

    def pose(wrist):
        q = current.copy()
        q[3:] = wrist
        return _transform(fk_work_tcp(q), "wrist FK")

    def residual(wrist):
        achieved = pose(wrist)
        rotation_error = rotation_log(achieved[:3, :3].T @ target_rotation)
        position_error = achieved[:3, 3] - hold_position
        return np.r_[rotation_error / np.deg2rad(max_orientation_error_deg),
                     position_error / max_position_error_m,
                     0.001 * (wrist - current[3:])]

    result = least_squares(
        residual, current[3:].copy(), bounds=(lower[3:], upper[3:]),
        # Native SDK FK uses float32. The default ~1e-8 finite difference is
        # smaller than its resolution, producing a zero/unstable Jacobian.
        jac=lambda wrist: np.column_stack([
            (residual(wrist + np.eye(3)[axis] * 1e-4)
             - residual(wrist - np.eye(3)[axis] * 1e-4)) / 2e-4
            for axis in range(3)
        ]),
        max_nfev=40, ftol=1e-6, xtol=1e-6, gtol=1e-6,
    )
    q_result = current.copy()
    q_result[3:] = result.x
    achieved = pose(result.x)
    position_error = float(np.linalg.norm(achieved[:3, 3] - hold_position))
    orientation_error = float(np.degrees(rotation_angle_rad(
        achieved[:3, :3], target_rotation
    )))
    if position_error > max_position_error_m or orientation_error > max_orientation_error_deg:
        raise ValueError(
            "wrist-only pose is unreachable with fixed J1-J3: "
            f"TCP drift {position_error * 1000:.1f} mm, "
            f"orientation error {orientation_error:.1f} deg"
        )
    return q_result, position_error, orientation_error
