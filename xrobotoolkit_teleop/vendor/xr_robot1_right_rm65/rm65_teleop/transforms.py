"""Small, explicit SE(3) helpers for the RM65 teleoperation chain.

RealMan Euler poses use metres and fixed-axis XYZ angles composed as
``Rz @ Ry @ Rx``.  Every transform in this package follows the convention
``T_parent_child``: it maps child-frame coordinates into the parent frame.
"""

from __future__ import annotations

import math

import numpy as np


def finite_vector(value, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {size} finite values")
    return vector.copy()


def validate_rotation(rotation, *, atol: float = 1.0e-6) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=atol):
        raise ValueError(
            "rotation must be proper/right-handed " f"(determinant={determinant:.9f})"
        )
    return rotation.copy()


def euler_xyz_to_rotation(euler_rad) -> np.ndarray:
    """Convert RealMan fixed-axis XYZ Euler angles to a rotation matrix."""
    rx, ry, rz = finite_vector(euler_rad, 3, "Euler angles")
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rotation_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rotation_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return validate_rotation(rotation_z @ rotation_y @ rotation_x)


def quaternion_xyzw_to_rotation(quaternion_xyzw) -> np.ndarray:
    """Convert an XR quaternion in ``[x, y, z, w]`` order to a rotation."""
    x, y, z, w = finite_vector(quaternion_xyzw, 4, "XR quaternion")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-12:
        raise ValueError("XR quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return validate_rotation(
        np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=float,
        )
    )


def rotation_to_quaternion_xyzw(rotation) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized ``[x, y, z, w]`` quaternion."""
    rotation = validate_rotation(rotation)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = 2.0 * math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.array(
                [0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale,
                 (rotation[0, 2] + rotation[2, 0]) / scale,
                 (rotation[2, 1] - rotation[1, 2]) / scale]
            )
        elif axis == 1:
            scale = 2.0 * math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.array(
                [(rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                 (rotation[1, 2] + rotation[2, 1]) / scale,
                 (rotation[0, 2] - rotation[2, 0]) / scale]
            )
        else:
            scale = 2.0 * math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.array(
                [(rotation[0, 2] + rotation[2, 0]) / scale,
                 (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale,
                 (rotation[1, 0] - rotation[0, 1]) / scale]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def rotation_to_euler_xyz(rotation) -> np.ndarray:
    """Convert a rotation matrix to RealMan fixed-axis XYZ Euler angles."""
    rotation = validate_rotation(rotation)
    sy = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) > 1.0e-8:
        rx = math.atan2(rotation[2, 1], rotation[2, 2])
        rz = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        # At gimbal lock, choose rx=0 and retain the represented rotation in rz.
        rx = 0.0
        rz = math.atan2(-rotation[0, 1], rotation[1, 1])
    return np.array([rx, ry, rz], dtype=float)


def pose6_to_transform(pose_m_rad) -> np.ndarray:
    pose = finite_vector(pose_m_rad, 6, "pose")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = euler_xyz_to_rotation(pose[3:6])
    transform[:3, 3] = pose[:3]
    return transform


def transform_to_pose6(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise ValueError("transform has an invalid homogeneous bottom row")
    rotation = validate_rotation(transform[:3, :3])
    return np.concatenate((transform[:3, 3], rotation_to_euler_xyz(rotation)))


def invert_transform(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("transform must be a finite 4x4 matrix")
    rotation = validate_rotation(transform[:3, :3])
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def rotation_angle_rad(first, second) -> float:
    first = validate_rotation(first)
    second = validate_rotation(second)
    cosine = float(np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def rotation_log(rotation) -> np.ndarray:
    """Return an angle-axis vector whose exponential is ``rotation``."""
    rotation = validate_rotation(rotation)
    angle = rotation_angle_rad(np.eye(3), rotation)
    if angle < 1.0e-10:
        return np.zeros(3, dtype=float)
    if math.pi - angle < 1.0e-6:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
        axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    ) / (2.0 * math.sin(angle))
    return axis * angle


def rotation_exp(angle_axis) -> np.ndarray:
    vector = finite_vector(angle_axis, 3, "angle-axis vector")
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.eye(3, dtype=float)
    axis = vector / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return validate_rotation(
        np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def interpolate_transform(current, target, alpha: float) -> np.ndarray:
    """Interpolate translation and the shortest relative rotation."""
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.shape != (4, 4) or target.shape != (4, 4):
        raise ValueError("current and target must be 4x4 transforms")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    current_rotation = validate_rotation(current[:3, :3])
    target_rotation = validate_rotation(target[:3, :3])
    relative = current_rotation.T @ target_rotation
    result = np.eye(4, dtype=float)
    result[:3, :3] = current_rotation @ rotation_exp(alpha * rotation_log(relative))
    result[:3, 3] = current[:3, 3] + alpha * (target[:3, 3] - current[:3, 3])
    return result


def work_tcp_from_base_flange(
    transform_base_flange,
    transform_base_work,
    transform_flange_tcp,
) -> np.ndarray:
    """Apply ``inv(T_base_work) @ T_base_flange @ T_flange_tcp``."""
    return (
        invert_transform(transform_base_work)
        @ np.asarray(transform_base_flange, dtype=float)
        @ np.asarray(transform_flange_tcp, dtype=float)
    )


def base_flange_from_work_tcp(
    transform_work_tcp,
    transform_base_work,
    transform_flange_tcp,
) -> np.ndarray:
    """Apply ``T_base_work @ T_work_tcp @ inv(T_flange_tcp)``."""
    return (
        np.asarray(transform_base_work, dtype=float)
        @ np.asarray(transform_work_tcp, dtype=float)
        @ invert_transform(transform_flange_tcp)
    )


def nearest_equivalent_angles(angles, reference) -> np.ndarray:
    """Choose the 2*pi-equivalent representation nearest ``reference``."""
    angles = finite_vector(angles, 6, "joint solution")
    reference = finite_vector(reference, 6, "joint reference")
    return reference + (angles - reference + math.pi) % (2.0 * math.pi) - math.pi
