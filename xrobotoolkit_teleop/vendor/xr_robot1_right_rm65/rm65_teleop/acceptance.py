"""Evaluate one Grip session against the requirement document."""

from __future__ import annotations

import numpy as np

from .models import SessionSummary
from .transforms import rotation_angle_rad


def dominant_axis(delta, *, idle_tolerance_m: float = 2.5e-4) -> str:
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        return "INVALID"
    index = int(np.argmax(np.abs(delta)))
    if abs(float(delta[index])) <= idle_tolerance_m:
        return "IDLE"
    return "XYZ"[index] + ("+" if delta[index] > 0.0 else "-")


def evaluate_session(
    *,
    session_id: int,
    stop_reason: str,
    requested_delta_work_m,
    actual_delta_work_m,
    start_rotation_work_tcp,
    actual_rotation_work_tcp,
) -> SessionSummary:
    requested = np.asarray(requested_delta_work_m, dtype=float)
    actual = np.asarray(actual_delta_work_m, dtype=float)
    request_axis = dominant_axis(requested)
    actual_axis = dominant_axis(actual)
    orientation_change_deg = float(
        np.degrees(
            rotation_angle_rad(
                start_rotation_work_tcp,
                actual_rotation_work_tcp,
            )
        )
    )
    if request_axis in ("IDLE", "INVALID"):
        return SessionSummary(
            session_id,
            stop_reason,
            requested,
            actual,
            request_axis,
            actual_axis,
            None,
            None,
            orientation_change_deg,
            "INCONCLUSIVE",
            {},
        )

    main_index = "XYZ".index(request_axis[0])
    requested_main = float(requested[main_index])
    actual_main = float(actual[main_index])
    cross_indices = [index for index in range(3) if index != main_index]
    cross_limit = max(0.15 * abs(requested_main), 0.003)
    distance_error_ratio = abs(actual_main - requested_main) / abs(requested_main)
    checks = {
        "same_axis_and_sign": actual_axis == request_axis,
        "distance_error_le_15_percent": distance_error_ratio <= 0.15,
        "cross_axis_within_limit": bool(np.all(np.abs(actual[cross_indices]) <= cross_limit)),
        "orientation_change_le_2_deg": orientation_change_deg <= 2.0,
        "requested_motion_is_single_axis": bool(
            np.all(np.abs(requested[cross_indices]) <= cross_limit)
        ),
    }
    result = "PASS" if all(checks.values()) else "FAIL"
    return SessionSummary(
        session_id,
        stop_reason,
        requested,
        actual,
        request_axis,
        actual_axis,
        distance_error_ratio,
        cross_limit,
        orientation_change_deg,
        result,
        checks,
    )
