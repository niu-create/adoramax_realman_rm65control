"""Statistics and calibration helpers for read-only XR packet capture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .transforms import validate_rotation
from .xr_atomic import AtomicXrPacket


def axis_capture_distance_bounds_mm(expected_distance_mm: float) -> tuple[float, float]:
    """Return the direct 1:1 diagnostic band for a ruler-labelled move."""
    expected = float(expected_distance_mm)
    if not math.isfinite(expected) or expected <= 0.0:
        raise ValueError("expected axis distance must be positive and finite")
    return expected * 0.75, expected * 1.25


def axis_capture_minimum_raw_distance_mm(expected_distance_mm: float) -> float:
    """Reject only an accidental near-A capture before scale calibration.

    The ruler distance is the physical ground truth.  Requiring the uncalibrated
    SDK delta to already equal that value creates an impossible retry loop when
    the upstream XR stream has a scale error.
    """
    expected = float(expected_distance_mm)
    if not math.isfinite(expected) or expected <= 0.0:
        raise ValueError("expected axis distance must be positive and finite")
    return max(5.0, expected * 0.20)


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def summarize_packets(packets: Iterable[AtomicXrPacket]) -> dict:
    samples = list(packets)
    if not samples:
        raise ValueError("at least one XR packet is required")
    positions = np.asarray([sample.position for sample in samples], dtype=float)
    poses = np.asarray([sample.pose_xyz_xyzw for sample in samples], dtype=float)
    quaternion_norms = np.asarray([sample.quaternion_norm for sample in samples], dtype=float)
    source_timestamps = np.asarray([sample.timestamp_ns for sample in samples], dtype=np.int64)
    host_timestamps = np.asarray(
        [sample.received_monotonic_ns for sample in samples], dtype=np.int64
    )
    source_delta_ns = np.diff(source_timestamps)
    host_delta_ns = np.diff(host_timestamps)
    positive_source_ms = source_delta_ns[source_delta_ns > 0].astype(float) / 1.0e6
    positive_host_ms = host_delta_ns[host_delta_ns > 0].astype(float) / 1.0e6
    elapsed_s = (
        float(host_timestamps[-1] - host_timestamps[0]) / 1.0e9
        if len(samples) > 1
        else 0.0
    )
    source_advances = int(np.count_nonzero(source_delta_ns > 0))
    duplicate_timestamps = int(np.count_nonzero(source_delta_ns == 0))
    timestamp_rollbacks = int(np.count_nonzero(source_delta_ns < 0))
    pose_changed = np.any(np.diff(poses, axis=0) != 0.0, axis=1)
    pose_change_count = int(np.count_nonzero(pose_changed))
    unique_pose_count = len({sample.pose_xyz_xyzw for sample in samples})
    longest_identical_pose_run = 1
    current_identical_pose_run = 1
    for changed in pose_changed:
        if changed:
            current_identical_pose_run = 1
        else:
            current_identical_pose_run += 1
            longest_identical_pose_run = max(
                longest_identical_pose_run, current_identical_pose_run
            )
    grip = np.asarray([sample.grip for sample in samples], dtype=float)
    trigger = np.asarray([sample.trigger for sample in samples], dtype=float)
    maximum_host_gap_ms = _percentile(positive_host_ms, 100.0)
    report = {
        "packet_count": len(samples),
        "elapsed_s": elapsed_s,
        "arrival_rate_hz": None if elapsed_s <= 0.0 else (len(samples) - 1) / elapsed_s,
        "source_timestamp_advances": source_advances,
        "source_timestamp_duplicates": duplicate_timestamps,
        "source_timestamp_rollbacks": timestamp_rollbacks,
        "pose_change_count": pose_change_count,
        "unique_pose_count": unique_pose_count,
        "longest_identical_pose_run": longest_identical_pose_run,
        "source_period_ms": {
            "median": _percentile(positive_source_ms, 50.0),
            "p95": _percentile(positive_source_ms, 95.0),
            "max": _percentile(positive_source_ms, 100.0),
        },
        "host_arrival_period_ms": {
            "median": _percentile(positive_host_ms, 50.0),
            "p95": _percentile(positive_host_ms, 95.0),
            "max": maximum_host_gap_ms,
        },
        "position_span_mm": (np.ptp(positions, axis=0) * 1000.0).tolist(),
        "quaternion_norm": {
            "min": float(np.min(quaternion_norms)),
            "median": float(np.median(quaternion_norms)),
            "max": float(np.max(quaternion_norms)),
            "max_abs_error_from_one": float(np.max(np.abs(quaternion_norms - 1.0))),
        },
        "grip_range": [float(np.min(grip)), float(np.max(grip))],
        "trigger_range": [float(np.min(trigger)), float(np.max(trigger))],
        "app_focus_values": sorted(
            {sample.app_focused for sample in samples if sample.app_focused is not None}
        ),
    }
    report["checks"] = {
        "enough_packets": len(samples) >= 3,
        "timestamp_advancing": source_advances >= 2,
        "pose_not_frozen": unique_pose_count >= 2,
        "no_timestamp_rollback": timestamp_rollbacks == 0,
        "host_gap_below_200ms": maximum_host_gap_ms is not None
        and maximum_host_gap_ms <= 200.0,
        "unit_quaternion_within_1pct": float(np.max(np.abs(quaternion_norms - 1.0)))
        <= 0.01,
        "grip_in_unit_range": float(np.min(grip)) >= -0.01
        and float(np.max(grip)) <= 1.01,
        "trigger_in_unit_range": float(np.min(trigger)) >= -0.01
        and float(np.max(trigger)) <= 1.01,
        "app_focused_when_reported": not any(
            sample.app_focused is False for sample in samples
        ),
    }
    report["passed"] = bool(all(report["checks"].values()))
    return report


def _button_transitions(values: np.ndarray) -> dict:
    """Count complete release/press transitions using the live Grip thresholds."""
    stable_state: str | None = None
    cycle_stage = 0
    press_count = 0
    release_count = 0
    released_seen = False
    pressed_seen = False
    for value in values:
        state = "released" if value < 0.1 else "pressed" if value > 0.9 else None
        if state is None:
            continue
        released_seen |= state == "released"
        pressed_seen |= state == "pressed"
        if cycle_stage == 0 and state == "released":
            cycle_stage = 1
        elif cycle_stage == 1 and state == "pressed":
            cycle_stage = 2
        elif cycle_stage == 2 and state == "released":
            cycle_stage = 3
        if stable_state is not None and state != stable_state:
            if state == "pressed":
                press_count += 1
            else:
                release_count += 1
        stable_state = state
    return {
        "released_seen": bool(released_seen),
        "pressed_seen": bool(pressed_seen),
        "press_transition_count": press_count,
        "release_transition_count": release_count,
        "complete_release_press_release": bool(
            cycle_stage == 3
        ),
    }


def summarize_motion_challenge(
    packets: Iterable[AtomicXrPacket],
    *,
    minimum_travel_mm: float = 30.0,
    maximum_identical_pose_s: float = 0.5,
) -> dict:
    """Validate a deliberate move plus Grip/Trigger press-and-release exercise."""
    samples = list(packets)
    minimum_travel_mm = float(minimum_travel_mm)
    maximum_identical_pose_s = float(maximum_identical_pose_s)
    if not np.isfinite(minimum_travel_mm) or minimum_travel_mm <= 0.0:
        raise ValueError("minimum motion-challenge travel must be positive")
    if not np.isfinite(maximum_identical_pose_s) or maximum_identical_pose_s <= 0.0:
        raise ValueError("maximum identical-pose duration must be positive")

    report = summarize_packets(samples)
    positions = np.asarray([sample.position for sample in samples], dtype=float)
    span_mm = np.ptp(positions, axis=0) * 1000.0
    travel_extent_mm = float(np.linalg.norm(span_mm))

    longest_duration_s = 0.0
    run_start = 0
    for index in range(1, len(samples) + 1):
        run_ended = (
            index == len(samples)
            or samples[index].pose_xyz_xyzw != samples[index - 1].pose_xyz_xyzw
        )
        if not run_ended:
            continue
        if index - run_start >= 2:
            duration_s = (
                samples[index - 1].received_monotonic_ns
                - samples[run_start].received_monotonic_ns
            ) / 1.0e9
            longest_duration_s = max(longest_duration_s, float(duration_s))
        run_start = index

    grip = np.asarray([sample.grip for sample in samples], dtype=float)
    trigger = np.asarray([sample.trigger for sample in samples], dtype=float)
    grip_transitions = _button_transitions(grip)
    trigger_transitions = _button_transitions(trigger)
    challenge_checks = {
        "deliberate_travel_reached": travel_extent_mm >= minimum_travel_mm,
        "no_long_identical_pose_run": (
            longest_duration_s <= maximum_identical_pose_s
        ),
        "grip_release_press_release": grip_transitions[
            "complete_release_press_release"
        ],
        "trigger_release_press_release": trigger_transitions[
            "complete_release_press_release"
        ],
    }
    report.update(
        {
            "motion_challenge": True,
            "minimum_travel_mm": minimum_travel_mm,
            "travel_extent_mm": travel_extent_mm,
            "maximum_identical_pose_s": maximum_identical_pose_s,
            "longest_identical_pose_duration_s": longest_duration_s,
            "grip_transitions": grip_transitions,
            "trigger_transitions": trigger_transitions,
        }
    )
    report["checks"].update(challenge_checks)
    report["passed"] = bool(all(report["checks"].values()))
    return report


def summarize_stationary_jitter(
    packets: Iterable[AtomicXrPacket],
    *,
    rotation_work_xr: Iterable[Iterable[float]] | None = None,
    maximum_p95_jitter_mm: float = 3.0,
    maximum_drift_mm: float = 5.0,
    drift_window_s: float = 1.0,
) -> dict:
    """Measure a stationary controller without hiding the official raw samples."""
    samples = list(packets)
    if not samples:
        raise ValueError("stationary jitter summary requires at least one packet")
    maximum_p95_jitter_mm = float(maximum_p95_jitter_mm)
    maximum_drift_mm = float(maximum_drift_mm)
    drift_window_s = float(drift_window_s)
    if not np.isfinite(maximum_p95_jitter_mm) or maximum_p95_jitter_mm <= 0.0:
        raise ValueError("maximum stationary p95 jitter must be positive")
    if not np.isfinite(maximum_drift_mm) or maximum_drift_mm <= 0.0:
        raise ValueError("maximum stationary drift must be positive")
    if not np.isfinite(drift_window_s) or drift_window_s <= 0.0:
        raise ValueError("stationary drift window must be positive")

    rotation = (
        np.eye(3, dtype=float)
        if rotation_work_xr is None
        else validate_rotation(rotation_work_xr)
    )
    report = summarize_packets(samples)
    poses = np.asarray([sample.pose_xyz_xyzw for sample in samples], dtype=float)
    quaternion_norm = np.linalg.norm(poses[:, 3:7], axis=1)
    valid_mask = (
        np.all(np.isfinite(poses), axis=1)
        & (np.linalg.norm(poses[:, :3], axis=1) > 1.0e-4)
        & (quaternion_norm >= 0.5)
        & (quaternion_norm <= 1.5)
    )
    valid_count = int(np.count_nonzero(valid_mask))
    invalid_count = len(samples) - valid_count

    jitter = {
        "center_work_m": None,
        "radial_rms_mm": None,
        "radial_p95_mm": None,
        "radial_max_mm": None,
        "axis_p95_abs_deviation_mm": None,
        "axis_peak_to_peak_mm": None,
        "sample_step_p95_mm": None,
        "sample_step_max_mm": None,
        "start_to_end_drift_mm": None,
        "drift_window_s": drift_window_s,
    }
    if valid_count:
        positions_xr = poses[valid_mask, :3]
        positions_work = positions_xr @ rotation.T
        center = np.median(positions_work, axis=0)
        deviation_mm = (positions_work - center) * 1000.0
        radial_mm = np.linalg.norm(deviation_mm, axis=1)
        valid_indices = np.flatnonzero(valid_mask)
        contiguous = np.diff(valid_indices) == 1
        steps_mm = np.linalg.norm(np.diff(positions_work, axis=0), axis=1) * 1000.0
        steps_mm = steps_mm[contiguous]

        source_ns = np.asarray(
            [sample.timestamp_ns for sample in samples], dtype=np.int64
        )[valid_mask]
        start_limit = source_ns[0] + int(drift_window_s * 1.0e9)
        end_limit = source_ns[-1] - int(drift_window_s * 1.0e9)
        start_points = positions_work[source_ns <= start_limit]
        end_points = positions_work[source_ns >= end_limit]
        if not len(start_points):
            start_points = positions_work[:1]
        if not len(end_points):
            end_points = positions_work[-1:]
        drift_mm = float(
            np.linalg.norm(np.median(end_points, axis=0) - np.median(start_points, axis=0))
            * 1000.0
        )
        jitter.update(
            {
                "center_work_m": center.tolist(),
                "radial_rms_mm": float(np.sqrt(np.mean(radial_mm**2))),
                "radial_p95_mm": float(np.percentile(radial_mm, 95.0)),
                "radial_max_mm": float(np.max(radial_mm)),
                "axis_p95_abs_deviation_mm": np.percentile(
                    np.abs(deviation_mm), 95.0, axis=0
                ).tolist(),
                "axis_peak_to_peak_mm": (
                    np.ptp(positions_work, axis=0) * 1000.0
                ).tolist(),
                "sample_step_p95_mm": (
                    None if not len(steps_mm) else float(np.percentile(steps_mm, 95.0))
                ),
                "sample_step_max_mm": (
                    None if not len(steps_mm) else float(np.max(steps_mm))
                ),
                "start_to_end_drift_mm": drift_mm,
            }
        )

    radial_p95 = jitter["radial_p95_mm"]
    drift = jitter["start_to_end_drift_mm"]
    report.update(
        {
            "stationary_test": True,
            "coordinate_frame": "416table",
            "valid_pose_count": valid_count,
            "invalid_pose_count": invalid_count,
            "maximum_p95_jitter_mm": maximum_p95_jitter_mm,
            "maximum_drift_mm": maximum_drift_mm,
            "jitter": jitter,
        }
    )
    report["checks"].update(
        {
            "no_invalid_poses": invalid_count == 0,
            "stationary_p95_within_limit": (
                radial_p95 is not None and radial_p95 <= maximum_p95_jitter_mm
            ),
            "stationary_drift_within_limit": (
                drift is not None and drift <= maximum_drift_mm
            ),
        }
    )
    report["passed"] = bool(all(report["checks"].values()))
    return report


@dataclass(frozen=True)
class StableCapture:
    center_m: np.ndarray
    sample_count: int
    duration_s: float
    radial_p95_mm: float
    radial_max_mm: float

    def as_dict(self) -> dict:
        return {
            "center_m": self.center_m.tolist(),
            "sample_count": self.sample_count,
            "duration_s": self.duration_s,
            "radial_p95_mm": self.radial_p95_mm,
            "radial_max_mm": self.radial_max_mm,
        }


def stable_capture(
    packets: Iterable[AtomicXrPacket],
    *,
    maximum_p95_jitter_mm: float = 2.0,
) -> StableCapture:
    samples = list(packets)
    if len(samples) < 5:
        raise ValueError("stable capture requires at least five packets")
    positions = np.asarray([sample.position for sample in samples], dtype=float)
    if len({sample.pose_xyz_xyzw for sample in samples}) < 2:
        raise ValueError(
            "controller pose is frozen while packet timestamps continue advancing"
        )
    center = np.median(positions, axis=0)
    radial_mm = np.linalg.norm(positions - center, axis=1) * 1000.0
    p95_mm = float(np.percentile(radial_mm, 95.0))
    maximum_mm = float(np.max(radial_mm))
    if p95_mm > float(maximum_p95_jitter_mm):
        raise ValueError(
            f"controller was not stationary: p95 jitter {p95_mm:.3f} mm "
            f"> {maximum_p95_jitter_mm:.3f} mm"
        )
    duration_s = (
        samples[-1].received_monotonic_ns - samples[0].received_monotonic_ns
    ) / 1.0e9
    return StableCapture(center, len(samples), duration_s, p95_mm, maximum_mm)


def solve_operator_from_xr(axis_deltas: dict[str, Iterable[float]]) -> dict:
    """Fit a proper rotation mapping labelled raw XR directions to +X/+Y/+Z."""
    required = ("X+", "Y+", "Z+")
    if set(axis_deltas) != set(required):
        raise ValueError("axis_deltas must contain exactly X+, Y+, and Z+")
    source_columns = []
    lengths_m = {}
    for axis in required:
        vector = np.asarray(axis_deltas[axis], dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{axis} delta must have three finite values")
        length = float(np.linalg.norm(vector))
        if length < 0.02:
            raise ValueError(f"{axis} capture is too short ({length * 1000.0:.3f} mm)")
        source_columns.append(vector / length)
        lengths_m[axis] = length
    source = np.column_stack(source_columns)
    target = np.eye(3)
    covariance = target @ source.T
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(left @ right_t))
    rotation = validate_rotation(left @ correction @ right_t)
    mapped = rotation @ source
    angular_errors_deg = {}
    for index, axis in enumerate(required):
        cosine = float(np.clip(mapped[:, index] @ target[:, index], -1.0, 1.0))
        angular_errors_deg[axis] = math.degrees(math.acos(cosine))
    rms = math.sqrt(
        sum(error * error for error in angular_errors_deg.values()) / len(required)
    )
    pairwise_dot = source.T @ source
    return {
        "rotation_operator_from_xr": rotation.tolist(),
        "determinant": float(np.linalg.det(rotation)),
        "axis_lengths_mm": {axis: length * 1000.0 for axis, length in lengths_m.items()},
        "measured_unit_axes_as_columns": source.tolist(),
        "measured_axis_pairwise_dot": pairwise_dot.tolist(),
        "angular_error_deg": angular_errors_deg,
        "rms_angular_error_deg": rms,
    }
