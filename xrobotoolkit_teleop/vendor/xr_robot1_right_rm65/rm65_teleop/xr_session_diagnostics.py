"""Read-only Grip-session diagnostics using production ``XrSample`` objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TeleopConfig
from .models import XrSample
from .transforms import pose6_to_transform


GRIP_ACTIVATE = 0.9
GRIP_RELEASE = 0.1


def xr_delta_to_operator(config: TeleopConfig, delta_xr) -> np.ndarray:
    delta_xr = np.asarray(delta_xr, dtype=float)
    if delta_xr.shape != (3,) or not np.all(np.isfinite(delta_xr)):
        raise ValueError("XR delta must contain three finite values")
    rotation_base_work = pose6_to_transform(config.work_frame.pose_m_rad)[:3, :3]
    mapped_delta_xr = config.translation_xr_axis_signs * delta_xr
    delta_base = config.rotation_base_xr @ mapped_delta_xr * config.position_scale
    return rotation_base_work.T @ delta_base


def valid_xr_pose(sample: XrSample) -> bool:
    pose = sample.pose_xyz_xyzw
    quaternion_norm = float(np.linalg.norm(pose[3:7]))
    return bool(
        np.all(np.isfinite(pose))
        and not np.allclose(pose[:3], 0.0, atol=1.0e-4)
        and 0.5 <= quaternion_norm <= 1.5
    )


@dataclass(frozen=True)
class MirrorSessionResult:
    session_id: int
    stop_reason: str
    peak_delta_operator_m: np.ndarray
    final_delta_operator_m: np.ndarray
    cumulative_path_m: float
    displacement_history_m: tuple[float, ...]

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "stop_reason": self.stop_reason,
            "peak_delta_operator_m": self.peak_delta_operator_m.tolist(),
            "peak_distance_mm": float(np.linalg.norm(self.peak_delta_operator_m) * 1000.0),
            "cumulative_path_length_mm": float(self.cumulative_path_m * 1000.0),
            "final_delta_operator_m": self.final_delta_operator_m.tolist(),
            "return_error_mm": float(np.linalg.norm(self.final_delta_operator_m) * 1000.0),
        }

    def closest_return_after_travel_mm(self, minimum_travel_mm: float) -> float | None:
        """Closest approach to A after first reaching the required displacement."""
        threshold_m = float(minimum_travel_mm) / 1000.0
        reached_index = next(
            (
                index
                for index, distance_m in enumerate(self.displacement_history_m)
                if distance_m >= threshold_m
            ),
            None,
        )
        if reached_index is None or reached_index + 1 >= len(self.displacement_history_m):
            return None
        return float(min(self.displacement_history_m[reached_index + 1 :]) * 1000.0)


class XrSessionMirror:
    """Mirror the production release/press/session rules without an arm object."""

    def __init__(self, config: TeleopConfig):
        self.config = config
        self.release_seen = False
        self.active = False
        self.session_id = 0
        self.start_position: np.ndarray | None = None
        self.last_position: np.ndarray | None = None
        self.last_delta_operator_m = np.zeros(3, dtype=float)
        self.peak_delta_operator_m = np.zeros(3, dtype=float)
        self.cumulative_path_m = 0.0
        self.displacement_history_m: list[float] = []
        self.sample_count = 0
        self.stale_sample_count = 0
        self.invalid_sample_count = 0
        self.results: list[MirrorSessionResult] = []

    def _stop(self, reason: str, *, final_delta, require_release: bool) -> None:
        if self.active:
            self.results.append(
                MirrorSessionResult(
                    self.session_id,
                    reason,
                    self.peak_delta_operator_m.copy(),
                    np.asarray(final_delta, dtype=float).copy(),
                    float(self.cumulative_path_m),
                    tuple(self.displacement_history_m),
                )
            )
        self.active = False
        self.start_position = None
        self.last_position = None
        self.last_delta_operator_m = np.zeros(3, dtype=float)
        self.peak_delta_operator_m = np.zeros(3, dtype=float)
        self.cumulative_path_m = 0.0
        self.displacement_history_m = []
        self.release_seen = not require_release

    def step(self, sample: XrSample) -> dict:
        self.sample_count += 1
        record = {
            "timestamp_ns": sample.timestamp_ns,
            "position": sample.position,
            "grip": float(sample.grip),
            "trigger": float(sample.trigger),
            "stream_fresh": bool(sample.stream_fresh),
            "phase": "idle",
            "session_id": self.session_id if self.active else None,
            "delta_xr_m": None,
            "delta_operator_m": None,
        }
        if not sample.stream_fresh:
            self.stale_sample_count += 1
            self._stop(
                "xr_stale",
                final_delta=self.last_delta_operator_m,
                require_release=True,
            )
            record["phase"] = "xr_stale"
            return record
        if not valid_xr_pose(sample) or not (
            -0.01 <= float(sample.grip) <= 1.01
            and -0.01 <= float(sample.trigger) <= 1.01
        ):
            self.invalid_sample_count += 1
            self._stop(
                "invalid_xr_input",
                final_delta=self.last_delta_operator_m,
                require_release=True,
            )
            record["phase"] = "invalid_xr_input"
            return record

        if not self.active:
            if sample.grip < GRIP_RELEASE:
                self.release_seen = True
                record["phase"] = "released_ready"
            elif sample.grip > GRIP_ACTIVATE and self.release_seen:
                self.session_id += 1
                self.active = True
                self.release_seen = False
                self.start_position = sample.position
                self.last_position = sample.position
                self.last_delta_operator_m = np.zeros(3, dtype=float)
                self.peak_delta_operator_m = np.zeros(3, dtype=float)
                self.cumulative_path_m = 0.0
                self.displacement_history_m = []
                record.update(
                    {
                        "phase": "session_start",
                        "session_id": self.session_id,
                        "delta_xr_m": np.zeros(3, dtype=float),
                        "delta_operator_m": np.zeros(3, dtype=float),
                    }
                )
            else:
                record["phase"] = "waiting_for_release"
            return record

        assert self.start_position is not None
        assert self.last_position is not None
        delta_xr = sample.position - self.start_position
        delta_operator = xr_delta_to_operator(self.config, delta_xr)
        record.update(
            {
                "session_id": self.session_id,
                "delta_xr_m": delta_xr,
                "delta_operator_m": delta_operator,
            }
        )
        if sample.grip < GRIP_RELEASE:
            self._stop(
                "grip_released",
                # Match the production controller: the release sample stops
                # immediately and is not used as one final motion update.
                final_delta=self.last_delta_operator_m,
                require_release=False,
            )
            record["phase"] = "session_end"
            return record
        sample_step = sample.position - self.last_position
        if float(np.linalg.norm(sample_step)) > self.config.max_xr_sample_step_m:
            self._stop(
                "xr_jump",
                final_delta=delta_operator,
                require_release=True,
            )
            record["phase"] = "xr_jump"
            return record
        self.cumulative_path_m += float(np.linalg.norm(sample_step)) * self.config.position_scale
        self.last_position = sample.position
        self.last_delta_operator_m = delta_operator.copy()
        self.displacement_history_m.append(float(np.linalg.norm(delta_operator)))
        if np.linalg.norm(delta_operator) > np.linalg.norm(self.peak_delta_operator_m):
            self.peak_delta_operator_m = delta_operator.copy()
        record["phase"] = "session_active"
        return record

    def finish(self) -> None:
        if self.active:
            self._stop(
                "program_end",
                final_delta=self.last_delta_operator_m,
                require_release=True,
            )

    def report(self, *, minimum_travel_mm: float, maximum_return_error_mm: float) -> dict:
        minimum_travel_mm = float(minimum_travel_mm)
        maximum_return_error_mm = float(maximum_return_error_mm)
        results = []
        for result in self.results:
            item = result.as_dict()
            item["closest_return_after_travel_mm"] = (
                result.closest_return_after_travel_mm(minimum_travel_mm)
            )
            results.append(item)
        completed = [item for item in results if item["stop_reason"] == "grip_released"]
        travelled = [
            item for item in completed if item["peak_distance_mm"] >= minimum_travel_mm
        ]
        returned = [
            item
            for item in travelled
            if item["closest_return_after_travel_mm"] is not None
            and item["closest_return_after_travel_mm"] <= maximum_return_error_mm
        ]
        abnormal = [item for item in results if item["stop_reason"] != "grip_released"]
        checks = {
            "samples_received": self.sample_count > 0,
            "release_observed": self.release_seen or bool(results),
            "grip_session_completed": bool(completed),
            "minimum_travel_reached": bool(travelled),
            "returned_near_a_before_release": bool(returned),
            "no_stale_samples": self.stale_sample_count == 0,
            "no_invalid_samples": self.invalid_sample_count == 0,
            "no_abnormal_session_stop": not abnormal,
        }
        return {
            "read_only": True,
            "uses_production_xr_client": True,
            "sample_count": self.sample_count,
            "session_count": len(results),
            "stale_sample_count": self.stale_sample_count,
            "invalid_sample_count": self.invalid_sample_count,
            "minimum_travel_mm": minimum_travel_mm,
            "minimum_peak_displacement_mm": minimum_travel_mm,
            "maximum_return_error_mm": maximum_return_error_mm,
            "return_acceptance_metric": "closest_to_a_after_reaching_minimum_travel",
            "sessions": results,
            "checks": checks,
            "passed": bool(all(checks.values())),
        }
