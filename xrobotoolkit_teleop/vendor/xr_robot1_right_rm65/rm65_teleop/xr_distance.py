"""Pure helpers for ruler-based XR translation scale calibration."""

from __future__ import annotations

import numpy as np

from .transforms import finite_vector


def stable_position_center(positions, *, max_span_m: float = 0.005) -> np.ndarray:
    samples = np.asarray(positions, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 3 or samples.shape[0] < 5:
        raise ValueError("stable XR capture requires at least five 3D samples")
    if not np.all(np.isfinite(samples)):
        raise ValueError("stable XR capture contains non-finite values")
    span = np.ptp(samples, axis=0)
    if float(np.max(span)) > float(max_span_m):
        raise ValueError(
            "XR controller was not stationary during capture: "
            f"axis_span_mm={(span * 1000.0).round(3).tolist()}"
        )
    return np.median(samples, axis=0)


def estimate_position_scale(start_position, end_position, known_distance_m: float):
    start = finite_vector(start_position, 3, "XR ruler start")
    end = finite_vector(end_position, 3, "XR ruler end")
    known = float(known_distance_m)
    if not np.isfinite(known) or known <= 0.0:
        raise ValueError("known ruler distance must be positive")
    raw_delta = end - start
    raw_distance = float(np.linalg.norm(raw_delta))
    if raw_distance < 0.01:
        raise ValueError(
            f"XR ruler displacement is too small: {raw_distance * 1000.0:.3f} mm"
        )
    return raw_delta, raw_distance, known / raw_distance
