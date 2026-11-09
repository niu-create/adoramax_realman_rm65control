"""Bounds for one real-XR commissioning Grip session."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .commissioning import AXIS_VECTORS
from .transforms import finite_vector


@dataclass(frozen=True)
class LiveAxisGuard:
    axis: str
    target_distance_m: float = 0.03
    max_main_distance_m: float = 0.035
    max_cross_distance_m: float = 0.01
    max_reverse_distance_m: float = 0.005

    def __post_init__(self):
        if self.axis not in ("X+", "Y+", "Z+"):
            raise ValueError("live commissioning supports X+, Y+, or Z+ only")
        values = (
            self.target_distance_m,
            self.max_main_distance_m,
            self.max_cross_distance_m,
            self.max_reverse_distance_m,
        )
        if not all(np.isfinite(value) and float(value) > 0.0 for value in values):
            raise ValueError("live commissioning limits must be positive")
        if self.max_main_distance_m < self.target_distance_m:
            raise ValueError("max main distance must cover the target distance")

    @property
    def main_index(self) -> int:
        return "XYZ".index(self.axis[0])

    @property
    def target_delta_m(self) -> np.ndarray:
        return AXIS_VECTORS[self.axis] * self.target_distance_m

    def validate(self, operator_delta_m) -> np.ndarray:
        delta = finite_vector(operator_delta_m, 3, "live operator delta")
        main = float(delta[self.main_index])
        if main > self.max_main_distance_m:
            raise RuntimeError(
                "live XR main-axis request exceeded bound: "
                f"actual={main * 1000.0:.3f} mm, "
                f"limit={self.max_main_distance_m * 1000.0:.3f} mm"
            )
        if main < -self.max_reverse_distance_m:
            raise RuntimeError(
                "live XR moved too far in the reverse direction: " f"actual={main * 1000.0:.3f} mm"
            )
        cross_indices = [index for index in range(3) if index != self.main_index]
        if np.any(np.abs(delta[cross_indices]) > self.max_cross_distance_m):
            raise RuntimeError(
                "live XR cross-axis request exceeded bound: "
                f"delta_mm={(delta * 1000.0).round(3).tolist()}, "
                f"limit={self.max_cross_distance_m * 1000.0:.3f} mm"
            )
        return delta
