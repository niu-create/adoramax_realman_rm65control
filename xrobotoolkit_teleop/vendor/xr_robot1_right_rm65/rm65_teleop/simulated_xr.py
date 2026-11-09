"""Deterministic XR samples for phase-two controller-chain validation."""

from __future__ import annotations

import time

import numpy as np

from .models import XrSample
from .transforms import finite_vector


def xr_delta_for_operator_delta(config, transform_base_work, operator_delta_m):
    """Invert the live controller chain for a desired robot target delta."""
    operator_delta = finite_vector(operator_delta_m, 3, "operator delta")
    transform_base_work = np.asarray(transform_base_work, dtype=float)
    if transform_base_work.shape != (4, 4):
        raise ValueError("transform_base_work must be 4x4")
    rotation_base_work = transform_base_work[:3, :3]
    return (
        config.translation_xr_axis_signs
        * (
            config.rotation_base_xr.T
            @ rotation_base_work
            @ operator_delta
        )
        / (config.position_scale * config.teleop_position_gain)
    )


def xr_delta_for_physical_operator_delta(config, transform_base_work, operator_delta_m):
    """Generate an XR delta for a physical hand displacement in operator axes."""
    operator_delta = finite_vector(operator_delta_m, 3, "physical operator delta")
    transform_base_work = np.asarray(transform_base_work, dtype=float)
    if transform_base_work.shape != (4, 4):
        raise ValueError("transform_base_work must be 4x4")
    rotation_base_work = transform_base_work[:3, :3]
    return (
        config.translation_xr_axis_signs
        * (
            config.rotation_base_xr.T
            @ rotation_base_work
            @ operator_delta
        )
        / config.position_scale
    )


def simulated_sample(position, grip, sequence: int, *, trigger=0.0):
    position = finite_vector(position, 3, "simulated XR position")
    sequence = int(sequence)
    if sequence <= 0:
        raise ValueError("simulated XR sequence must be positive")
    return XrSample(
        np.array([*position, 0.0, 0.0, 0.0, 1.0], dtype=float),
        float(grip),
        float(trigger),
        sequence * 50_000_000,
        time.monotonic(),
        stream_fresh=True,
    )
