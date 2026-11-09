"""Compact stdout events for the local official-XR trajectory visualizer."""

from __future__ import annotations

import json
import math
import time
from typing import Iterable


TRAJECTORY_PREFIX = "@@XR_TRAJECTORY@@"


def make_trajectory_event(
    position_m: Iterable[float],
    *,
    grip: float | None = None,
    trigger: float | None = None,
    timestamp_ns: int | None = None,
    received_monotonic_ns: int | None = None,
    sequence: int | None = None,
    phase: str = "stream",
) -> dict:
    position = [float(value) for value in position_m]
    if len(position) != 3 or not all(math.isfinite(value) for value in position):
        raise ValueError("trajectory position must contain three finite values")
    event = {
        "schema_version": 1,
        "position_m": position,
        "grip": None if grip is None else float(grip),
        "trigger": None if trigger is None else float(trigger),
        "timestamp_ns": None if timestamp_ns is None else int(timestamp_ns),
        "received_monotonic_ns": (
            time.monotonic_ns()
            if received_monotonic_ns is None
            else int(received_monotonic_ns)
        ),
        "sequence": None if sequence is None else int(sequence),
        "phase": str(phase),
    }
    for name in ("grip", "trigger"):
        value = event[name]
        if value is not None and not math.isfinite(value):
            raise ValueError(f"trajectory {name} must be finite")
    return event


def emit_trajectory_point(position_m: Iterable[float], **metadata) -> None:
    """Publish one machine-readable line; the debug UI removes it from logs."""
    event = make_trajectory_event(position_m, **metadata)
    print(
        TRAJECTORY_PREFIX + json.dumps(event, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )
