from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np


@dataclass(frozen=True)
class CameraSample:
    role: str
    image: np.ndarray
    monotonic_time: float


@dataclass
class SourceBundle:
    arms: Dict[str, dict]
    grippers: Dict[str, dict]
    cameras: Dict[str, CameraSample]


@dataclass(frozen=True)
class CollectionFrame:
    monotonic_time: float
    observation_state: np.ndarray
    action: np.ndarray
    images: Mapping[str, np.ndarray]
    source_ages_ms: Mapping[str, float]
    source_timestamps: Mapping[str, float]
