"""Detect deliberate translation/wrist gestures without a second button."""

from collections import deque

import numpy as np

from .transforms import rotation_exp, rotation_log, rotation_to_quaternion_xyzw


class AutoWristSwitch:
    DWELL_S = 0.20
    TRANSLATION_M = 0.007
    ROTATION_RAD = np.deg2rad(2.0)
    DOMINANCE = 1.2
    RECENT_WINDOW_S = 0.25
    ENTRY_ROTATION_RAD = np.deg2rad(5.0)
    ENTRY_RECENT_ROTATION_RAD = np.deg2rad(4.0)
    ENTRY_MAX_RECENT_TRANSLATION_M = 0.008
    HOLD_MAX_RECENT_TRANSLATION_M = 0.003
    HOLD_RECENT_ROTATION_RAD = np.deg2rad(3.0)
    EXIT_RECENT_TRANSLATION_M = 0.018
    EXIT_MAX_RECENT_ROTATION_RAD = np.deg2rad(3.0)
    EXIT_STRONG_TRANSLATION_M = 0.055
    EXIT_EARLY_TRANSLATION_M = 0.006
    EXIT_EARLY_DOMINANCE = 1.5
    EXIT_EARLY_WINDOW_S = 0.08
    EXIT_EARLY_CONFIRM_S = 0.01

    def __init__(self, axes, jump_deg):
        self.axes = list(axes)
        self.jump_rad = np.deg2rad(jump_deg)
        self.origin_position = None
        self.origin_rotation = None
        self.last_rotation = None
        self.mode = "translation"
        self.last_switch = float("-inf")
        self.translation_blocked = False
        self.wrist_exit_pending = False
        self.wrist_translation_blocked = False
        self.wrist_exit_candidate_since = None
        self.recent_samples = deque()

    def reset(self, position, rotation, now, *, mode="translation"):
        self.mode = mode
        self.last_switch = now
        self.translation_blocked = False
        self.wrist_exit_pending = False
        self.wrist_translation_blocked = False
        self.wrist_exit_candidate_since = None
        self._rebase(position, rotation)
        self.recent_samples.clear()
        self.recent_samples.append((now, np.array(position, copy=True), np.array(rotation, copy=True)))

    def _rebase(self, position, rotation):
        self.origin_position = np.array(position, copy=True)
        self.origin_rotation = np.array(rotation, copy=True)
        self.last_rotation = np.array(rotation, copy=True)

    def update(self, position, rotation, now):
        position = np.asarray(position)
        rotation = np.asarray(rotation)
        if self.origin_position is None:
            self.reset(position, rotation, now)
            return False
        if np.linalg.norm(rotation_log(self.last_rotation.T @ rotation)) > self.jump_rad:
            # Tracking relocalization must not be interpreted as intent.
            self.reset(position, rotation, now, mode=self.mode)
            return False
        self.last_rotation = rotation.copy()
        self.recent_samples.append((now, position.copy(), rotation.copy()))
        while len(self.recent_samples) > 1 and now - self.recent_samples[0][0] > self.RECENT_WINDOW_S:
            self.recent_samples.popleft()
        recent_time, recent_position, recent_rotation = self.recent_samples[0]
        recent_span = now - recent_time
        recent_translation = np.linalg.norm(position - recent_position)
        recent_rotation_rad = np.linalg.norm(
            rotation_log(recent_rotation.T @ rotation)[self.axes]
        )
        translation_distance = np.linalg.norm(position - self.origin_position)
        translation_score = translation_distance / self.TRANSLATION_M
        # Express the gesture in the controller's local/ray frame.  The mapped
        # axes then keep their physical meaning at every hand orientation:
        # local Y can control J4, X controls J5, and Z controls J6.
        rotation_score = np.linalg.norm(
            rotation_log(self.origin_rotation.T @ rotation)[self.axes]
        ) / self.ROTATION_RAD
        translating = translation_score >= 1 and translation_score >= self.DOMINANCE * rotation_score
        rotating = rotation_score >= 1 and rotation_score >= self.DOMINANCE * translation_score
        self.translation_blocked = False
        self.wrist_exit_pending = False
        self.wrist_translation_blocked = False
        new_mode = self.mode
        entry_ready = (
            recent_span >= 0.12
            and recent_translation <= self.ENTRY_MAX_RECENT_TRANSLATION_M
            and recent_rotation_rad >= self.ENTRY_RECENT_ROTATION_RAD
            and rotation_score * self.ROTATION_RAD >= self.ENTRY_ROTATION_RAD
            and rotating
        )
        if self.mode == "translation" and entry_ready:
            new_mode = "wrist"
        elif self.mode == "translation":
            self.translation_blocked = (
                recent_span >= 0.08
                and recent_translation <= self.HOLD_MAX_RECENT_TRANSLATION_M
                and recent_rotation_rad >= self.HOLD_RECENT_ROTATION_RAD
                and rotation_score >= self.DOMINANCE * translation_score
            )
        elif self.mode == "wrist":
            # Give the controller an early braking signal before the full
            # mode-switch threshold/dwell. Do not mistake a pitch hand arc or
            # small pose jitter for an instruction to leave wrist control.
            early_time, early_position, early_rotation = next(
                s for s in self.recent_samples if now - s[0] <= self.EXIT_EARLY_WINDOW_S + 1e-9
            )
            early_translation = np.linalg.norm(position - early_position)
            early_rotation_rad = np.linalg.norm(rotation_log(early_rotation.T @ rotation)[self.axes])
            self.wrist_translation_blocked = (
                now - early_time >= .02 - 1e-9
                and early_translation >= self.EXIT_EARLY_TRANSLATION_M
                and early_rotation_rad <= self.EXIT_MAX_RECENT_ROTATION_RAD
                and early_translation / self.TRANSLATION_M
                >= self.EXIT_EARLY_DOMINANCE * early_rotation_rad / self.ROTATION_RAD
            )
            if self.wrist_translation_blocked:
                if self.wrist_exit_candidate_since is None:
                    self.wrist_exit_candidate_since = now
                self.wrist_exit_pending = now - self.wrist_exit_candidate_since >= self.EXIT_EARLY_CONFIRM_S - 1e-9
            else:
                self.wrist_exit_candidate_since = None
            selected_rotation = rotation_score * self.ROTATION_RAD
            if recent_translation >= self.EXIT_STRONG_TRANSLATION_M or (
                recent_translation >= self.EXIT_RECENT_TRANSLATION_M
                and recent_rotation_rad <= self.EXIT_MAX_RECENT_ROTATION_RAD
            ):
                new_mode = "translation"
            elif selected_rotation >= self.ROTATION_RAD:
                # Hand position naturally follows an arc during J5 pitch.
                # Keep the translation origin on that arc so it cannot be
                # mistaken for an instruction to return to Cartesian mode.
                self._rebase(position, rotation)
                return False
        if new_mode == self.mode:
            if self.mode == "translation" and translating:
                # Rebase during the current action, but accumulate small/slow
                # opposite-mode gestures until they cross the same threshold.
                self._rebase(position, rotation)
            return False
        if now - self.last_switch < self.DWELL_S:
            return False
        self.reset(position, rotation, now, mode=new_mode)
        return True


def ray_rotation_components(current, reference, axes):
    """Return selected ray swing axes and signed twist in the reference frame."""
    axes = list(axes)
    if len(axes) not in (2, 3) or len(set(axes)) != len(axes):
        raise ValueError("ray axes must be two or three distinct local axes")
    roll_axis = axes[-1]
    relative = reference.T @ current
    quaternion = rotation_to_quaternion_xyzw(relative)
    roll_unit = np.zeros(3, dtype=float)
    roll_unit[roll_axis] = 1.0
    projected = float(np.dot(quaternion[:3], roll_unit))
    twist_norm = float(np.hypot(projected, quaternion[3]))
    if twist_norm < 1.0e-10:
        roll = 0.0
        twist = np.eye(3)
    else:
        roll = 2.0 * np.arctan2(projected / twist_norm, quaternion[3] / twist_norm)
        roll = (roll + np.pi) % (2.0 * np.pi) - np.pi
        twist = rotation_exp(roll_unit * roll)
    swing = relative @ twist.T
    swing_vector = rotation_log(swing)
    components = [float(swing_vector[axis]) for axis in axes[:-1]]
    components.append(roll)
    return np.array(components), float(np.linalg.norm(rotation_log(relative)))


def select_wrist_component(
    raw_components,
    *,
    activation_deg=2.0,
    dominance=1.35,
):
    """Select one dominant ray axis; keep ambiguous motion unassigned."""
    magnitude_deg = np.abs(np.rad2deg(np.asarray(raw_components, dtype=float)))
    if magnitude_deg.ndim != 1 or magnitude_deg.size not in (2, 3):
        raise ValueError("expected two or three ray components")
    selected = int(np.argmax(magnitude_deg))
    other = max(float(value) for index, value in enumerate(magnitude_deg) if index != selected)
    if magnitude_deg[selected] < activation_deg:
        return None
    if magnitude_deg[selected] < dominance * max(other, 1.0e-9):
        return None
    return selected


def filter_wrist_rotation(
    current,
    reference,
    axes,
    deadband_deg,
    max_step_deg,
    *,
    active_component,
):
    """Consume ray swing/twist while commanding only the locked component."""
    raw, total_rotation = ray_rotation_components(current, reference, axes)
    if np.rad2deg(total_rotation) > max_step_deg:
        return np.zeros(len(raw)), current.copy(), True
    axes = list(axes)
    deadband = np.deg2rad(deadband_deg)
    filtered = np.sign(raw) * np.maximum(np.abs(raw) - deadband, 0.0)
    if active_component not in range(len(raw)):
        raise ValueError("active_component must name a selected ray axis")
    filtered[np.arange(len(raw)) != active_component] = 0.0
    residual = raw - filtered
    residual[np.arange(len(raw)) != active_component] = 0.0
    swing_residual = np.zeros(3, dtype=float)
    for component, axis in zip(residual[:-1], axes[:-1]):
        swing_residual[axis] = component
    roll_unit = np.zeros(3, dtype=float)
    roll_unit[axes[-1]] = 1.0
    residual_rotation = (
        rotation_exp(swing_residual)
        @ rotation_exp(roll_unit * residual[-1])
    )
    return filtered, current @ residual_rotation.T, False
