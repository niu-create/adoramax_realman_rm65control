"""Three-axis relative tool orientation, independent of individual joints."""

import numpy as np

from .transforms import rotation_angle_rad, rotation_exp, rotation_log, validate_rotation
from .auto_wrist import filter_wrist_rotation, ray_rotation_components, select_wrist_component


class RayWristIntent:
    """Keep the commissioned local ray directions as virtual joint intents.

    Axis selection does not wait for the arm to finish the previous target.
    The increments generate an orientation goal, followed by whole-arm IK.
    """

    def __init__(self, rotation, axes, signs, scales, *, deadband_deg, jump_deg):
        self.reference = validate_rotation(rotation).copy()
        self.last_raw = self.reference.copy()
        self.axes = tuple(axes)
        self.signs = np.asarray(signs)
        self.scales = np.asarray(scales)
        self.deadband_deg = deadband_deg
        self.jump_deg = jump_deg
        self.component = None
        self.last_active = None
        self.last_component = None
        self.component_changed = False
        self.selection_reference = self.reference.copy()

    def discard_rotation(self, rotation):
        self.reference = validate_rotation(rotation).copy()
        self.last_raw = self.reference.copy()
        self.selection_reference = self.reference.copy()

    def update(self, rotation, now, *, settled):
        # `settled` remains accepted for callers, but hand intent must not
        # wait for an old robot target to finish before selecting another axis.
        self.component_changed = False
        rotation = validate_rotation(rotation)
        if np.degrees(rotation_angle_rad(self.last_raw, rotation)) > self.jump_deg:
            raise ValueError("XR orientation jumped during assisted wrist control")
        self.last_raw = rotation.copy()
        candidate, _ = ray_rotation_components(rotation, self.selection_reference, self.axes)
        selected = select_wrist_component(candidate)
        if (self.component is not None and selected is not None and selected != self.component
                and now - self.last_active >= .18):
            self.component = None
            self.reference = self.selection_reference.copy()
        if self.component is None:
            self.component = selected
            if self.component is None:
                return np.zeros(len(self.axes))
            self.reference = self.selection_reference.copy()
            self.component_changed = self.last_component is not None and self.component != self.last_component
            self.last_component = self.component
            self.last_active = now
        filtered, reference, jumped = filter_wrist_rotation(
            rotation, self.reference, self.axes, self.deadband_deg,
            self.jump_deg, active_component=self.component,
        )
        if jumped:
            raise ValueError("XR ray rotation jumped during assisted wrist control")
        self.reference = reference
        if np.max(np.abs(filtered)) > 1e-10:
            self.last_active = now
            self.selection_reference = self.reference.copy()
        elif now - self.last_active >= .18 and np.max(np.abs(candidate)) <= np.deg2rad(self.deadband_deg) + 1e-9:
            self.component = None
            # Keep a nascent next-axis gesture across unlock. Rebasing to
            # current here would repeatedly eat slow sub-deadband pitch.
            self.reference = self.selection_reference.copy()
        return filtered * self.signs * self.scales


class PoseOrientationTarget:
    """Follow all rotation axes with a radial deadband and a raw-frame jump guard.

    The fixed basis maps the input tracking frame to the robot work frame.
    Deadband is measured from the last accepted rotation, so slow movement
    accumulates, while a stationary hand never integrates additional rotation.
    """

    def __init__(self, hand_rotation, tcp_rotation, basis, *, deadband_deg, max_step_deg):
        self.reference = validate_rotation(hand_rotation).copy()
        self.last_raw = self.reference.copy()
        self.target = validate_rotation(tcp_rotation).copy()
        self.basis = np.asarray(basis, dtype=float).copy()
        if (self.basis.shape != (3, 3) or not np.all(np.isfinite(self.basis))
                or not np.allclose(self.basis @ self.basis.T, np.eye(3), atol=1e-6)):
            raise ValueError("pose orientation basis must be orthogonal")
        self.deadband = float(np.deg2rad(deadband_deg))
        self.max_step = float(np.deg2rad(max_step_deg))
        if not 0 <= self.deadband < self.max_step <= np.pi:
            raise ValueError("invalid pose orientation deadband or jump limit")

    def update(self, hand_rotation):
        current = validate_rotation(hand_rotation)
        if rotation_angle_rad(current, self.last_raw) > self.max_step:
            raise ValueError("XR orientation jumped beyond the per-sample limit")
        self.last_raw = current.copy()
        delta = rotation_log(current @ self.reference.T)
        angle = float(np.linalg.norm(delta))
        if angle > self.deadband + 1e-12:
            accepted = rotation_exp(delta * ((angle - self.deadband) / angle))
            self.reference = accepted @ self.reference
            self.target = self.basis @ accepted @ self.basis.T @ self.target
        return self.target.copy()
