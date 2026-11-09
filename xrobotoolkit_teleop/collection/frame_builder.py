from typing import Dict

import numpy as np

from .models import CollectionFrame, SourceBundle


class FrameValidationError(RuntimeError):
    """A source bundle cannot safely become a training frame."""


class CollectionFrameBuilder:
    CAMERA_ROLES = ("top", "left_wrist", "right_wrist")

    def __init__(
        self,
        *,
        max_gripper_age_s: float = 0.25,
        max_camera_age_s: float = 0.25,
        max_arm_age_s: float = 0.25,
    ):
        self.max_gripper_age_s = float(max_gripper_age_s)
        self.max_camera_age_s = float(max_camera_age_s)
        self.max_arm_age_s = float(max_arm_age_s)

    @staticmethod
    def _age(sample_time: float, timestamp, label: str, limit: float) -> float:
        if timestamp is None:
            raise FrameValidationError(f"{label} timestamp missing")
        age = float(sample_time) - float(timestamp)
        if age < 0.0:
            raise FrameValidationError(f"{label} timestamp is in the future")
        if age > limit:
            raise FrameValidationError(
                f"{label} stale: age_ms={age * 1000.0:.3f}, "
                f"limit_ms={limit * 1000.0:.3f}"
            )
        return age

    @staticmethod
    def _six(value, label: str) -> np.ndarray:
        if value is None:
            raise FrameValidationError(f"{label} missing")
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (6,) or not np.all(np.isfinite(array)):
            raise FrameValidationError(f"{label} must be six finite values")
        return array

    def build(self, sample_time: float, sources: SourceBundle) -> CollectionFrame:
        sample_time = float(sample_time)
        ages: Dict[str, float] = {}
        timestamps: Dict[str, float] = {}
        state_parts = []
        action_parts = []

        for side in ("left", "right"):
            arm_name = f"{side}_arm"
            if arm_name not in sources.arms:
                raise FrameValidationError(f"{side} arm source missing")
            if arm_name not in sources.grippers:
                raise FrameValidationError(f"{side} gripper source missing")
            arm = sources.arms[arm_name]
            gripper = sources.grippers[arm_name]
            feedback = self._six(
                arm.get("feedback_positions"), f"{side} arm feedback"
            )
            action_value = arm.get("action_positions")
            if action_value is None:
                action_value = arm.get("sent_positions")
            if action_value is None:
                action_value = feedback
            action = self._six(action_value, f"{side} arm action")
            position = gripper.get("position")
            target = gripper.get("target_position")
            if position is None or not 0 <= float(position) <= 100:
                raise FrameValidationError(f"{side} gripper feedback invalid")
            if target is None or not 0 <= float(target) <= 100:
                raise FrameValidationError(f"{side} gripper action invalid")

            arm_age = self._age(
                sample_time,
                arm.get("feedback_time"),
                f"{side} arm",
                self.max_arm_age_s,
            )
            gripper_age = self._age(
                sample_time,
                gripper.get("feedback_time"),
                f"{side} gripper",
                self.max_gripper_age_s,
            )
            ages[f"{side}_arm"] = arm_age * 1000.0
            ages[f"{side}_gripper"] = gripper_age * 1000.0
            timestamps[f"{side}_arm"] = float(arm["feedback_time"])
            timestamps[f"{side}_gripper"] = float(gripper["feedback_time"])
            state_parts.extend((feedback, np.array([position], dtype=np.float32)))
            action_parts.extend((action, np.array([target], dtype=np.float32)))

        images = {}
        for role in self.CAMERA_ROLES:
            sample = sources.cameras.get(role)
            if sample is None:
                raise FrameValidationError(f"{role} camera source missing")
            image = np.asarray(sample.image)
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise FrameValidationError(f"{role} camera frame invalid")
            age = self._age(
                sample_time,
                sample.monotonic_time,
                f"{role} camera",
                self.max_camera_age_s,
            )
            ages[f"{role}_camera"] = age * 1000.0
            timestamps[f"{role}_camera"] = float(sample.monotonic_time)
            images[f"observation.images.{role}"] = image.copy()

        return CollectionFrame(
            monotonic_time=sample_time,
            observation_state=np.concatenate(state_parts).astype(
                np.float32, copy=False
            ),
            action=np.concatenate(action_parts).astype(np.float32, copy=False),
            images=images,
            source_ages_ms=ages,
            source_timestamps=timestamps,
        )


class RightArmFrameBuilder(CollectionFrameBuilder):
    """Build pi0.5/LeRobot frames from one physical right RM65."""

    def build(self, sample_time: float, sources: SourceBundle) -> CollectionFrame:
        sample_time = float(sample_time)
        arm = sources.arms.get("right_arm")
        gripper = sources.grippers.get("right_arm")
        if arm is None:
            raise FrameValidationError("right arm source missing")
        if gripper is None:
            raise FrameValidationError("right gripper source missing")
        feedback = self._six(arm.get("feedback_positions"), "right arm feedback")
        action_value = arm.get("action_positions")
        if action_value is None:
            action_value = arm.get("sent_positions")
        if action_value is None:
            action_value = feedback
        action = self._six(action_value, "right arm action")
        position = gripper.get("position")
        target = gripper.get("target_position")
        if position is None or not 0 <= float(position) <= 100:
            raise FrameValidationError("right gripper feedback invalid")
        if target is None or not 0 <= float(target) <= 100:
            raise FrameValidationError("right gripper action invalid")

        ages = {}
        timestamps = {}
        for label, timestamp, limit in (
            ("right_arm", arm.get("feedback_time"), self.max_arm_age_s),
            ("right_gripper", gripper.get("feedback_time"), self.max_gripper_age_s),
        ):
            age = self._age(sample_time, timestamp, label.replace("_", " "), limit)
            ages[label] = age * 1000.0
            timestamps[label] = float(timestamp)

        images = {}
        for role in self.CAMERA_ROLES:
            sample = sources.cameras.get(role)
            if sample is None:
                raise FrameValidationError(f"{role} camera source missing")
            image = np.asarray(sample.image)
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise FrameValidationError(f"{role} camera frame invalid")
            age = self._age(sample_time, sample.monotonic_time, f"{role} camera", self.max_camera_age_s)
            ages[f"{role}_camera"] = age * 1000.0
            timestamps[f"{role}_camera"] = float(sample.monotonic_time)
            images[f"observation.images.{role}"] = image.copy()

        return CollectionFrame(
            monotonic_time=sample_time,
            observation_state=np.concatenate((feedback, [position])).astype(np.float32),
            action=np.concatenate((action, [target])).astype(np.float32),
            images=images,
            source_ages_ms=ages,
            source_timestamps=timestamps,
        )
