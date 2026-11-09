"""LeRobot frame builder for one physical left RM65."""

import numpy as np

from .frame_builder import CollectionFrameBuilder, FrameValidationError
from .models import CollectionFrame, SourceBundle


class LeftArmFrameBuilder(CollectionFrameBuilder):
    """Build a seven-dimensional state/action frame for the left arm."""

    CAMERA_ROLES = ("top", "left_wrist")

    def build(self, sample_time: float, sources: SourceBundle) -> CollectionFrame:
        sample_time = float(sample_time)
        arm = sources.arms.get("left_arm")
        gripper = sources.grippers.get("left_arm")
        if arm is None:
            raise FrameValidationError("left arm source missing")
        if gripper is None:
            raise FrameValidationError("left gripper source missing")

        feedback = self._six(arm.get("feedback_positions"), "left arm feedback")
        action_value = arm.get("action_positions")
        if action_value is None:
            action_value = arm.get("sent_positions")
        if action_value is None:
            action_value = feedback
        action = self._six(action_value, "left arm action")

        position = gripper.get("position")
        target = gripper.get("target_position")
        if position is None or not 0 <= float(position) <= 100:
            raise FrameValidationError("left gripper feedback invalid")
        if target is None or not 0 <= float(target) <= 100:
            raise FrameValidationError("left gripper action invalid")

        ages = {}
        timestamps = {}
        for label, timestamp, limit in (
            ("left_arm", arm.get("feedback_time"), self.max_arm_age_s),
            ("left_gripper", gripper.get("feedback_time"), self.max_gripper_age_s),
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
            observation_state=np.concatenate((feedback, [position])).astype(np.float32),
            action=np.concatenate((action, [target])).astype(np.float32),
            images=images,
            source_ages_ms=ages,
            source_timestamps=timestamps,
        )
