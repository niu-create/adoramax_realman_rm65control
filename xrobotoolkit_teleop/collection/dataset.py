import inspect

import numpy as np

from .models import CollectionFrame


class LeRobotDatasetEnvironmentError(RuntimeError):
    pass


_AUTO = object()


class LeRobotDatasetSink:
    """Narrow adapter over the installed LeRobot episode writer."""

    CAMERA_KEYS = (
        "observation.images.top",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )

    def __init__(self, dataset):
        self.dataset = dataset
        self._task = None

    @classmethod
    def create(
        cls,
        *,
        root,
        repo_id: str,
        fps: int = 30,
        image_height: int = 480,
        image_width: int = 640,
        dimension_names=None,
        image_writer_threads: int = 8,
        dataset_class=_AUTO,
    ):
        if dataset_class is _AUTO:
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
            except Exception as exc:
                raise LeRobotDatasetEnvironmentError(
                    "LeRobotDataset is unavailable; activate the verified "
                    "LeRobot collection environment"
                ) from exc
            dataset_class = LeRobotDataset
        if dataset_class is None or not callable(
            getattr(dataset_class, "create", None)
        ):
            raise LeRobotDatasetEnvironmentError(
                "LeRobotDataset.create is unavailable or incompatible"
            )
        if dimension_names is None:
            dimension_names = [
                "left_q1", "left_q2", "left_q3", "left_q4", "left_q5",
                "left_q6", "left_gripper", "right_q1", "right_q2",
                "right_q3", "right_q4", "right_q5", "right_q6",
                "right_gripper",
            ]
        dimension_names = list(dimension_names)
        image_writer_threads = int(image_writer_threads)
        if image_writer_threads < 0:
            raise ValueError("image_writer_threads may not be negative")
        if not dimension_names or len(set(dimension_names)) != len(dimension_names):
            raise ValueError("dimension_names must be nonempty and unique")
        feature_size = len(dimension_names)
        features = {
            "observation.state": {
                "dtype": "float32", "shape": (feature_size,),
                "names": dimension_names,
            },
            "action": {
                "dtype": "float32", "shape": (feature_size,),
                "names": dimension_names,
            },
        }
        for key in cls.CAMERA_KEYS:
            features[key] = {
                "dtype": "video",
                "shape": (image_height, image_width, 3),
                "names": ["height", "width", "channel"],
            }
        try:
            dataset = dataset_class.create(
                repo_id=repo_id,
                fps=int(fps),
                root=root,
                features=features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=image_writer_threads,
            )
        except TypeError as exc:
            raise LeRobotDatasetEnvironmentError(
                "installed LeRobotDataset.create API is incompatible"
            ) from exc
        return cls(dataset)

    def add_frame(self, frame: CollectionFrame) -> None:
        payload = {
            "observation.state": np.asarray(
                frame.observation_state, dtype=np.float32
            ),
            "action": np.asarray(frame.action, dtype=np.float32),
            **{key: np.asarray(value) for key, value in frame.images.items()},
        }
        parameters = inspect.signature(self.dataset.add_frame).parameters
        if "task" in parameters:
            self.dataset.add_frame(payload, task=self._task)
        else:
            if self._task is not None:
                payload["task"] = self._task
            self.dataset.add_frame(payload)

    def begin_episode(self, task: str) -> None:
        self._task = str(task)

    def accept_episode(self, task: str) -> None:
        self._task = str(task)
        try:
            parameters = inspect.signature(
                self.dataset.save_episode
            ).parameters
            if "task" in parameters:
                self.dataset.save_episode(task=self._task)
            else:
                self.dataset.save_episode()
        finally:
            self._task = None

    def discard_episode(self) -> None:
        clear = getattr(self.dataset, "clear_episode_buffer", None)
        if callable(clear):
            clear()
            return
        buffer = getattr(self.dataset, "episode_buffer", None)
        if hasattr(buffer, "clear"):
            buffer.clear()
            return
        raise LeRobotDatasetEnvironmentError(
            "installed LeRobot dataset cannot discard an active episode"
        )
