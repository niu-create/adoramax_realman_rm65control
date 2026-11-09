"""Raw episode checkpoints, independent of LeRobot's destructive video save.

PNG files are hard-linked before encoding/clearing the LeRobot buffer. Numeric
data and real capture times are journaled as frames arrive. Checkpoints are kept
even after successful encoding; they are not automatically training data.
"""

import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np

from .dataset import LeRobotDatasetSink


class CheckpointError(RuntimeError):
    """The native buffer is untouched; the operator may retry saving."""


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class RecoverableLeRobotDatasetSink(LeRobotDatasetSink):
    KEEP_DISCARDED_BACKUP = True

    def __init__(self, dataset):
        super().__init__(dataset)
        self.recovery_path = None
        self._records = []

    def begin_episode(self, task):
        super().begin_episode(task)
        self.recovery_path = Path(self.dataset.root) / "recovery" / uuid.uuid4().hex
        self.recovery_path.mkdir(parents=True, exist_ok=False)
        self._records = []
        self._write_manifest("recording", [])

    def _write_manifest(self, status, reasons):
        _atomic_json(self.recovery_path / "manifest.json", {
            "version": 1, "status": status, "task": self._task,
            "fps": self.dataset.fps, "features": self.dataset.features,
            "frame_count": len(self._records), "quality_issues": list(reasons),
        })

    def add_frame(self, frame):
        super().add_frame(frame)
        buffer = self.dataset.episode_buffer
        record = {
            "monotonic_time": float(frame.monotonic_time),
            "observation_state": np.asarray(frame.observation_state).tolist(),
            "action": np.asarray(frame.action).tolist(),
            "source_ages_ms": dict(frame.source_ages_ms),
            "source_timestamps": dict(frame.source_timestamps),
            "images": {key: str(buffer[key][-1]) for key in self.CAMERA_KEYS},
        }
        # No image encoding or fsync in the sampling loop. The explicit save
        # checkpoint below flushes and verifies the complete raw episode.
        with (self.recovery_path / "journal.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._records.append(record)

    def checkpoint(self, reasons):
        try:
            from PIL import Image

            self.dataset._wait_image_writer()
            records = []
            for index, record in enumerate(self._records):
                images = {}
                for key, source in record["images"].items():
                    relative = Path("images") / key / f"{index:06d}.png"
                    target = self.recovery_path / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        try:
                            os.link(source, target)
                        except OSError:
                            temporary = target.with_suffix(".tmp")
                            shutil.copy2(source, temporary)
                            temporary.replace(target)
                    if target.stat().st_size == 0:
                        raise OSError(f"empty image: {target}")
                    with Image.open(target) as image:
                        image.verify()
                    with target.open("rb") as stream:
                        os.fsync(stream.fileno())
                    images[key] = str(relative)
                records.append({**record, "images": images})
            _atomic_json(self.recovery_path / "frames.json", records)
            # Only this final manifest marks a self-contained, complete backup.
            self._write_manifest("raw_saved", reasons)
        except Exception as exc:
            raise CheckpointError(
                f"原始数据备份未完成，尚未调用视频保存；可重试。{self.recovery_path}: {exc}"
            ) from exc
        print(f"EPISODE RAW SAVED: {self.recovery_path}", flush=True)
        return self.recovery_path

    def discard_episode(self):
        # LeRobot's clear operation does not wait for its asynchronous writer.
        self.dataset._wait_image_writer()
        manifest = self.recovery_path / "manifest.json"
        if json.loads(manifest.read_text(encoding="utf-8"))["status"] != "raw_saved":
            self._write_manifest("discarded", [])
        super().discard_episode()
        # With synchronous image writing, installed LeRobot leaves these
        # files behind. Unlink the exact source paths so the next episode
        # cannot overwrite an inode still used by a hard-linked checkpoint.
        for record in self._records:
            for source in record["images"].values():
                Path(source).unlink(missing_ok=True)
        if not self.KEEP_DISCARDED_BACKUP:
            # Only this sink's current UUID directory, never older episodes.
            path = self.recovery_path.resolve()
            expected_parent = (Path(self.dataset.root) / "recovery").resolve()
            if path.parent != expected_parent or len(path.name) != 32:
                raise RuntimeError(f"refusing to remove unexpected recovery path: {path}")
            shutil.rmtree(path)
        self._records = []
