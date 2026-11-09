from .models import CollectionFrame


class EpisodeStateError(RuntimeError):
    pass


class EpisodeManager:
    def __init__(self, sink, *, task: str, preserve_invalid: bool = False,
                 backup_valid: bool = False):
        if not str(task).strip():
            raise ValueError("task may not be empty")
        self.sink = sink
        self.task = str(task)
        self.recording = False
        self.invalid_reason = None
        self.frame_count = 0
        self.preserve_invalid = bool(preserve_invalid)
        self.backup_valid = bool(backup_valid)
        self.quality_issues = []

    def start(self) -> None:
        if self.recording:
            raise EpisodeStateError("an episode is already active")
        self.invalid_reason = None
        self.frame_count = 0
        self.quality_issues = []
        begin = getattr(self.sink, "begin_episode", None)
        if callable(begin):
            begin(self.task)
        self.recording = True

    def append(self, frame: CollectionFrame) -> None:
        if not self.recording:
            raise EpisodeStateError("no active episode")
        if self.invalid_reason is not None and not self.preserve_invalid:
            raise EpisodeStateError(
                f"active episode is invalid: {self.invalid_reason}"
            )
        self.sink.add_frame(frame)
        self.frame_count += 1

    def invalidate(self, reason: str) -> None:
        if self.recording and str(reason) not in self.quality_issues:
            self.quality_issues.append(str(reason))
        if self.recording and self.invalid_reason is None:
            self.invalid_reason = str(reason)

    def accept(self):
        """Return 'archived' for quarantined raw data, None for accepted data."""
        if not self.recording:
            raise EpisodeStateError("no active episode")
        if self.invalid_reason is not None and not self.preserve_invalid:
            raise EpisodeStateError(
                f"cannot accept invalid episode: {self.invalid_reason}"
            )
        if self.frame_count == 0:
            raise EpisodeStateError("cannot accept an empty episode")
        if self.preserve_invalid or self.backup_valid:
            # Before LeRobot mutates its buffer or deletes source images.
            # Failure here leaves the active episode available for retry.
            self.sink.checkpoint(self.quality_issues)
            if self.invalid_reason is not None:
                self.sink.discard_episode()
                self.recording = False
                self.frame_count = 0
                return "archived"
        try:
            self.sink.accept_episode(self.task)
        except BaseException:
            # LeRobot mutates its buffer before encoding. Discarding it after
            # an interrupted save both raises and can erase recoverable images.
            self.recording = False
            self.invalid_reason = "save_failed; files preserved for recovery"
            raise
        self.recording = False
        self.frame_count = 0

    def discard(self, reason: str = "operator") -> None:
        if not self.recording:
            raise EpisodeStateError("no active episode")
        self.sink.discard_episode()
        self.recording = False
        self.invalid_reason = str(reason)
        self.frame_count = 0

    def close(self) -> None:
        if self.recording:
            if self.preserve_invalid and self.frame_count:
                self.invalidate("unfinished_on_exit")
                self.accept()
            else:
                self.discard("unfinished_on_exit")
