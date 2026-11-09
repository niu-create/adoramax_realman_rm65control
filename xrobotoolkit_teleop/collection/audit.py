import json
import threading
from pathlib import Path


class JsonlAuditWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, event: dict) -> None:
        with self._lock:
            self._stream.write(
                json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if not self._stream.closed:
                self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
