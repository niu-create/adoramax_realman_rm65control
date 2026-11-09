"""Preview cached collection frames without opening camera devices again."""

import time
import textwrap

import numpy as np


class CameraPreviewKeyboard:
    """Display both cameras and handle window keys on the main thread."""

    WINDOW = "RM65 cameras | HEAD / RIGHT WRIST"
    CAMERA_ROLES = (("top", "HEAD"), ("right_wrist", "RIGHT WRIST"))

    def __init__(self, keyboard, cameras, recording, *, status=None, notice=None, cv_module=None,
                 camera_roles=None):
        self.camera_roles = tuple(self.CAMERA_ROLES if camera_roles is None else camera_roles)
        if not self.camera_roles or any(role not in cameras for role, _ in self.camera_roles):
            raise ValueError("preview roles must refer to available cameras")
        self.WINDOW = "RM65 cameras | " + " / ".join(label for _, label in self.camera_roles)
        if cv_module is None:
            import cv2 as cv_module
        self.cv = cv_module
        self.keyboard = keyboard
        self.cameras = cameras
        self.recording = recording
        self.status = status
        self.notice = notice
        self.next_frame = 0.0
        self._window_available = True
        self.cv.setNumThreads(1)
        try:
            self.cv.namedWindow(self.WINDOW, self.cv.WINDOW_NORMAL)
            self.cv.resizeWindow(self.WINDOW, 1280, 480)
            self.refresh()
            self.cv.waitKeyEx(1)
        except self.cv.error as exc:
            # OpenCV can report a NULL Qt window when the desktop session or
            # window manager closes the preview during startup. Collection
            # remains usable through TerminalKeyboard in that case.
            self._window_available = False
            print(f"[CAMERA PREVIEW] Window unavailable; continuing without preview: {exc}", flush=True)

    def refresh(self):
        if not getattr(self, "_window_available", True):
            return
        now = time.monotonic()
        if now < self.next_frame:
            return
        panels = []
        for role, label in getattr(self, "camera_roles", self.CAMERA_ROLES):
            frame = self.cameras[role].snapshot()
            panel = self.cv.cvtColor(frame.image, self.cv.COLOR_RGB2BGR)
            recording = self.recording()
            state = self.status() if self.status else ("REC" if recording else "READY")
            age_ms = max(0., now - frame.monotonic_time) * 1000
            self.cv.putText(panel, f"{label} | {state} | {age_ms:.0f} ms",
                            (12, 28), self.cv.FONT_HERSHEY_SIMPLEX, .65,
                            (0, 0, 255) if recording else (0, 255, 0), 2)
            if state in ("INVALID", "REJECTED"):
                self.cv.rectangle(panel, (0, 40), (panel.shape[1], 160), (0, 0, 160), -1)
                message = ("BAD DATA - PRESS S TO DISCARD" if state == "INVALID"
                           else "NOT SAVED - DISCARDED - ENTER TO RETRY")
                self.cv.putText(panel, message, (12, 65), self.cv.FONT_HERSHEY_SIMPLEX,
                                .55, (255, 255, 255), 2)
                reason = self.notice() if self.notice else ""
                width = max(16, int((panel.shape[1] - 24) / 9))
                for index, line in enumerate(textwrap.wrap(reason, width=width)[:3]):
                    self.cv.putText(panel, line, (12, 92 + index * 22), self.cv.FONT_HERSHEY_SIMPLEX,
                                    .45, (255, 255, 255), 1)
            panels.append(panel)
        self.cv.imshow(self.WINDOW, np.concatenate(panels, axis=1))
        self.next_frame = now + 1. / 15.

    def poll(self, timeout_s):
        if not getattr(self, "_window_available", True):
            return self.keyboard.poll(timeout_s)
        try:
            self.refresh()
            key = self.cv.waitKeyEx(1)
            visible = self.cv.getWindowProperty(self.WINDOW, self.cv.WND_PROP_VISIBLE)
        except self.cv.error as exc:
            self._window_available = False
            print(f"[CAMERA PREVIEW] Window closed by GUI; continuing without preview: {exc}", flush=True)
            return self.keyboard.poll(timeout_s)
        if visible < 1:
            self._window_available = False
            print("[CAMERA PREVIEW] Window closed; continuing without preview.", flush=True)
            return self.keyboard.poll(timeout_s)
        action = {13: "ENTER", 10: "ENTER", 27: "ESC", 115: "RIGHT", 83: "RIGHT",
                  65363: "RIGHT", 2555904: "RIGHT",
                  65361: "LEFT", 2424832: "LEFT"}.get(key)
        return action if action is not None else self.keyboard.poll(timeout_s)

    def close(self):
        try:
            self.cv.destroyWindow(self.WINDOW)
        except self.cv.error:
            pass
