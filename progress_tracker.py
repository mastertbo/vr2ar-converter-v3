import json
import os
import threading
from typing import Any, Dict


class ProcessingProgress:
    """Persistent progress tracker for long running video jobs."""

    _lock = threading.Lock()

    def __init__(self, base_dir: str, metadata: Dict[str, Any]):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.path = os.path.join(self.base_dir, "progress.json")
        self.state: Dict[str, Any] = {
            "job_id": metadata.get("job_id"),
            "video": metadata.get("video"),
            "projection": metadata.get("projection"),
            "last_frame": 0,
            "mask_idx": 0,
            "reverse_queue": [],
            "completed": False,
            "result": None,
            "total_frames": metadata.get("total_frames"),
            "version": metadata.get("version"),
        }
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    persisted = json.load(fh)
                if self._compatible(persisted, metadata):
                    self.state.update(persisted)
                else:
                    self._write_state()
            except json.JSONDecodeError:
                self._write_state()
        else:
            self._write_state()

    def _write_state(self) -> None:
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=2)
        os.replace(tmp_path, self.path)

    def _compatible(self, persisted: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        if persisted.get("job_id") != metadata.get("job_id"):
            return False
        if persisted.get("video") != metadata.get("video"):
            return False
        if persisted.get("projection") != metadata.get("projection"):
            return False
        return True

    def update(self, *, last_frame: int, mask_idx: int, total_frames: int | None = None,
               reverse_queue: list[int] | None = None) -> None:
        with self._lock:
            self.state["last_frame"] = max(self.state.get("last_frame", 0), last_frame)
            self.state["mask_idx"] = mask_idx
            if total_frames is not None:
                self.state["total_frames"] = total_frames
            if reverse_queue is not None:
                self.state["reverse_queue"] = reverse_queue
            self._write_state()

    def mark_completed(self, result_path: str) -> None:
        with self._lock:
            self.state["completed"] = True
            self.state["result"] = result_path
            self._write_state()

    def reset(self) -> None:
        with self._lock:
            self.state.update({
                "last_frame": 0,
                "mask_idx": 0,
                "reverse_queue": [],
                "completed": False,
                "result": None,
            })
            self._write_state()

    @property
    def last_frame(self) -> int:
        return int(self.state.get("last_frame", 0))

    @property
    def mask_idx(self) -> int:
        return int(self.state.get("mask_idx", 0))

    @property
    def reverse_queue(self) -> list[int]:
        return list(self.state.get("reverse_queue", []))

    @property
    def completed(self) -> bool:
        return bool(self.state.get("completed", False))

    @property
    def result(self) -> str | None:
        return self.state.get("result")

    def clear(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)
