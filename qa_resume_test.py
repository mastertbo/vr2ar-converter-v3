"""
QA script to validate resumable processing by simulating a crash mid-job and verifying resume.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import prepare_frame, fix_mask2  # noqa: E402
import processing_core  # noqa: E402


def _create_sample_video(video_path: Path, width: int = 512, height: int = 256, fps: int = 12, seconds: int = 5) -> None:
    """Generate a small synthetic side-by-side clip using ffmpeg's test source."""
    duration = max(seconds, 1)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "lavfi",
        f"-i",
        f"testsrc=size={width}x{height}:rate={fps}",
        "-vf",
        "format=yuv420p",
        "-t",
        str(duration),
        str(video_path),
    ]
    subprocess.run(cmd, check=True)


def _build_mask_payload(video_path: Path) -> Dict[str, object]:
    """Extract the first frame and craft the mask payload expected by the pipeline."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read sample video")

    height, width = frame.shape[:2]
    half_width = width // 2
    frameL = frame[:, :half_width]
    frameR = frame[:, half_width:]

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    maskL = Image.fromarray(np.full_like(grayL, 255, dtype=np.uint8), mode="L")
    maskR = Image.fromarray(np.full_like(grayR, 255, dtype=np.uint8), mode="L")

    return {
        "maskL": maskL,
        "maskR": maskR,
        "frameL": frameL,
        "frameR": frameR,
        "frameLGray": grayL,
        "frameRGray": grayR,
    }


def _run_process(job_id: str, video_path: Path, masks, crash_after_frame: int | None):
    def set_status(message: str) -> None:
        if crash_after_frame and message.startswith("Create Mask"):
            current = int(message.split()[2].split("/")[0])
            if current >= crash_after_frame:
                raise RuntimeError("Simulated crash")

    helpers = {
        "prepare_frame": prepare_frame,
        "fix_mask2": fix_mask2,
        "set_status": set_status,
    }

    return processing_core.process_video(
        video=str(video_path),
        projection="fisheye180",
        masks=masks,
        crf=28,
        erode=False,
        force_init_mask=False,
        reverse_tracking=False,
        helpers=helpers,
        job_id=job_id,
        job_version=999,
        warmup=1,
        ssim_threshold=0.7,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        os.chdir(workdir)

        (workdir / "process" / "runs").mkdir(parents=True, exist_ok=True)
        (workdir / "jobs").mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / "mask.png", workdir / "mask.png")

        video_path = workdir / "sample.mp4"
        _create_sample_video(video_path)
        mask_payload = _build_mask_payload(video_path)
        masks = [mask_payload]

        job_id = processing_core._compute_job_id(str(video_path), "fisheye180")
        progress_path = workdir / "process" / "runs" / job_id / "progress.json"

        try:
            _run_process(job_id, video_path, masks, crash_after_frame=3)
        except RuntimeError as exc:
            if "Simulated crash" not in str(exc):
                raise

        if not progress_path.exists():
            raise AssertionError("Progress file missing after simulated crash")

        partial = json.loads(progress_path.read_text())
        if partial.get("completed"):
            raise AssertionError("Job unexpectedly marked completed after crash")
        if partial.get("last_frame", 0) < 3:
            raise AssertionError("Checkpoint did not capture expected progress")

        result_path = _run_process(job_id, video_path, masks, crash_after_frame=None)
        completed = json.loads(progress_path.read_text())
        mask_dir = progress_path.parent / "masks"
        rendered = sorted(mask_dir.glob("*.png"))

        assert completed.get("completed"), "Resume run did not mark job completed"
        assert completed.get("last_frame") == completed.get("total_frames"), "Last frame does not match total"
        assert result_path and Path(result_path).exists(), "Result video not produced"
        assert rendered, "Mask frames missing after resume"

        print("Resume QA test succeeded")
        print(f"Checkpoint: {progress_path}")
        print(f"Result video: {result_path}")


if __name__ == "__main__":
    main()
