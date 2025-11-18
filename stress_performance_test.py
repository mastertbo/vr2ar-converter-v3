"""Synthetic stress test for downscale/upsample performance.

This runs a loop that mimics SAM2 mask resizing on 8K frames without
requiring any model weights. It exercises the OpenCV path used for
mask downscaling and upsampling to validate CPU throughput.
"""

from __future__ import annotations

import time
from typing import Tuple

import cv2
import numpy as np


def _downscale_and_upsample(frame: np.ndarray, max_side: int) -> Tuple[np.ndarray, np.ndarray]:
    """Downscale an 8K frame to ``max_side`` and upsample a synthetic mask back."""
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width)) if max_side else 1.0
    target_h = max(1, int(round(height * scale)))
    target_w = max(1, int(round(width * scale)))

    frame_small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    mask_small = np.random.rand(target_h, target_w).astype(np.float32)
    mask_uint8 = (mask_small * 255).astype(np.uint8)
    mask_full = cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame_small, mask_full


def run(iterations: int = 25, max_side: int = 2048) -> None:
    frame = np.random.randint(0, 256, (4320, 7680, 3), dtype=np.uint8)
    start = time.perf_counter()
    for _ in range(iterations):
        _downscale_and_upsample(frame, max_side)
    elapsed = time.perf_counter() - start
    fps = iterations / elapsed
    print(f"Ran {iterations} iterations on 8K frame -> {max_side}px in {elapsed:.2f}s ({fps:.2f} it/s)")


if __name__ == "__main__":
    run()
