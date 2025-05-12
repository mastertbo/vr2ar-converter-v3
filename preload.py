#!/usr/bin/env python3
import os
import sys
import threading
import time
from PIL import Image

# Import your core processing function from main.py
# (Make sure main.py and preload.py live in the same directory, or adjust PYTHONPATH)
from main import process  

# Configuration
INPUT_DIR  = "segment"            # folder you mount/upload your videos into
OUTPUT_DIR = "segment/processed"  # where processed files will land
PROJECTION = "eq"                 # default projection
CRF        = 16                   # default encoding parameter
ERODE      = False                # default erode setting

# Prepare output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

def batch_process():
    video_files = sorted([
        f for f in os.listdir(INPUT_DIR) 
        if f.lower().endswith((".mp4", ".mkv"))
    ])

    if not video_files:
        print("No video files found in", INPUT_DIR)
        sys.exit(1)

    for idx, fname in enumerate(video_files, start=1):
        input_path = os.path.join(INPUT_DIR, fname)
        # Create a fake file-like object that mimics what main.process expects:
        class V:
            def __init__(self, path):
                self.name = path
        video_obj = V(input_path)

        print(f"[{idx}/{len(video_files)}] Processing {fname}...")
        try:
            # process(job_id, video, projection, maskL, maskR, crf, erode)
            # Here we pass a blank mask (all zeros) for L/R so your function will still run:
            blank_mask = Image.new("L", (1,1), color=0)
            _, result_path = process(
                idx, 
                video_obj, 
                PROJECTION, 
                blank_mask,  # maskL
                blank_mask,  # maskR
                CRF, 
                ERODE
            )
            # Move/rename the produced file into OUTPUT_DIR
            if result_path and os.path.exists(result_path):
                base = os.path.basename(result_path)
                dest = os.path.join(OUTPUT_DIR, base)
                os.replace(result_path, dest)
                print("  → saved:", dest)
            else:
                print("  ⚠️ no output produced for", fname)
        except Exception as e:
            print("  ❌ error processing", fname, ":", e)

    print("✅ Batch processing complete.")

if __name__ == "__main__":
    batch_process()
