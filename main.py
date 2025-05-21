import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import gradio as gr
import subprocess
import tempfile
import shutil
import pickle
import torch
import torch.nn as nn
import cv2
import gc
import glob
import time
import threading
import queue
import asyncio
import subprocess
import requests
import argparse
import json
import redis
import boto3
import io
import random
from sam2_executor import GroundingDinoSAM2Segment, SAM2PointSegment
from PIL import Image

import torch
# Detect available GPUs
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
# Assign left/right eye models to separate GPUs if available
if num_gpus >= 2:
    deviceL = torch.device('cuda:0')
    deviceR = torch.device('cuda:1')
elif num_gpus == 1:
    deviceL = deviceR = torch.device('cuda:0')
else:
    deviceL = deviceR = torch.device('cpu')

# --- Tiling configuration (can be overridden with environment variables) ---
TILE_ROWS    = int(os.getenv("TILE_ROWS", "2"))     # number of tile rows
TILE_COLS    = int(os.getenv("TILE_COLS", "2"))     # number of tile columns
TILE_OVERLAP = int(os.getenv("TILE_OVERLAP", "32")) # overlap (pixels) between tiles

import torch.nn.functional as F
import numpy as np

from matanyone.model.matanyone import MatAnyone
from matanyone.inference.inference_core import InferenceCore

from data.ffmpegstream import FFmpegStream
from data.ArVideoWriter import ArVideoWriter
from video_process import ImageFrame
from filebrowser_client import FilebrowserClient

WORKER_STATUS = "Idle"
MASK_SIZE = 1440

def gen_dilate(alpha, min_kernel_size, max_kernel_size): 
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)*255
    return dilate.astype(np.float32)

def gen_erosion(alpha, min_kernel_size, max_kernel_size): 
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    erode = cv2.erode(fg, kernel, iterations=1)*255
    return erode.astype(np.float32)


# --- Helper utilities for tiling ---
def split_into_tiles(img: np.ndarray, rows: int, cols: int, overlap: int):
    """Return cropped tiles and their (y0,y1,x0,x1) coords."""
    h, w = img.shape[:2]
    th, tw = h // rows, w // cols
    tiles, coords = [], []
    for r in range(rows):
        for c in range(cols):
            y0 = max(r * th - overlap, 0)
            x0 = max(c * tw - overlap, 0)
            y1 = min((r + 1) * th + overlap, h)
            x1 = min((c + 1) * tw + overlap, w)
            tiles.append(img[y0:y1, x0:x1].copy())
            coords.append((y0, y1, x0, x1))
    return tiles, coords

def blend_tile(target: np.ndarray, tile_mask: np.ndarray, coord):
    """Blend `tile_mask` into `target` at coord using max‑alpha."""
    y0, y1, x0, x1 = coord
    target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], tile_mask)


# Device‑aware prepare_frame
def prepare_frame(frame: np.ndarray, device: torch.device | None = None):
    tensor = torch.from_numpy(frame)
    if tensor.shape[-1] == 3:  # HWC → CHW
        tensor = tensor.permute(2, 0, 1)
    if device is not None:
        tensor = tensor.to(device, non_blocking=True)
    return tensor.float() / 255.0

def fix_mask2(mask):
    mask = np.array(mask)
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)
    mask = torch.from_numpy(mask)
    if torch.torch.cuda.is_available():
        mask = mask.cuda()

    return mask

@torch.no_grad()
# helper_masks: Optional[dict], e.g. {frame_number: mask}
def process(job_id, video, projection, maskL, maskR, crf = 16, erode = False, helper_masks=None):
    global WORKER_STATUS

    _, mask_h = maskL.size

    maskL = fix_mask2(maskL)
    maskR = fix_mask2(maskR)

    original_filename = os.path.basename(video)
    file_name, file_extension = os.path.splitext(original_filename)
    
    video_info = FFmpegStream.get_video_info(video)

    # --- helper_masks: allow for external keyframe mask injection ---
    helper_masks_local = helper_masks or {}
    
    reader_config = {
        "parameter": {
            "width": video_info.width,
            "height": video_info.height,
        }
    }
    
    if "eq" == projection:
        reader_config["filter_complex"] = "[0:v]split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; [right]crop=ih:ih:ih:0[right_crop]; [left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; [right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; [leftfisheye][rightfisheye]hstack[v]"
        projection = "fisheye180"

    WORKER_STATUS = f"Load Models to create Masks"

    current_frame = 0
    objects = [1]

    ffmpeg = FFmpegStream(
        video_path = video,
        config = reader_config,
        skip_frames = 0
    )

    result_tmp_name = file_name.replace(' ', '_') + "_" + str(projection).upper() + "_alpha_tmp" + file_extension 
    result_name = file_name.replace(' ', '_') + "_" + str(projection).upper() + "_alpha" + file_extension 
    writer = ArVideoWriter(result_tmp_name, video_info.fps, crf)

    # --- build a single MatAnyOne2 model wrapped in DataParallel for multi-GPU inference ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MatAnyone.from_pretrained("PeiqingYang/MatAnyone")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device).eval()
    proc = InferenceCore(model, cfg=model.cfg)
    processors = [proc]
    device_list = [device]
    num_processors = 1

    # create a full‑frame initial mask (L+R) resized to the video resolution
    maskL_np  = maskL.cpu().numpy() if torch.is_tensor(maskL) else np.array(maskL)
    maskR_np  = maskR.cpu().numpy() if torch.is_tensor(maskR) else np.array(maskR)
    init_mask = cv2.hconcat([maskL_np, maskR_np])
    init_mask = cv2.resize(init_mask, (video_info.width, video_info.height), interpolation=cv2.INTER_NEAREST)

    while ffmpeg.isOpen():
        img = ffmpeg.read()
        if img is None:
            break
        current_frame += 1

        tiles, coords = split_into_tiles(img, TILE_ROWS, TILE_COLS, TILE_OVERLAP)
        full_mask_alpha = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for tile_idx, (tile_img, coord) in enumerate(zip(tiles, coords)):
            proc   = processors[0]
            device = device_list[0]

            tile_tensor = prepare_frame(tile_img, device=device)

            # inject initial or helper masks at configured keyframes
            if current_frame == 1 or current_frame in helper_masks_local:
                # override init_mask if at a helper frame
                if current_frame in helper_masks_local:
                    init_mask = helper_masks_local[current_frame]
                y0, y1, x0, x1 = coord
                tile_init = torch.from_numpy(init_mask[y0:y1, x0:x1]).to(device)
                prob = proc.step(tile_tensor, tile_init, objects=objects)
            else:
                prob = proc.step(tile_tensor)

            tile_mask = (proc.output_prob_to_mask(prob).cpu().numpy() * 255).astype(np.uint8)
            blend_tile(full_mask_alpha, tile_mask, coord)

        if erode:
            full_mask_alpha = cv2.erode(full_mask_alpha, (3,3), iterations=1)

        writer.add_frame(img, full_mask_alpha)
        WORKER_STATUS = f"Create Mask {current_frame}/{video_info.length}"
        gc.collect()

    for p in processors:
        del p

    if torch.torch.cuda.is_available():
        torch.cuda.empty_cache()

    ffmpeg.stop()
    writer.finalize()

    total_frames = current_frame
    while not writer.is_finished():
        WORKER_STATUS = f"Encode Frame {writer.get_current_frame_number()}/{total_frames}"
        time.sleep(0.5)

    gc.collect()

    WORKER_STATUS = f"Combine Video and Audio..." 
    subprocess.run([
        "ffmpeg",
        "-i", result_tmp_name,
        "-i", video,
        "-c", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        result_name
    ])

    os.remove(result_tmp_name)

    WORKER_STATUS = f"Convertion completed" 
    return result_name


job_id = 0
result_list = []

def background_worker():
    global file_list
    global WORKER_STATUS
    surplus_url = os.environ.get('JOB_SURPLUS_CHECK_URL')
    while True:
        if surplus_url:
            try:
                start_job = "True" in str(requests.get(surplus_url).json())
            except Exception as ex:
                start_job = False
                print(ex)

            if not start_job:
                WORKER_STATUS = "Wait for surplus"
                time.sleep(30)
                continue

        pkl = [x for x in glob.glob("/jobs/*.pkl")]

        if len(pkl) == 0:
            time.sleep(2)
            continue

        pkl = sorted(pkl)[0]
        time.sleep(2) # ensure file is fully written
        with open(pkl, 'rb') as f:
            job = pickle.load(f)

        print("Start job", job['id'])
        result = process(job['id'], job['video'], job['projection'], job['maskL'], job['maskR'], job['crf'], job['erode'])
        if result is not None:
            result_list.append(result)
            if filebrowser_host := os.environ.get('FILEBROWSER_HOST'):
                WORKER_STATUS = "Uploading..."
                if filebrowser_user := os.environ.get('FILEBROWSER_USER'):
                    client = FilebrowserClient(filebrowser_host, username=filebrowser_user, password=os.environ.get('FILEBROWSER_PASSWORD'), insecure=True)
                else:
                    client = FilebrowserClient(filebrowser_host, insecure=True)

                asyncio.run(client.connect())
                couroutine = client.upload(
                    local_path=result,
                    remote_path=os.environ.get('FILEBROWSER_PATH', os.path.basename(result)).replace("{filename}", os.path.basename(result)),
                    override=False,
                    concurrent=1,
                )
                asyncio.run(couroutine)


        os.remove(job['video'])
        os.remove(pkl)
        time.sleep(1)
        WORKER_STATUS = "Idle"

def add_job(video, projection, maskL, maskR, crf, erode):
    global job_id

    if video is None:
        raise gr.Error("video missing", duration=3)

    if maskL is None: 
        raise gr.Error("maskL not set", duration=3)

    if maskR is None:
        raise gr.Error("maskR not set", duration=3)

    if Image.fromarray(maskL).convert("L").getextrema() == (0, 0):
        raise gr.Error("maskL is empty", duration=3)

    if Image.fromarray(maskR).convert("L").getextrema() == (0, 0):
        raise gr.Error("maskR is empty", duration=3)

    job_id += 1
    print("Add job", job_id)

    ts = str(int(time.time()))

    dest = '/jobs/' + ts + os.path.basename(video.name)
    shutil.move(video.name, dest)

    job_data = {
        'id': job_id,
        'video': dest,
        'projection': projection,
        'crf': crf,
        'maskL': Image.fromarray(maskL).convert('L'),
        'maskR': Image.fromarray(maskR).convert('L'),
        'erode': erode
    }

    with open(f"/jobs/{ts}.pkl", "wb") as f:
        pickle.dump(job_data, f)

    return None, None, None, None, None, None, None, None, None, None, None, None, None

def status_text():
    pending_jobs = len([x for x in glob.glob("/jobs/*.pkl")])
    return "Worker Status: " + WORKER_STATUS + "\n" \
        + f"Pending Jobs: {pending_jobs}"

current_origin_frame = {
    "L": None,
    "R": None
}

colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]

def add_mark(frame):
    if frame is None:
        return None
    result = frame.frame_data.copy()
    marker_size = 25
    marker_thickness = 3
    marker_default_width = 1200
    width = result.shape[0]
    ratio = width / marker_default_width
    marker_final_size = int(marker_size * ratio)
    if marker_final_size < 3:
        marker_final_size = 3
    marker_final_thickness = int(marker_thickness * ratio)
    if marker_final_thickness < 2:
        marker_final_thickness = 2
    for (x, y, label) in frame.point_set:
        cv2.drawMarker(result, (x, y), colors[label], markerType=markers[label], markerSize=marker_final_size, thickness=marker_final_thickness)
    return result

def get_frame(video, projection):
    if str(projection) == "eq":
        filter_complex = "[0:v]split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; [right]crop=ih:ih:ih:0[right_crop]; [left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; [right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; [leftfisheye][rightfisheye]hstack[v]"
        frame = FFmpegStream.get_frame(video.name, 0, filter_complex)
        width = 2*MASK_SIZE
        frame = cv2.resize(frame, (int(width), int(width/2)))
        frameL = frame[:, :int(width/2)]
        frameR = frame[:, int(width/2):]
    else:
        frame = FFmpegStream.get_frame(video.name, 0)
        width = 2*MASK_SIZE
        frame = cv2.resize(frame, (int(width), int(width/2)))
        frameL = frame[:, :int(width/2)]
        frameR = frame[:, int(width/2):]

    frameL = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)
    frameR = cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB)

    current_origin_frame['L'] = ImageFrame(frameL, 0)
    current_origin_frame['R'] = ImageFrame(frameR, 0)

    return Image.fromarray(frameL), Image.fromarray(frameR), Image.fromarray(frameL), Image.fromarray(frameR), Image.fromarray(np.zeros([MASK_SIZE, MASK_SIZE, 3], dtype=np.uint8)).convert("L"), Image.fromarray(np.zeros([MASK_SIZE, MASK_SIZE, 3], dtype=np.uint8)).convert("L")

def get_mask(frameL, frameR, maskLPrompt, maskRPrompt, maskLThreshold, maskRThreshold, maskLNegativePrompt, maskRNegativePrompt):
    if frameL is None or frameR is None:
        return None, None, None, None

    sam2 = GroundingDinoSAM2Segment()

    if len( maskLPrompt) > 0: 
        (_, imgLMask) = sam2.predict([frameL], maskLThreshold, maskLPrompt)
        maskL = (imgLMask[0].squeeze().cpu().numpy() * 255).astype(np.uint8)
        if len( maskLNegativePrompt) > 0: 
            (_, negativeImgLMask) = sam2.predict([frameL], maskLThreshold, maskLNegativePrompt)
            invNegativeMaskL = 255 - (negativeImgLMask[0].squeeze().cpu().numpy() * 255).astype(np.uint8)
            maskL = np.bitwise_and(maskL, invNegativeMaskL)

        maskL = Image.fromarray(maskL, mode='L')

        previewL = Image.composite(
            Image.new("RGB", maskL.size, "blue"),
            Image.fromarray(frameL).convert("RGBA"),
            maskL.point(lambda p: 100 if p > 1 else 0)
        )
    else:
        previewL = frameL
        maskL = Image.fromarray(np.zeros([MASK_SIZE, MASK_SIZE, 3], dtype=np.uint8)).convert("L")

    if len( maskRPrompt) > 0: 
        (_, imgRMask) = sam2.predict([frameR], maskRThreshold, maskRPrompt)

        maskR = (imgRMask[0].squeeze().cpu().numpy() * 255).astype(np.uint8)

        if len( maskRNegativePrompt) > 0: 
            (_, negativeImgRMask) = sam2.predict([frameR], maskRThreshold, maskRNegativePrompt)
            invNegativeMaskR = 255 - (negativeImgRMask[0].squeeze().cpu().numpy() * 255).astype(np.uint8)
            maskR = np.bitwise_and(maskR, invNegativeMaskR)

        maskR = Image.fromarray(maskR, mode='L')
        previewR = Image.composite(
                Image.new("RGB", maskR.size, "blue"),
                Image.fromarray(frameR).convert("RGBA"),
                maskR.point(lambda p: 100 if p > 1 else 0)
            )
    else:
        previewR = frameR
        maskR = Image.fromarray(np.zeros([MASK_SIZE, MASK_SIZE, 3], dtype=np.uint8)).convert("L")

    del sam2
    
    return previewL, maskL, previewR, maskR

def get_mask2():
    global current_origin_frame

    if current_origin_frame['L'] is None or current_origin_frame['R'] is None:
        return None, None, None, None

    frameL = current_origin_frame['L'].frame_data
    frameR = current_origin_frame['R'].frame_data

    pointsL = []
    for x, y, label in current_origin_frame['L'].point_set:
        pointsL.append(((x, y), label))

    pointsR = []
    for x, y, label in current_origin_frame['R'].point_set:
        pointsR.append(((x, y), label))

    sam2 = SAM2PointSegment()

    if len(pointsL) != 0:
        (_, imgLMask) = sam2.predict(frameL, pointsL)
        maskL = Image.fromarray((imgLMask * 255).astype(np.uint8), mode='L')
        previewL = Image.composite(
            Image.new("RGB", maskL.size, "blue"),
            Image.fromarray(frameL).convert("RGBA"),
            maskL.point(lambda p: 100 if p > 1 else 0)
        )
    else:
        previewL = frameL
        maskL = Image.fromarray(np.zeros([MASK_SIZE, MASK_SIZE, 3], dtype=np.uint8)).convert("L") 


    if len(pointsR) != 0:
        (_, imgRMask) = sam2.predict(frameR, pointsR)
    
        maskR = Image.fromarray((imgRMask * 255).astype(np.uint8), mode='L')

        previewR = Image.composite(
            Image.new("RGB", maskR.size, "blue"),
            Image.fromarray(frameR).convert("RGBA"),
            maskR.point(lambda p: 100 if p > 1 else 0)
        )
    else:
        previewR = frameR
        maskR = Image.fromarray(np.zeros([MASK_SIZE, MASK_SIZE, 3], dtype=np.uint8)).convert("L") 

    del sam2

    return previewL, maskL, previewR, maskR

def merge_add_mask(maskL, maskR, mergedMaskL, mergedMaskR):
    if maskL is not None and mergedMaskL is not None:
        mergedMaskL = np.bitwise_or(np.array(mergedMaskL), np.array(maskL))
        mergedMaskL = Image.fromarray(mergedMaskL).convert("L")
    if maskR is not None and mergedMaskR is not None:
        mergedMaskR = np.bitwise_or(np.array(mergedMaskR), np.array(maskR))
        mergedMaskR = Image.fromarray(mergedMaskR).convert("L")
    return  mergedMaskL, mergedMaskR

def merge_subtract_mask(maskL, maskR, mergedMaskL, mergedMaskR):
    if maskL is not None and mergedMaskL is not None:
        invMaskL = 255 - np.array(maskL)
        mergedMaskL = np.bitwise_and(np.array(mergedMaskL), invMaskL)
        mergedMaskL = Image.fromarray(mergedMaskL).convert("L")
    if maskR is not None and mergedMaskR is not None:
        invMaskR = 255 - np.array(maskR)
        mergedMaskR = np.bitwise_and(np.array(mergedMaskR), invMaskR)
        mergedMaskR = Image.fromarray(mergedMaskR).convert("L")
    return  mergedMaskL, mergedMaskR

def set_mask_size(x):
    global MASK_SIZE
    MASK_SIZE = int(x)

def update_maskL_preview(maskL):
    if maskL is None:
        return None

    frameL = current_origin_frame['L'].frame_data
    if frameL is None:
        return None

    maskL = Image.fromarray(maskL).convert("L")
    previewL = Image.composite(
        Image.new("RGB", maskL.size, "blue"),
        Image.fromarray(frameL).convert("RGBA"),
        maskL.point(lambda p: 100 if p > 1 else 0)
    )

    return previewL

def update_maskR_preview(maskR):
    if maskR is None:
        return None

    frameR = current_origin_frame['R'].frame_data
    if frameR is None:
        return None

    maskR = Image.fromarray(maskR).convert("L")
    previewR = Image.composite(
        Image.new("RGB", maskR.size, "blue"),
        Image.fromarray(frameR).convert("RGBA"),
        maskR.point(lambda p: 100 if p > 1 else 0)
    )

    return previewR

with gr.Blocks() as demo:
    gr.Markdown("# Video VR2AR Converter")
    gr.Markdown('''
        Process:
        1. Upload Your video
        2. Select Video Source Format
        3. Select Mask Size (Higher value require more VRAM!)
        3. Extract first projection Frame
        4. Generate Initial Mask with Button. Use Points or Prompts to generate individual Masks and merge them with add or subtract button to the initial mask. To create a second mask with points you have to use the crea button to select the points for the next partial mask. Use the foreground and subtract to remove areas from mask. use foreground and add to add areas to the inital mask. You can also try to specify additional backrground points but for me this always results in worse results.
        5. Add Video Mask Job
    ''')
    with gr.Column():
        gr.Markdown("## Stage 1 - Video")
        input_video = gr.File(label="Upload Video (MKV or MP4)", file_types=["mkv", "mp4", "video"])
        projection_dropdown = gr.Dropdown(choices=["eq", "fisheye180", "fisheye190", "fisheye200"], label="VR Video Source Format", value="eq")
        mask_size = gr.Number(
            label="Mask Size (Mask Size larger than 40% of video height make no sense, higher mask value require more VRAM, 1440 require approx. 20GB VRAM)",
            minimum=512,
            maximum=2048,
            step=1,
            value=MASK_SIZE
        )
        mask_size.change(
            fn=set_mask_size,
            inputs=mask_size
        )

    with gr.Column():
        gr.Markdown("## Stage 2 - Extract First Frame")
        frame_button = gr.Button("Extract Projection Frame")
        gr.Markdown("### Result")
        with gr.Row():
            framePreviewL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
            framePreviewR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')

    gr.Markdown("## Stage 3 - Generate Initial Mask")
    with gr.Tabs():
        with gr.Tab("Prompt"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        maskLPrompt = gr.Textbox(label="Positive Prompt", value="top person", lines=1, max_lines=1)
                        maskLNegativePrompt = gr.Textbox(label="Negative Prompt", value="", lines=1, max_lines=1)
                        maskLThreshold = gr.Number(
                            label="Threshold",
                            minimum=0.01,
                            maximum=0.99,
                            step=0.01,
                            value=0.3
                        )
                    with gr.Column():
                        maskRPrompt = gr.Textbox(label="Positive Prompt", value="top person", lines=1, max_lines=1)
                        maskRNegativePrompt = gr.Textbox(label="Negative Prompt", value="", lines=1, max_lines=1)
                        maskRThreshold = gr.Number(
                            label="Threshold",
                            minimum=0.01,
                            maximum=0.99,
                            step=0.01,
                            value=0.3
                        )
                mask_button = gr.Button("Generate Mask")
        with gr.Tab("Points"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        undoL = gr.Button('Undo')
                        removeL = gr.Button('Clear')
                    with gr.Column():
                        maskSelectionL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
                        groupL = gr.Radio(['foreground', 'background'], label='Object Type', value='foreground')
                with gr.Column():
                    with gr.Row():
                        undoR = gr.Button('Undo')
                        removeR = gr.Button('Clear')
                    with gr.Column():
                        maskSelectionR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
                        groupR = gr.Radio(['foreground', 'background'], label='Object Type', value='foreground')

            def add_mark_point(t, point_type, event: gr.SelectData):
                global current_origin_frame
                label = 1
                if point_type == 'foreground':
                    label = 1
                elif point_type == 'background':
                    label = 0
                
                if current_origin_frame[t] is None:
                    return None
                
                current_origin_frame[t].add(*event.index, label)
                return add_mark(current_origin_frame[t])

            def add_mark_point_l(point_type, event: gr.SelectData):
                return add_mark_point('L', point_type, event)
            
            def add_mark_point_r(point_type, event: gr.SelectData):
                return add_mark_point('R', point_type, event)
            
            def undo_last_point_l():
                global current_origin_frame
                if current_origin_frame['L'] is None:
                    return None
                current_origin_frame['L'].pop()
                return add_mark(current_origin_frame['L'])

            def undo_last_point_r():
                global current_origin_frame
                if current_origin_frame['R'] is None:
                    return None
                current_origin_frame['R'].pop()
                return add_mark(current_origin_frame['R'])

            def remove_all_points_l():
                global current_origin_frame
                if current_origin_frame['L'] is None:
                    return None
                current_origin_frame['L'].clear()
                return current_origin_frame['L'].frame_data
            
            def remove_all_points_r():
                global current_origin_frame
                if current_origin_frame['R'] is None:
                    return None
                current_origin_frame['R'].clear()
                return current_origin_frame['R'].frame_data

            gr.Image.select(maskSelectionL, add_mark_point_l, inputs = [groupL], outputs = [maskSelectionL])
            gr.Image.select(maskSelectionR, add_mark_point_r, inputs = [groupR], outputs = [maskSelectionR])

            gr.Button.click(undoL, undo_last_point_l, inputs=None, outputs=maskSelectionL)
            gr.Button.click(undoR, undo_last_point_r, inputs=None, outputs=maskSelectionR)
            gr.Button.click(removeL, remove_all_points_l, inputs=None, outputs=maskSelectionL)
            gr.Button.click(removeR, remove_all_points_r, inputs=None, outputs=maskSelectionR)
            mask2_button = gr.Button("Generate Mask")




    with gr.Column():
        gr.Markdown("### Mask Step 1")
        gr.Markdown("Preview")
        with gr.Row():
            maskPreviewL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
            maskPreviewR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
        gr.Markdown("Mask")
        with gr.Row():
            maskL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
            maskR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
    with gr.Column():
        gr.Markdown("### Mask Step 2")
        gr.Markdown("Add or subtract mask from mask step 1 to the frame initial Mask")
        with gr.Row():
            mask_add_button = gr.Button("Add Mask")
            mask_subtract_button = gr.Button("Subtract Mask")

        gr.Markdown("Combinend Mask")
        with gr.Row():
            mergedMaskL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
            mergedMaskR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')

        gr.Markdown("Preview")
        with gr.Row():
            previewMergedMaskL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
            previewMergedMaskR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')

        mergedMaskL.change(
            fn=update_maskL_preview, 
            inputs=mergedMaskL, 
            outputs=previewMergedMaskL
        )

        mergedMaskR.change(
            fn=update_maskR_preview, 
            inputs=mergedMaskR, 
            outputs=previewMergedMaskR
        )

        frame_button.click(
            fn=get_frame,
            inputs=[input_video, projection_dropdown],
            outputs=[framePreviewL, framePreviewR, maskSelectionL, maskSelectionR, mergedMaskL, mergedMaskR]
        )

        mask_button.click(
            fn=get_mask,
            inputs=[framePreviewL, framePreviewR, maskLPrompt, maskRPrompt, maskLThreshold, maskRThreshold, maskLNegativePrompt, maskRNegativePrompt],
            outputs=[maskPreviewL, maskL, maskPreviewR, maskR]
        )

        mask2_button.click(
            fn=get_mask2,
            inputs=None,
            outputs=[maskPreviewL, maskL, maskPreviewR, maskR]
        )

        mask_add_button.click(
            fn=merge_add_mask,
            inputs=[maskL, maskR, mergedMaskL, mergedMaskR],
            outputs=[mergedMaskL, mergedMaskR]
        )

        mask_subtract_button.click(
            fn=merge_subtract_mask,
            inputs=[maskL, maskR, mergedMaskL, mergedMaskR],
            outputs=[mergedMaskL, mergedMaskR]
        )

    with gr.Column():
        gr.Markdown("## Stage 4 - Add Job")
        crf_dropdown = gr.Dropdown(choices=[16,17,18,19,20,21,22], label="Encode CRF", value=16)
        erode_checkbox = gr.Checkbox(label="Erode Mask Output", value=False, info="")
        add_button = gr.Button("Add Job")
        add_button.click(
            fn=add_job,
            inputs=[input_video, projection_dropdown, mergedMaskL, mergedMaskR, crf_dropdown, erode_checkbox],
            outputs=[input_video, framePreviewL, framePreviewR, maskPreviewL, mergedMaskL, maskPreviewR, mergedMaskR, maskL, maskR, maskSelectionL, maskSelectionR, previewMergedMaskL, previewMergedMaskR ]
        )

    with gr.Column():
        gr.Markdown("## Job Results")
        status = gr.Textbox(label="Status", lines=2)
        output_videos = gr.File(value=[], label="Download AR Videos", visible=True)
        restart_button = gr.Button("CLEANUP AND RESTART".upper())
        restart_button.click(
            # dirty hack, we use k8s restart pod
            fn=lambda: os.system("pkill python"),
            inputs=[],
            outputs=[]
        )

    timer1 = gr.Timer(2, active=True)
    timer5 = gr.Timer(5, active=True)
    timer1.tick(status_text, outputs=status)
    timer5.tick(lambda: result_list, outputs=output_videos)
    demo.load(fn=status_text, outputs=status)
    demo.load(fn=lambda: result_list, outputs=output_videos)


# Only launch the Gradio app when run as a script
if __name__ == "__main__":
    demo.queue(concurrency_count=1)
    demo.launch()



# --- Distributed/queue mode functions ---
def launch_ui(redis_url, s3_bucket):
    r = redis.Redis.from_url(redis_url)
    s3 = boto3.client("s3")
    def enqueue_job(video, projection, maskL, maskR, crf, erode):
        # upload video
        vid_key = f"inputs/{int(time.time())}_{os.path.basename(video.name)}"
        s3.upload_fileobj(video, s3_bucket, vid_key)
        # serialize and upload maskL
        mask_key = vid_key.replace("inputs/", "masks/").rsplit('.',1)[0] + "_L.png"
        buf = io.BytesIO()
        maskL.save(buf, format="PNG")
        buf.seek(0)
        s3.upload_fileobj(buf, s3_bucket, mask_key)
        # serialize and upload maskR
        mask_keyR = vid_key.replace("inputs/", "masks/").rsplit('.',1)[0] + "_R.png"
        buf2 = io.BytesIO()
        maskR.save(buf2, format="PNG")
        buf2.seek(0)
        s3.upload_fileobj(buf2, s3_bucket, mask_keyR)
        # enqueue JSON job
        job = {
            "video_key": vid_key,
            "maskL_key": mask_key,
            "maskR_key": mask_keyR,
            "projection": projection,
            "crf": crf,
            "erode": erode
        }
        r.lpush("video_jobs", json.dumps(job))
        return None, None, None, None, None, None, None, None, None, None, None, None, None
    # rebind add_button to enqueue_job
    add_button.click(
        fn=enqueue_job,
        inputs=[input_video, projection_dropdown, maskL, maskR, crf_dropdown, erode_checkbox],
        outputs=[input_video, framePreviewL, framePreviewR, maskPreviewL, maskL, maskPreviewR, maskR]
    )