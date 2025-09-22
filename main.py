import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import traceback
import json
import gradio as gr
import subprocess
import tempfile
import shutil
import pickle
import torch
import cv2
import gc
import glob
import time
import threading
import queue
import urllib3
import asyncio
import subprocess
import requests
import random
from pathlib import Path
from typing import List, Optional
from sam2_executor import GroundingDinoSAM2Segment, SAM2PointSegment
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import torch.nn.functional as F
import numpy as np

from matanyone.model.matanyone import MatAnyone
from matanyone.inference.inference_core import InferenceCore

from data.ffmpegstream import FFmpegStream
from data.ArVideoWriter import ArVideoWriter
from video_process import ImageFrame
import processing_core
from filebrowser_client import FilebrowserClient

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

WORKER_STATUS = "Idle"
MASK_SIZE = 1440
SECONDS = 10
WARMUP = 4
JOB_VERSION = 3
SSIM_THRESHOLD = 0.983
DEBUG = False
MASK_DEBUG = False
SURPLUS_IGNORE = False
SCHEDULE = bool(os.environ.get('EXECUTE_SCHEDULER_ON_START', "True"))
RESULT_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.webm'}
COMPLETED_JOB_DIR = Path('/jobs/completed')
PENDING_JOBS_DIR = Path('/jobs/pending')
WORKERS_DIR = Path('/jobs/workers')
FAILED_JOB_DIR = Path('/jobs/failed')


def _ensure_base_dirs() -> None:
    for directory in (COMPLETED_JOB_DIR, PENDING_JOBS_DIR, WORKERS_DIR, FAILED_JOB_DIR):
        directory.mkdir(parents=True, exist_ok=True)


_ensure_base_dirs()

def _set_worker_status(message: str) -> None:
    global WORKER_STATUS
    WORKER_STATUS = message


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

def prepare_frame(frame, has_cuda=True):
    vframes = torch.from_numpy(frame)
        
    if vframes.shape[-1] == 3:
        vframes = vframes.permute(2, 0, 1)
    
    if has_cuda:
         vframes =  vframes.cuda()

    image_input = vframes.float() / 255.0

    return image_input

def prepare_mask(mask):
    mask = np.array(mask)
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)

    return mask

def finalize_mask(mask):
    mask = torch.from_numpy(mask)
    if torch.torch.cuda.is_available():
        mask = mask.cuda()

    return mask

def fix_mask2(mask):
    mask = np.array(mask)
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)
    mask = torch.from_numpy(mask)
    if torch.torch.cuda.is_available():
        mask = mask.cuda()

    return mask

@torch.no_grad()
def process(video, projection, masks, crf = 16, erode = False, force_init_mask=False, job_id: str | None = None):
    helpers = {
        'prepare_frame': prepare_frame,
        'fix_mask2': fix_mask2,
        'set_status': _set_worker_status,
    }
    return processing_core.process_video(
        video=video,
        projection=projection,
        masks=masks,
        crf=crf,
        erode=erode,
        force_init_mask=force_init_mask,
        reverse_tracking=False,
        helpers=helpers,
        warmup=WARMUP,
        ssim_threshold=SSIM_THRESHOLD,
        job_id=job_id,
        job_version=JOB_VERSION,
    )

@torch.no_grad()
def process_with_reverse_tracking(video, projection, masks, crf = 16, erode = False, force_init_mask=False, job_id: Optional[str] = None):
    global WORKER_STATUS

    maskIdx = 0
    mask_w, mask_h = masks[maskIdx]['maskL'].size

    original_filename = os.path.basename(video)
    file_name, file_extension = os.path.splitext(original_filename)
    
    video_info = FFmpegStream.get_video_info(video)
    
    reader_config = {
        "parameter": {
            "width": 2*mask_w,
            "height": mask_h,
        }
    }
    
    output_w = 2*mask_w
    if "eq" == projection:
        reader_config["filter_complex"] = f"[0:v]split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; [right]crop=ih:ih:ih:0[right_crop]; [left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; [right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; [leftfisheye][rightfisheye]hstack,scale={output_w}:{mask_h}[v]"
        projection_out = "fisheye180"
    else:
        projection_out = projection
        reader_config["video_filter"] = f"scale={output_w}:{mask_h}"

    WORKER_STATUS = f"Load Models to create Masks"

    current_frame = 0
    objects = [1]

    has_cuda = torch.torch.cuda.is_available()

    ffmpeg = FFmpegStream(
        video_path = video,
        config = reader_config,
        skip_frames = 0,
        watchdog_timeout_in_seconds = 0 # we can not use wd here
    )

    result_name = file_name + "_" + str(projection_out).upper() + "_alpha" + file_extension

    matanyone1 = MatAnyone.from_pretrained("PeiqingYang/MatAnyone")
    if torch.torch.cuda.is_available():
        matanyone1 = matanyone1.cuda()
    matanyone1 = matanyone1.eval()
    processor1 = InferenceCore(matanyone1, cfg=matanyone1.cfg)

    matanyone2 = MatAnyone.from_pretrained("PeiqingYang/MatAnyone")
    if torch.torch.cuda.is_available():
        matanyone2 = matanyone2.cuda()
    matanyone2 = matanyone2.eval()
    processor2 = InferenceCore(matanyone2, cfg=matanyone2.cfg)

    for i in range(len(masks)):
        imgLV = prepare_frame(masks[i]['frameL'], has_cuda)
        imgRV = prepare_frame(masks[i]['frameR'], has_cuda)
        imgLMask = fix_mask2(masks[i]['maskL'])
        imgRMask = fix_mask2(masks[i]['maskR'])
        _ = processor1.step(imgLV, imgLMask, objects=objects, force_permanent=True)
        _ = processor2.step(imgRV, imgRMask, objects=objects, force_permanent=True)

    if os.path.exists("process/frames"):
        shutil.rmtree("process/frames")

    if os.path.exists("process/masks"):
        shutil.rmtree("process/masks")

    if os.path.exists("process/debug"):
        shutil.rmtree("process/debug")

    os.makedirs("process", exist_ok=True)
    os.makedirs("process/frames", exist_ok=True)
    os.makedirs("process/masks", exist_ok=True)
    os.makedirs("process/debug", exist_ok=True)
    reverse_track = False

    while ffmpeg.isOpen():
        img = ffmpeg.read()
        if img is None:
            break
        current_frame += 1

        img_scaled = img

        _, width = img_scaled.shape[:2]
        imgL = img_scaled[:, :int(width/2)]
        imgR = img_scaled[:, int(width/2):]

        frame_match = False
        if force_init_mask and current_frame == 1:
            frame_match = True

        if maskIdx < len(masks):
            s1 = ssim(masks[maskIdx]['frameLGray'], cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY))
            if DEBUG:
                print("ssim1", s1)
            if s1 > SSIM_THRESHOLD:
                s2 = ssim(masks[maskIdx]['frameRGray'], cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY))
                if not DEBUG:
                    print("ssim1", s1)
                print("ssim2", s2)
                if s2 > SSIM_THRESHOLD:
                    frame_match = True

        imgLV = prepare_frame(imgL, has_cuda)
        imgRV = prepare_frame(imgR, has_cuda)

        if frame_match:
            print("match at", current_frame)
            imgLMask = fix_mask2(masks[maskIdx]['maskL'])
            imgRMask = fix_mask2(masks[maskIdx]['maskR'])
            maskIdx += 1
            reverse_track = True

            output_prob_L = processor1.step(imgLV, imgLMask, objects=objects)
            output_prob_R = processor2.step(imgRV, imgRMask, objects=objects)
 
            for _ in range(WARMUP):
                output_prob_L = processor1.step(imgLV, first_frame_pred=maskIdx==1)
                output_prob_R = processor2.step(imgRV, first_frame_pred=maskIdx==1)
        elif maskIdx > 0:
            output_prob_L = processor1.step(imgLV)
            output_prob_R = processor2.step(imgRV)
        else:
            print("Warning: Start frame not found yet")
            continue

        WORKER_STATUS = f"Create Mask {current_frame}/{video_info.length}"

        mask_output_L = processor1.output_prob_to_mask(output_prob_L)
        mask_output_R = processor2.output_prob_to_mask(output_prob_R)

        mask_output_L_pha = mask_output_L.unsqueeze(2).cpu().detach().numpy()
        mask_output_R_pha = mask_output_R.unsqueeze(2).cpu().detach().numpy()

        mask_output_L_pha = (mask_output_L_pha*255).astype(np.uint8)
        mask_output_R_pha = (mask_output_R_pha*255).astype(np.uint8)

        combined_mask = cv2.hconcat([mask_output_L_pha, mask_output_R_pha])

        if maskIdx < len(masks):
            cv2.imwrite('process/frames/' + str(current_frame).zfill(6) + ".png", img_scaled)
        
        cv2.imwrite('process/masks/' + str(current_frame).zfill(6) + ".png", combined_mask)
        if MASK_DEBUG:
            cv2.imwrite('process/debug/' + str(current_frame).zfill(6) + ".png", combined_mask)

        if reverse_track:
            imgLV_end = imgLV
            imgRV_end = imgRV
            reverse_track = False
            frame_files = sorted(['process/frames/' + f for f in os.listdir('process/frames/') if f.endswith(".png")],  reverse=True)
            subprocess_len = len(frame_files)
            for idx, frame_file in enumerate(frame_files):
                img_scaled = cv2.imread(frame_file)
                os.remove(frame_file)

                _, width = img_scaled.shape[:2]
                imgL = img_scaled[:, :int(width/2)]
                imgR = img_scaled[:, int(width/2):]

                imgLV = prepare_frame(imgL, has_cuda)
                imgRV = prepare_frame(imgR, has_cuda)

                output_prob_L = processor1.step(imgLV)
                output_prob_R = processor2.step(imgRV)
                WORKER_STATUS = f"Create Mask {current_frame}/{video_info.length} - Subprocess {idx}/{subprocess_len}"

                mask_output_L = processor1.output_prob_to_mask(output_prob_L)
                mask_output_R = processor2.output_prob_to_mask(output_prob_R)

                mask_output_L_pha = mask_output_L.unsqueeze(2).cpu().detach().numpy()
                mask_output_R_pha = mask_output_R.unsqueeze(2).cpu().detach().numpy()

                mask_output_L_pha = (mask_output_L_pha*255).astype(np.uint8)
                mask_output_R_pha = (mask_output_R_pha*255).astype(np.uint8)

                combined_mask = cv2.hconcat([mask_output_L_pha, mask_output_R_pha])
                maskA = cv2.imread(frame_file.replace('frames', 'masks'), cv2.IMREAD_UNCHANGED)
                
                if MASK_DEBUG:
                    cv2.imwrite(frame_file.replace('frames', 'debug').replace('.png', '_rev.png'), combined_mask)
                
                # using avg or other merge gives much worse reults at edges
                mergedA = np.bitwise_or(np.array(maskA), np.array(combined_mask))

                cv2.imwrite(frame_file.replace('frames', 'masks'), mergedA)
                if MASK_DEBUG:
                    cv2.imwrite(frame_file.replace('frames', 'debug').replace('.png', '_res.png'), mergedA)

            print("reverse tracking of", subprocess_len, "completed")
            # set model state to forware tracking again
            imgLMask = fix_mask2(masks[maskIdx-1]['maskL'])
            imgRMask = fix_mask2(masks[maskIdx-1]['maskR'])
            output_prob_L = processor1.step(imgLV_end, imgLMask, objects=objects)
            output_prob_R = processor2.step(imgRV_end, imgRMask, objects=objects)
            for _ in range(WARMUP):
                output_prob_L = processor1.step(imgLV_end)
                output_prob_R = processor2.step(imgRV_end)
        
        gc.collect()

    shutil.rmtree("process/frames")
    if maskIdx < len(masks):
        print("ERROR: not all frames found in video!")

    del processor1
    del processor2
    del matanyone1
    del matanyone2

    if torch.torch.cuda.is_available():
        torch.cuda.empty_cache()

    ffmpeg.stop()

    gc.collect()

    WORKER_STATUS = f"Create Mask Video..."
    print("create Video", result_name)
    scale = video_info.height / mask_h * 0.4

    fc2 = f'"[1]scale=iw*{scale}:-1[alpha];[2][alpha]scale2ref[mask][alpha];[alpha][mask]alphamerge,split=2[masked_alpha1][masked_alpha2]; [masked_alpha1]crop=iw/2:ih:0:0,split=2[masked_alpha_l1][masked_alpha_l2]; [masked_alpha2]crop=iw/2:ih:iw/2:0,split=4[masked_alpha_r1][masked_alpha_r2][masked_alpha_r3][masked_alpha_r4]; [0][masked_alpha_l1]overlay=W*0.5-w*0.5:-0.5*h[out_lt];[out_lt][masked_alpha_l2]overlay=W*0.5-w*0.5:H-0.5*h[out_tb]; [out_tb][masked_alpha_r1]overlay=0-w*0.5:-0.5*h[out_l_lt];[out_l_lt][masked_alpha_r2]overlay=0-w*0.5:H-0.5*h[out_tb_ltb]; [out_tb_ltb][masked_alpha_r3]overlay=W-w*0.5:-0.5*h[out_r_lt];[out_r_lt][masked_alpha_r4]overlay=W-w*0.5:H-0.5*h"'


    cmd = [
        "ffmpeg",
        '-hide_banner',
        '-loglevel', 'warning',
        '-thread_queue_size', '64',
        '-ss', FFmpegStream.frame_to_timestamp(0, video_info.fps),
        '-hwaccel', 'auto',
        '-i', "\""+str(video)+"\"",
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vsync', 'passthrough',
        '-vcodec', 'rawvideo',
        '-an',
        '-sn'
    ]

    if "eq" == projection:
        cmd += [
            "-filter_complex", "\"[0:v]split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; [right]crop=ih:ih:ih:0[right_crop]; [left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; [right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; [leftfisheye][rightfisheye]hstack[v]\"",
            "-map", "[v]"
        ]

    cmd += [
        '-',
        '|',
        "ffmpeg",
        '-y',
        '-hide_banner',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{video_info.width}x{video_info.height}',
        '-r', str(video_info.fps),
        '-thread_queue_size', '64',
        '-i', 'pipe:0',
        '-r', str(video_info.fps),
        '-thread_queue_size', '64',
        "-i", "\"process/masks/%06d.png\"",
        "-i", "mask.png",
        '-r', str(video_info.fps),
        '-i', "\""+str(video)+"\"",
        "-filter_complex",
        fc2,
        "-c:v", "libx265", 
        "-crf", str(crf),
        "-preset", "veryfast",
        "-map", "\"3:a:?\"",
        "-c:a", "copy",
        "\""+result_name+"\"",
        "-y"
    ]

    if DEBUG:
        print(cmd)

    subprocess.run(' '.join(cmd), shell=True)

    shutil.rmtree("process/masks")

    WORKER_STATUS = f"Convertion completed" 
    return result_name

result_list: List[str] = []


def _refresh_result_list() -> None:
    global result_list
    refreshed: List[str] = []
    if COMPLETED_JOB_DIR.exists():
        for candidate in sorted(COMPLETED_JOB_DIR.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() in RESULT_EXTENSIONS:
                refreshed.append(str(candidate))
    result_list = refreshed



def _load_existing_results() -> None:
    _refresh_result_list()


def _get_result_files() -> List[str]:
    _refresh_result_list()
    return result_list


def _persist_result(result_path: Optional[str]) -> Optional[str]:
    if not result_path:
        return None
    src = Path(result_path)
    if not src.exists():
        return None
    try:
        COMPLETED_JOB_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print("Unable to prepare completed job directory", exc)
        return str(src)
    target = COMPLETED_JOB_DIR / src.name
    if src == target:
        return str(src)
    try:
        shutil.move(str(src), str(target))
        return str(target)
    except Exception as move_exc:
        print("Failed to move result to completed folder", move_exc)
        try:
            shutil.copy2(str(src), str(target))
            os.remove(str(src))
            return str(target)
        except Exception as copy_exc:
            print("Failed to copy result to completed folder", copy_exc)
            return str(src)


def _run_filebrowser_upload(result_path: Optional[str]) -> None:
    if not result_path or not os.path.exists(result_path):
        return
    filebrowser_host = os.environ.get('FILEBROWSER_HOST')
    if not filebrowser_host:
        return
    try:
        if filebrowser_user := os.environ.get('FILEBROWSER_USER'):
            client = FilebrowserClient(
                filebrowser_host,
                username=filebrowser_user,
                password=os.environ.get('FILEBROWSER_PASSWORD'),
                insecure=True,
            )
        else:
            client = FilebrowserClient(filebrowser_host, insecure=True)
        asyncio.run(client.connect())
        remote_target = os.environ.get('FILEBROWSER_PATH', os.path.basename(result_path)).replace("{filename}", os.path.basename(result_path))
        coroutine = client.upload(
            local_path=result_path,
            remote_path=remote_target,
            override=True,
            concurrent=1,
        )
        asyncio.run(coroutine)
    except Exception as ex:
        print("upload failed", ex)


def _archive_job_artifacts(job: dict, pkl_path: Path) -> None:
    try:
        COMPLETED_JOB_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print("Unable to prepare completed job directory", exc)
    video_path = Path(str(job.get('video', '')))
    if video_path.exists():
        target_video = COMPLETED_JOB_DIR / video_path.name
        try:
            shutil.move(str(video_path), str(target_video))
        except (FileNotFoundError, shutil.Error) as exc:
            print("Failed to archive source video", video_path, exc)
    try:
        shutil.move(str(pkl_path), str(COMPLETED_JOB_DIR / pkl_path.name))
    except (FileNotFoundError, shutil.Error) as exc:
        print("Failed to archive job metadata", pkl_path, exc)


def _mark_job_failed(job: dict, pkl_path: Path, error: Exception) -> None:
    try:
        FAILED_JOB_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print("Unable to prepare failed job directory", exc)
    video_path = Path(str(job.get('video', '')))
    if video_path.exists():
        target_video = FAILED_JOB_DIR / video_path.name
        try:
            shutil.move(str(video_path), str(target_video))
        except (FileNotFoundError, shutil.Error) as exc:
            print("Failed to move failed job video", video_path, exc)
    try:
        shutil.move(str(pkl_path), str(FAILED_JOB_DIR / pkl_path.name))
    except (FileNotFoundError, shutil.Error) as exc:
        print("Failed to move failed job metadata", pkl_path, exc)
    failure_note = FAILED_JOB_DIR / f"{pkl_path.stem}.error.txt"
    try:
        failure_note.write_text(f"{time.time()}: {error}\n")

    except OSError:
        pass


def _run_job(job: dict) -> Optional[str]:
    if job.get('reverseTracking'):
        return process_with_reverse_tracking(
            job['video'],
            job['projection'],
            job['masks'],
            job['crf'],
            job['erode'],
            job['forceInitMask'],
            job_id=job.get('job_id'),
        )
    return process(
        job['video'],
        job['projection'],
        job['masks'],
        job['crf'],
        job['erode'],
        job['forceInitMask'],
        job_id=job.get('job_id'),
    )


def _get_worker_directories() -> List[Path]:
    if not WORKERS_DIR.exists():
        return []
    return sorted([entry for entry in WORKERS_DIR.iterdir() if entry.is_dir()])


def _read_worker_status(worker_dir: Path) -> dict:
    status_path = worker_dir / 'status.json'
    if status_path.exists():
        try:
            return json.loads(status_path.read_text())
        except json.JSONDecodeError:
            return {'state': 'unknown'}
    return {'state': 'unknown'}


def _find_idle_worker(worker_dirs: List[Path]) -> Optional[Path]:
    for worker_dir in worker_dirs:
        status = _read_worker_status(worker_dir)
        assigned = list(worker_dir.glob('*.pkl'))
        if not assigned and status.get('state') in (None, 'idle', 'unknown'):
            return worker_dir
    return None


def _dispatch_job_to_worker(pkl_path: Path, worker_dir: Path) -> None:
    with open(pkl_path, 'rb') as handle:
        job = pickle.load(handle)
    video_path = Path(job['video'])
    target_video = worker_dir / video_path.name
    target_video.parent.mkdir(parents=True, exist_ok=True)
    if video_path.exists():
        shutil.move(str(video_path), str(target_video))
    job['video'] = str(target_video)
    job['assigned_to'] = worker_dir.name
    job['dispatched_at'] = time.time()
    target_pkl = worker_dir / pkl_path.name
    with open(target_pkl, 'wb') as handle:
        pickle.dump(job, handle)
    try:
        pkl_path.unlink()
    except FileNotFoundError:
        pass


def _pending_job_files() -> List[Path]:
    if not PENDING_JOBS_DIR.exists():
        return []
    return sorted(PENDING_JOBS_DIR.glob('*.pkl'))


def _write_worker_status(worker_dir: Path, state: str, job_id: Optional[str] = None) -> None:
    status_payload = {
        'state': state,
        'job_id': job_id,
        'updated_at': time.time(),
    }
    status_path = worker_dir / 'status.json'
    try:
        status_path.write_text(json.dumps(status_payload))
    except OSError as exc:
        print('Failed to update worker status', worker_dir, exc)


_load_existing_results()
frame_name = None

def background_worker():
    global WORKER_STATUS
    surplus_url = os.environ.get('JOB_SURPLUS_CHECK_URL')
    while True:
        if not SCHEDULE:
            time.sleep(3)
            continue

        if surplus_url and not SURPLUS_IGNORE:
            try:
                start_job = "True" in str(requests.get(surplus_url, verify=False).json())
            except Exception as ex:
                start_job = False
                print(ex)

            if not start_job:
                WORKER_STATUS = "Wait for [v] surplus"
                time.sleep(30)
                continue

        worker_dirs = _get_worker_directories()
        pending_jobs = _pending_job_files()

        if worker_dirs:
            if not pending_jobs:
                WORKER_STATUS = "Idle"
                time.sleep(2)
                continue

            idle_worker = _find_idle_worker(worker_dirs)
            if idle_worker is None:
                WORKER_STATUS = "All workers busy"
                time.sleep(2)
                continue

            job_path = pending_jobs[0]
            WORKER_STATUS = f"Dispatch {job_path.name} -> {idle_worker.name}"
            try:
                _dispatch_job_to_worker(job_path, idle_worker)
            except Exception as exc:
                print("Failed to dispatch job", job_path, exc)
                time.sleep(2)
            else:
                WORKER_STATUS = "Idle"
            continue

        if not pending_jobs:
            WORKER_STATUS = "Idle"
            time.sleep(2)
            continue

        job_path = pending_jobs[0]

        try:
            with open(job_path, 'rb') as handle:
                job = pickle.load(handle)
        except Exception as exc:
            print("Failed to read job", job_path, exc)
            _mark_job_failed({'video': ''}, job_path, exc)
            time.sleep(2)
            continue

        if job.get('version') != JOB_VERSION:
            print("Skip job due to version mismatch", job_path)
            _mark_job_failed(job, job_path, Exception('JOB_VERSION mismatch'))
            time.sleep(2)
            continue

        WORKER_STATUS = f"Processing {job_path.name}"
        try:
            result = _run_job(job)
            persisted_result = _persist_result(result)
            if persisted_result and os.path.exists(persisted_result):
                _refresh_result_list()
                WORKER_STATUS = "Uploading..."
                _run_filebrowser_upload(persisted_result)
            _archive_job_artifacts(job, job_path)
        except Exception as exc:
            traceback.print_exc()
            _mark_job_failed(job, job_path, exc)
        finally:
            WORKER_STATUS = "Idle"
            time.sleep(1)


def add_job(video, projection, crf, erode, forceInitMask, reverseTracking):
    RETURN_VALUES = 16
    if video is None:
        gr.Warning("Could not add Job: Video not found", duration=5)
        return tuple(gr.skip() for _ in range(RETURN_VALUES))

    if not all(os.path.exists(os.path.join(x, '0000.png')) for x in ["masksL", "masksR"]):
        gr.Warning("Could not add Job: Mask for first frame does not exists", duration=5)
        return tuple(gr.skip() for _ in range(RETURN_VALUES))

    masksFiles = sorted([f for f in os.listdir('masksL') if f.endswith(".png") and os.path.exists(os.path.join('masksR', f))])
    masks = []
    for f in masksFiles:
        frame = cv2.imread(os.path.join('frames', f))
        width = 2*MASK_SIZE

        frameL = frame[:, :int(width/2)]
        frameR = frame[:, int(width/2):]

        maskL = Image.open(os.path.join('masksL', f)).convert('L')
        maskR = Image.open(os.path.join('masksR', f)).convert('L')

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        masks.append({
            'maskL': maskL,
            'maskR': maskR,
            'frameL': frameL,
            'frameR': frameR,
            'frameLGray': grayL,
            'frameRGray': grayR
        })

    if len(masks) == 0:
        gr.Warning("Could not add Job: Mask Missing", duration=5)
        return tuple(gr.skip() for _ in range(RETURN_VALUES))

    ts = str(int(time.time()))

    PENDING_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = PENDING_JOBS_DIR / f"{ts}_" + os.path.basename(video.name)
    shutil.move(video.name, str(dest_path))
    job_id = processing_core._compute_job_id(str(dest_path), projection)

    job_data = {
        'version': JOB_VERSION,
        'job_id': job_id,
        'video': str(dest_path),
        'projection': projection,
        'crf': crf,
        'masks': masks,
        'erode': erode,
        'forceInitMask': forceInitMask,
        'reverseTracking': reverseTracking
    }

    job_pkl_path = PENDING_JOBS_DIR / f"{ts}.pkl"
    with job_pkl_path.open("wb") as f:
        pickle.dump(job_data, f)

    return tuple(None for _ in range(RETURN_VALUES))

def status_text():
    pending_jobs = len(_pending_job_files())
    assigned_jobs = sum(len(list(worker_dir.glob('*.pkl'))) for worker_dir in _get_worker_directories())
    return ("Worker Status: " + WORKER_STATUS + "\n"
        + f"Pending Jobs: {pending_jobs} | Assigned Jobs: {assigned_jobs}")

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


def generate_gallery_list():
    frames = sorted([os.path.join('frames', f) for f in os.listdir('frames') if f.endswith(".png")])
    for idx in range(len(frames)):
        if os.path.exists(frames[idx].replace('frames', 'previews')):
            frames[idx] = frames[idx].replace('frames', 'previews')
    gallery_items = [(frame, str(str(max(0, int(idx*SECONDS - SECONDS/2))) + " sec")) for idx, frame in enumerate(frames)]
    return gallery_items

def set_mask_size(x):
    global MASK_SIZE
    MASK_SIZE = int(x)
    print("set mask size to", MASK_SIZE)

def set_extract_frames_step(x):
    global SECONDS
    SECONDS = int(x)
    print("set extract seconds to", SECONDS)

def extract_frames(video, projection, mask_size, frames_seconds):
    set_mask_size(mask_size)
    set_extract_frames_step(frames_seconds)
    for dir in ["frames", "previews", "masksL", 'masksR']:
        if os.path.exists(dir):
            shutil.rmtree(dir)

        os.makedirs(dir, exist_ok=True)

    if str(projection) == "eq":
        filter_complex = "split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; [right]crop=ih:ih:ih:0[right_crop]; [left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; [right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; [leftfisheye][rightfisheye]hstack[v]"
        os.system(f"ffmpeg -hide_banner -loglevel warning -hwaccel auto -i \"{video.name}\" -frames:v 1 -filter_complex \"[0:v]{filter_complex}\" -map \"[v]\" frames/0000.png")
        if SECONDS > 0:
            os.system(f"ffmpeg -hide_banner -loglevel warning -hwaccel auto -i \"{video.name}\" -filter_complex \"[0:v]fps=1/{SECONDS},{filter_complex}\" -map \"[v]\" -start_number 1 frames/%04d.png")
    else:
        os.system(f"ffmpeg -hide_banner -loglevel warning -hwaccel auto -i \"{video.name}\" -frames:v 1 -pix_fmt bgr24 frames/0000.png")
        if SECONDS > 0:
            os.system(f"ffmpeg -hide_banner -loglevel warning -hwaccel auto -i \"{video.name}\" -vf fps=1/{SECONDS} -pix_fmt bgr24 -start_number 1 frames/%04d.png")
    
    frames = [os.path.join('frames', f) for f in os.listdir('frames') if f.endswith(".png")]

    #NOTE: use same method for resizing to get pixel exact matches
    for frame in frames:
        img_scaled = cv2.resize(cv2.imread(frame), (2*MASK_SIZE, MASK_SIZE))
        cv2.imwrite(frame, img_scaled)

    return generate_gallery_list()

def get_selected(selected):
    global frame_name
    if selected is None or "image" not in selected:
        return None, None, None, None, None, None

    frame_name = os.path.basename(selected['image']['path'])
    if os.path.exists(os.path.join('frames', frame_name)):
        frame = cv2.imread(os.path.join('frames', frame_name))
    else:
        return None, None, None, None, None, None
    
    width = 2*MASK_SIZE

    frameL = frame[:, :int(width/2)]
    frameR = frame[:, int(width/2):]

    frameL = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)
    frameR = cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB)

    current_origin_frame['L'] = ImageFrame(frameL, 0)
    current_origin_frame['R'] = ImageFrame(frameR, 0)

    maskL = Image.fromarray(np.zeros([MASK_SIZE, MASK_SIZE, 3], dtype=np.uint8)).convert("L")
    maskR = Image.fromarray(np.zeros([MASK_SIZE, MASK_SIZE, 3], dtype=np.uint8)).convert("L")

    if os.path.exists(os.path.join('masksL', frame_name)):
        maskL = Image.open(os.path.join('masksL', frame_name)).convert("L")

    if os.path.exists(os.path.join('masksR', frame_name)):
        maskR = Image.open(os.path.join('masksR', frame_name)).convert("L")

    return Image.fromarray(frameL), \
        Image.fromarray(frameR), \
        Image.fromarray(frameL), \
        Image.fromarray(frameR), \
        maskL, \
        maskR

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

def generate_mask_preview(mask, eye: str):
    if mask is None:
        return None

    frame = current_origin_frame[eye].frame_data
    if frame is None:
        return None

    mask = Image.fromarray(mask).convert("L")
    preview = Image.composite(
        Image.new("RGB", mask.size, "blue"),
        Image.fromarray(frame).convert("RGBA"),
        mask.point(lambda p: 100 if p > 1 else 0)
    )

    return preview

def postprocess_mask(maskL, maskR, dilate, erode):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    # first dilate then erode to ensure to fill holes
    maskL = cv2.dilate(maskL, kernel, iterations=dilate)
    maskL = cv2.erode(maskL, kernel, iterations=erode)
    maskR = cv2.dilate(maskR, kernel, iterations=dilate)
    maskR = cv2.erode(maskR, kernel, iterations=erode)

    maskL = Image.fromarray(maskL).convert("L")
    maskL.save(os.path.join('masksL', frame_name))
    maskR = Image.fromarray(maskR).convert("L")
    maskR.save(os.path.join('masksR', frame_name))

    pL = generate_mask_preview(np.array(maskL), 'L')
    pR = generate_mask_preview(np.array(maskR), 'R')
    if pL is not None and pR is not None:
        pL = cv2.cvtColor(np.array(pL), cv2.COLOR_RGB2BGR)
        pR = cv2.cvtColor(np.array(pR), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join('previews', frame_name), cv2.hconcat([pL, pR]))

    return maskL, maskR, generate_gallery_list(), os.path.join('previews', frame_name)

def merge_add_mask(maskL, maskR, mergedMaskL, mergedMaskR, dilate, erode):
    if maskL is not None and mergedMaskL is not None:
        mergedMaskL = np.bitwise_or(np.array(mergedMaskL), np.array(maskL))
        mergedMaskL = Image.fromarray(mergedMaskL).convert("L")
    if maskR is not None and mergedMaskR is not None:
        mergedMaskR = np.bitwise_or(np.array(mergedMaskR), np.array(maskR))
        mergedMaskR = Image.fromarray(mergedMaskR).convert("L")

    a, b, c, d = postprocess_mask(np.array(mergedMaskL), np.array(mergedMaskR), dilate, erode)

    return  mergedMaskL, mergedMaskR, a, b, c, d

def merge_subtract_mask(maskL, maskR, mergedMaskL, mergedMaskR, dilate, erode):
    if maskL is not None and mergedMaskL is not None:
        invMaskL = 255 - np.array(maskL)
        mergedMaskL = np.bitwise_and(np.array(mergedMaskL), invMaskL)
        mergedMaskL = Image.fromarray(mergedMaskL).convert("L")
    if maskR is not None and mergedMaskR is not None:
        invMaskR = 255 - np.array(maskR)
        mergedMaskR = np.bitwise_and(np.array(mergedMaskR), invMaskR)
        mergedMaskR = Image.fromarray(mergedMaskR).convert("L")

    a, b, c, d = postprocess_mask(np.array(mergedMaskL), np.array(mergedMaskR), dilate, erode) 

    return  mergedMaskL, mergedMaskR, a, b, c, d

def set_schedule(x):
    global SCHEDULE
    SCHEDULE = bool(x)

def set_sureplus_ignore(x):
    global SURPLUS_IGNORE
    SURPLUS_IGNORE = bool(x)

def generate_example(maskL, maskR):
    frameL = current_origin_frame['L'].frame_data
    frameR = current_origin_frame['R'].frame_data

    with torch.no_grad():
        matanyone1 = MatAnyone.from_pretrained("PeiqingYang/MatAnyone")
        if torch.torch.cuda.is_available():
            matanyone1 = matanyone1.cuda()
        matanyone1 = matanyone1.eval()
        processor1 = InferenceCore(matanyone1, cfg=matanyone1.cfg)

        result = []

        maskL = np.array(Image.fromarray(maskL).convert("L"))
        maskR = np.array(Image.fromarray(maskR).convert("L"))
        for (frame, mask) in [(frameL, maskL), (frameR, maskR)]:
            objects = [1]
            mask = fix_mask2(mask)
            frame = prepare_frame(frame, torch.torch.cuda.is_available())
            output_prob = processor1.step(frame, mask, objects=objects)
            for _ in range(WARMUP):
                output_prob = processor1.step(frame, first_frame_pred=True)
            mask_output = processor1.output_prob_to_mask(output_prob)
            mask_output_pha = mask_output.unsqueeze(2).cpu().detach().numpy()
            mask_output_pha = (mask_output_pha*255).astype(np.uint8)
            result.append(mask_output_pha)

    # Workaround since Pillow can not laod directly?
    cv2.imwrite("tmp_l.png", result[0])
    cv2.imwrite("tmp_r.png", result[1])

    return generate_mask_preview(cv2.imread("tmp_l.png"), 'L'), generate_mask_preview(cv2.imread('tmp_r.png'), 'R')

def clear_completed_jobs():
    global result_list
    if not COMPLETED_JOB_DIR.exists():
        return
    for file_path in COMPLETED_JOB_DIR.iterdir():
        if file_path.is_file():
            try:
                file_path.unlink()
            except OSError as exc:
                print("Failed to remove completed file", file_path, exc)
    _refresh_result_list()



def run_worker() -> None:
    worker_id = os.environ.get('VR2AR_WORKER_ID') or os.environ.get('HOSTNAME') or 'worker'
    worker_dir = WORKERS_DIR / worker_id
    worker_dir.mkdir(parents=True, exist_ok=True)
    print(f"Worker {worker_id} watching {worker_dir}")
    _write_worker_status(worker_dir, 'idle')
    while True:
        assigned_jobs = sorted(worker_dir.glob('*.pkl'))
        if not assigned_jobs:
            _write_worker_status(worker_dir, 'idle')
            time.sleep(2)
            continue

        job_path = assigned_jobs[0]
        try:
            with open(job_path, 'rb') as handle:
                job = pickle.load(handle)
        except Exception as exc:
            print('Failed to load assigned job', job_path, exc)
            _mark_job_failed({'video': ''}, job_path, exc)
            _write_worker_status(worker_dir, 'idle')
            time.sleep(2)
            continue

        if job.get('version') != JOB_VERSION:
            print('Skip job due to version mismatch', job_path)
            _mark_job_failed(job, job_path, Exception('JOB_VERSION mismatch'))
            _write_worker_status(worker_dir, 'idle')
            time.sleep(1)
            continue

        job_id = job.get('job_id')
        _write_worker_status(worker_dir, 'processing', job_id)
        try:
            result = _run_job(job)
            persisted_result = _persist_result(result)
            if persisted_result and os.path.exists(persisted_result):
                _run_filebrowser_upload(persisted_result)
            _archive_job_artifacts(job, job_path)
        except Exception as exc:
            traceback.print_exc()
            _mark_job_failed(job, job_path, exc)
        finally:
            _write_worker_status(worker_dir, 'idle')
            time.sleep(1)

with gr.Blocks() as demo:
    gr.Markdown("# Video VR2AR Converter")
    gr.Markdown('''
        Process:
        1. Upload Your video
        2. Select Video Source Format
        3. Select Mask Size (Higher value require more VRAM!)
        3. Extract Frames
        4. Generate Initial Mask. Use Points or Prompts to generate individual Masks and merge them with add or subtract button to the initial mask. To create a second mask with points you have to use the crea button to select the points for the next partial mask. Use the foreground and subtract to remove areas from mask. use foreground and add to add areas to the inital mask. You can also try to specify additional backrground points but for me this always results in worse results.
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
        extract_frames_step = gr.Number(
            label="Extract frames every x seconds",
            minimum=0,
            maximum=600,
            step=1,
            value=SECONDS
        )

    with gr.Column():
        gr.Markdown("## Stage 2 - Extract First Frame")
        frame_button = gr.Button("Extract Projection Frame")
        gr.Markdown("### Frames")
        gr.Markdown("Important: You need to provide/generate a mask for 0 sec, Masks for the other frames are optional")
        gallery = gr.Gallery(label="Extracted Frames", show_label=True, columns=4, object_fit="contain")
        select_button = gr.Button("Load Slected Projection Frame")
        with gr.Row():
            framePreviewL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
            framePreviewR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')

    gr.Markdown("## Stage 3 - Generate Mask")
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

        gr.Markdown("Postporcessed Mask")
        mask_dilate = gr.Slider(minimum=0, maximum=10, step=1, value=1, label="Dilate Iterrations")
        mask_erode = gr.Slider(minimum=0, maximum=10, step=1, value=3, label="Erode Iterrations")
        with gr.Row():
            postprocessedMaskL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
            postprocessedMaskR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')

        gr.Markdown("Preview")
        previewMergedMask = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')

        gr.Markdown("Optional: Generate MatAnyone Example Output (For debug purpose only)")
        example_button = gr.Button("Generate Example Output")
        with gr.Row():
            exampleL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
            exampleR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')

        mask_dilate.change(
            fn=postprocess_mask,
            inputs=[mergedMaskL, mergedMaskR, mask_dilate, mask_erode],
            outputs=[postprocessedMaskL, postprocessedMaskR, gallery, previewMergedMask]
        )

        mask_erode.change(
            fn=postprocess_mask,
            inputs=[mergedMaskL, mergedMaskR, mask_dilate, mask_erode],
            outputs=[postprocessedMaskL, postprocessedMaskR, gallery, previewMergedMask]
        )

        frame_button.click(
            fn=extract_frames,
            inputs=[input_video, projection_dropdown, mask_size, extract_frames_step],
            outputs=[gallery]
        )

        selected_frame = gr.State(None)
        def store_index(evt: gr.SelectData):
            return evt.value

        gallery.select(store_index, None, selected_frame)
        select_button.click(
            fn=get_selected,
            inputs=[selected_frame],
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
            inputs=[maskL, maskR, mergedMaskL, mergedMaskR, mask_dilate, mask_erode],
            outputs=[mergedMaskL, mergedMaskR, postprocessedMaskL, postprocessedMaskR, gallery, previewMergedMask]
        )

        mask_subtract_button.click(
            fn=merge_subtract_mask,
            inputs=[maskL, maskR, mergedMaskL, mergedMaskR, mask_dilate, mask_erode],
            outputs=[mergedMaskL, mergedMaskR, postprocessedMaskL, postprocessedMaskR, gallery, previewMergedMask]
        )

        example_button.click(
            fn=generate_example,
            inputs=[postprocessedMaskL, postprocessedMaskR],
            outputs=[exampleL, exampleR]
        )

    with gr.Column():
        gr.Markdown("## Stage 4 - Add Job")
        crf_dropdown = gr.Dropdown(choices=[16,17,18,19,20,21,22], label="Encode CRF", value=16)
        erode_checkbox = gr.Checkbox(label="Erode Mask Output", value=True, info="")
        force_init_mask_checkbox = gr.Checkbox(label="Force Init Mask", value=False, info="")
        reverse_tracking_checkbox = gr.Checkbox(label="Reverse Tracking (caches mask for all frames inside /app/process until completed!)", value=True, info="")
        add_button = gr.Button("Add Job")
        add_button.click(
            fn=add_job,
            inputs=[input_video, projection_dropdown, crf_dropdown, erode_checkbox, force_init_mask_checkbox, reverse_tracking_checkbox],
            outputs=[input_video, framePreviewL, framePreviewR, maskPreviewL, mergedMaskL, maskPreviewR, mergedMaskR, maskL, maskR, maskSelectionL, maskSelectionR, previewMergedMask, exampleL, exampleR, postprocessedMaskL, postprocessedMaskR]
        )

    with gr.Column():
        gr.Markdown("## Job Control")
        schedule_checkbox = gr.Checkbox(label="Enable Job Scheduling", value=SCHEDULE, info="")
        ignore_surplus_checkbox = gr.Checkbox(label="Ignore Surplus Scheduling", value=SURPLUS_IGNORE, info="")
        clear_completed_jobs_button = gr.Button("Clear completed Jobs")
        restart_button = gr.Button("CLEANUP AND RESTART".upper())

        restart_button.click(
            # dirty hack, we use k8s restart pod
            fn=lambda: os.system("pkill python"),
            inputs=[],
            outputs=[]
        )

        clear_completed_jobs_button.click(
            fn=clear_completed_jobs
        )

        schedule_checkbox.change(
            fn=set_schedule,
            inputs=schedule_checkbox
        )

        ignore_surplus_checkbox.change(
            fn=set_sureplus_ignore,
            inputs=ignore_surplus_checkbox
        )

    with gr.Column():
        gr.Markdown("## Job Results")
        status = gr.Textbox(label="Status", lines=2)
        output_videos = gr.File(value=[], label="Download AR Videos", visible=True)
        

    timer1 = gr.Timer(2, active=True)
    timer5 = gr.Timer(5, active=True)
    timer1.tick(status_text, outputs=status)
    timer5.tick(_get_result_files, outputs=output_videos)
    demo.load(fn=status_text, outputs=status)
    demo.load(fn=_get_result_files, outputs=output_videos)


def run_ui() -> None:
    print("gradio version", gr.__version__)
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    worker_thread = threading.Thread(target=background_worker, daemon=True)
    worker_thread.start()
    demo.launch(server_name="0.0.0.0", server_port=7860)
    print("exit")


if __name__ == "__main__":
    role = os.environ.get('VR2AR_ROLE', 'ui').lower()
    if role == 'worker':
        run_worker()
    else:
        run_ui()





