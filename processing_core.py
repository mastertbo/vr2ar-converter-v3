"""Core processing pipeline with checkpointing and resume support."""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from data.ffmpegstream import FFmpegStream
from data.ArVideoWriter import ArVideoWriter
from matanyone.inference.inference_core import InferenceCore
from matanyone.model.matanyone import MatAnyone
from progress_tracker import ProcessingProgress


Helpers = Dict[str, Callable[..., Any]]

_MODEL_CACHE: Dict[str, MatAnyone] = {}


def _compute_job_id(video_path: str, projection: str) -> str:
    normalized = (str(Path(video_path).resolve()) + '|' + projection).encode('utf-8')
    return hashlib.sha1(normalized).hexdigest()[:16]


def _get_base_model(device: str) -> MatAnyone:
    model = _MODEL_CACHE.get(device)
    if model is None:
        model = MatAnyone.from_pretrained("PeiqingYang/MatAnyone")
        if device == 'cuda':
            model = model.cuda()
        model = model.eval()
        _MODEL_CACHE[device] = model
    elif device == 'cuda' and next(model.parameters()).device.type != 'cuda':
        model = model.cuda().eval()
        _MODEL_CACHE[device] = model
    elif device == 'cpu' and next(model.parameters()).device.type != 'cpu':
        model = model.cpu().eval()
        _MODEL_CACHE[device] = model
    return model


def _create_processor(has_cuda: bool) -> InferenceCore:
    device = 'cuda' if has_cuda else 'cpu'
    base_model = _get_base_model(device)
    return InferenceCore(base_model, cfg=base_model.cfg)


def _prepare_manual_masks(masks: List[Dict[str, Any]], prepare_frame: Callable[[np.ndarray, bool], torch.Tensor],
                          fix_mask2: Callable[[Image.Image], torch.Tensor], has_cuda: bool) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for entry in masks:
        prepared.append({
            'imgLV': prepare_frame(entry['frameL'], has_cuda),
            'imgRV': prepare_frame(entry['frameR'], has_cuda),
            'maskL': fix_mask2(entry['maskL']),
            'maskR': fix_mask2(entry['maskR']),
            'frameLGray': entry['frameLGray'],
            'frameRGray': entry['frameRGray'],
        })
    return prepared


def _detect_last_mask(mask_dir: Path) -> int:
    if not mask_dir.exists():
        return 0
    frames: List[int] = []
    for png in mask_dir.glob('*.png'):
        try:
            frames.append(int(png.stem))
        except ValueError:
            continue
    return max(frames) if frames else 0


def _warm_start(processor_left: InferenceCore, processor_right: InferenceCore,
                video: str, reader_config: Dict[str, Any], resume_frame: int,
                mask_w: int, mask_h: int, has_cuda: bool, objects: List[int], mask_dir: Path,
                prepare_frame: Callable[[np.ndarray, bool], torch.Tensor],
                fix_mask2: Callable[[Image.Image], torch.Tensor], warmup: int) -> None:
    if resume_frame <= 0:
        return
    mask_path = mask_dir / f"{resume_frame:06d}.png"
    if not mask_path.exists():
        return

    filter_complex = reader_config.get('filter_complex')
    frame = FFmpegStream.get_frame(
        video,
        max(resume_frame - 1, 0),
        filter_complex=filter_complex,
    )
    if frame is None:
        return
    if filter_complex is None:
        frame = cv2.resize(frame, (mask_w * 2, mask_h))

    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return

    imgL = frame[:, :mask_w]
    imgR = frame[:, mask_w:]

    split = mask_img.shape[1] // 2
    maskL = mask_img[:, :split]
    maskR = mask_img[:, split:]

    imgLV = prepare_frame(imgL, has_cuda)
    imgRV = prepare_frame(imgR, has_cuda)
    maskLTensor = fix_mask2(Image.fromarray(maskL))
    maskRTensor = fix_mask2(Image.fromarray(maskR))

    processor_left.step(imgLV, maskLTensor, objects=objects)
    processor_right.step(imgRV, maskRTensor, objects=objects)
    for _ in range(max(warmup, 1)):
        processor_left.step(imgLV)
        processor_right.step(imgRV)


def _combine_with_alpha(video: str, reader_config: Dict[str, Any], mask_dir: Path,
                        tmp_name: str, result_name: str, video_info: Any, crf: int,
                        set_status: Callable[[str], None]) -> None:
    writer = ArVideoWriter(tmp_name, video_info.fps, crf)
    ffmpeg = FFmpegStream(
        video_path=video,
        config=reader_config,
        skip_frames=0,
        start_frame=0,
        watchdog_timeout_in_seconds=30,
    )

    frame_idx = 0
    set_status('Encode Alpha Video...')
    while ffmpeg.isOpen():
        frame = ffmpeg.read()
        if frame is None:
            break
        frame_idx += 1
        mask_path = mask_dir / f"{frame_idx:06d}.png"
        if not mask_path.exists():
            break
        mask_frame = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        writer.add_frame(frame, mask_frame)

    ffmpeg.stop()
    writer.finalize()
    while not writer.is_finished():
        time.sleep(0.5)

    subprocess.run([
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'warning',
        '-i', tmp_name,
        '-i', video,
        '-c', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0?',
        result_name,
    ], check=False)

    if os.path.exists(tmp_name):
        os.remove(tmp_name)


def process_video(*, video: str, projection: str, masks: List[Dict[str, Any]],
                  crf: int, erode: bool, force_init_mask: bool,
                  reverse_tracking: bool, helpers: Helpers,
                  job_id: Optional[str] = None, job_version: int = 0,
                  warmup: int = 4, ssim_threshold: float = 0.983) -> str:
    if not masks:
        raise ValueError('mask list is empty')

    prepare_frame = helpers['prepare_frame']
    fix_mask2 = helpers['fix_mask2']
    set_status = helpers['set_status']

    mask_w, mask_h = masks[0]['maskL'].size
    projection_out = 'fisheye180' if projection == 'eq' else projection

    reader_config: Dict[str, Any] = {
        'parameter': {
            'width': 2 * mask_w,
            'height': mask_h,
        }
    }
    if projection == 'eq':
        reader_config['filter_complex'] = (
            f"[0:v]split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; "
            f"[right]crop=ih:ih:ih:0[right_crop]; "
            f"[left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; "
            f"[right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; "
            f"[leftfisheye][rightfisheye]hstack,scale={2*mask_w}:{mask_h}[v]"
        )
    else:
        reader_config['video_filter'] = f'scale={2 * mask_w}:{mask_h}'

    video_info = FFmpegStream.get_video_info(video)
    has_cuda = torch.cuda.is_available()
    job_identifier = job_id or _compute_job_id(video, projection_out)

    job_root = Path('process') / 'runs' / job_identifier
    mask_dir = job_root / 'masks'
    mask_dir.mkdir(parents=True, exist_ok=True)
    (job_root / 'debug').mkdir(parents=True, exist_ok=True)

    metadata = {
        'job_id': job_identifier,
        'video': video,
        'projection': projection_out,
        'total_frames': video_info.length,
        'version': job_version,
    }
    progress = ProcessingProgress(str(job_root), metadata)
    resume_frame = max(progress.last_frame, _detect_last_mask(mask_dir))
    maskIdx = max(0, min(progress.mask_idx, len(masks)))

    prepared_masks = _prepare_manual_masks(masks, prepare_frame, fix_mask2, has_cuda)

    processor_left = _create_processor(has_cuda)
    processor_right = _create_processor(has_cuda)
    objects = [1]

    for idx in range(maskIdx):
        entry = prepared_masks[idx]
        processor_left.step(entry['imgLV'], entry['maskL'], objects=objects, force_permanent=True)
        processor_right.step(entry['imgRV'], entry['maskR'], objects=objects, force_permanent=True)

    _warm_start(
        processor_left,
        processor_right,
        video,
        reader_config,
        resume_frame,
        mask_w,
        mask_h,
        has_cuda,
        objects,
        mask_dir,
        prepare_frame,
        fix_mask2,
        warmup,
    )

    ffmpeg = FFmpegStream(
        video_path=video,
        config=reader_config,
        skip_frames=0,
        start_frame=resume_frame,
        watchdog_timeout_in_seconds=0 if reverse_tracking else 30,
    )

    current_frame = resume_frame
    set_status(f"Create Mask {current_frame}/{video_info.length}")

    while ffmpeg.isOpen():
        frame = ffmpeg.read()
        if frame is None:
            break
        current_frame += 1

        imgL = frame[:, :mask_w]
        imgR = frame[:, mask_w:]

        imgLV = prepare_frame(imgL, has_cuda)
        imgRV = prepare_frame(imgR, has_cuda)

        frame_match = False
        if force_init_mask and current_frame == 1:
            frame_match = True
        if maskIdx < len(prepared_masks):
            entry = prepared_masks[maskIdx]
            if ssim(entry['frameLGray'], cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)) > ssim_threshold:
                if ssim(entry['frameRGray'], cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)) > ssim_threshold:
                    frame_match = True

        if frame_match:
            entry = prepared_masks[maskIdx]
            maskIdx += 1
            output_prob_L = processor_left.step(imgLV, entry['maskL'], objects=objects)
            output_prob_R = processor_right.step(imgRV, entry['maskR'], objects=objects)
            for _ in range(max(warmup, 1)):
                output_prob_L = processor_left.step(imgLV, first_frame_pred=maskIdx == 1)
                output_prob_R = processor_right.step(imgRV, first_frame_pred=maskIdx == 1)
        elif maskIdx > 0:
            output_prob_L = processor_left.step(imgLV)
            output_prob_R = processor_right.step(imgRV)
        else:
            continue

        mask_output_L = processor_left.output_prob_to_mask(output_prob_L)
        mask_output_R = processor_right.output_prob_to_mask(output_prob_R)

        mask_output_L_pha = (mask_output_L.unsqueeze(2).cpu().detach().numpy() * 255).astype(np.uint8)
        mask_output_R_pha = (mask_output_R.unsqueeze(2).cpu().detach().numpy() * 255).astype(np.uint8)

        if erode:
            mask_output_L_pha = cv2.erode(mask_output_L_pha, (3, 3), iterations=1)
            mask_output_R_pha = cv2.erode(mask_output_R_pha, (3, 3), iterations=1)

        combined_mask = cv2.hconcat([mask_output_L_pha, mask_output_R_pha])
        mask_path = mask_dir / f"{current_frame:06d}.png"
        cv2.imwrite(str(mask_path), combined_mask)

        progress.update(last_frame=current_frame, mask_idx=maskIdx, total_frames=video_info.length)
        set_status(f"Create Mask {current_frame}/{video_info.length}")

    ffmpeg.stop()

    tmp_name = f"{os.path.splitext(os.path.basename(video))[0]}_{projection_out.upper()}_alpha_tmp{os.path.splitext(os.path.basename(video))[1]}"
    result_name = f"{os.path.splitext(os.path.basename(video))[0]}_{projection_out.upper()}_alpha{os.path.splitext(os.path.basename(video))[1]}"

    _combine_with_alpha(
        video=video,
        reader_config=reader_config,
        mask_dir=mask_dir,
        tmp_name=tmp_name,
        result_name=result_name,
        video_info=video_info,
        crf=crf,
        set_status=set_status,
    )

    progress.mark_completed(result_name)
    return result_name

