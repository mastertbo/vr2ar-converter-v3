"""Core processing pipeline with checkpointing and resume support."""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from data.ffmpegstream import FFmpegStream
from data.ArVideoWriter import ArVideoWriter
from matanyone.inference.inference_core import InferenceCore
from matanyone.model.matanyone import MatAnyone
from sam2.build_sam import build_sam2_video_predictor
from progress_tracker import ProcessingProgress


Helpers = Dict[str, Callable[..., Any]]


class MaskBackend:
    """Abstract mask backend interface."""

    mask_idx: int = 0

    def warm_up(self) -> None:
        """Prepare backend state before processing starts."""

    def get_mask(self, frame_idx: int, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Return mask for the frame (uint8 2D) or None to skip."""
        raise NotImplementedError

    def finalize(self) -> None:
        """Cleanup resources."""


@dataclass
class MaskBackendConfig:
    mask_backend: str = "matanyone"
    mask_infer_max_side: int = 2048
    sam2_config_path: Optional[str] = None
    sam2_checkpoint_path: Optional[str] = None

_MODEL_CACHE: Dict[str, MatAnyone] = {}


class MatAnyOneBackend(MaskBackend):
    def __init__(
        self,
        *,
        prepared_masks: List[Dict[str, Any]],
        mask_dir: Path,
        mask_w: int,
        mask_h: int,
        reader_config: Dict[str, Any],
        has_cuda: bool,
        resume_frame: int,
        initial_mask_idx: int,
        warmup: int,
        ssim_threshold: float,
        erode: bool,
        video: str,
        objects: List[int],
        prepare_frame: Callable[[np.ndarray, bool], torch.Tensor],
        fix_mask2: Callable[[Image.Image], torch.Tensor],
        force_init_mask: bool,
    ) -> None:
        self.prepared_masks = prepared_masks
        self.mask_dir = mask_dir
        self.mask_w = mask_w
        self.mask_h = mask_h
        self.reader_config = reader_config
        self.has_cuda = has_cuda
        self.resume_frame = resume_frame
        self.mask_idx = initial_mask_idx
        self.warmup = warmup
        self.ssim_threshold = ssim_threshold
        self.erode = erode
        self.video = video
        self.objects = objects
        self.prepare_frame = prepare_frame
        self.fix_mask2 = fix_mask2
        self.force_init_mask = force_init_mask

        self.processor_left: Optional[InferenceCore] = None
        self.processor_right: Optional[InferenceCore] = None

    def warm_up(self) -> None:
        self.processor_left = _create_processor(self.has_cuda)
        self.processor_right = _create_processor(self.has_cuda)

        for idx in range(self.mask_idx):
            entry = self.prepared_masks[idx]
            self.processor_left.step(entry['imgLV'], entry['maskL'], objects=self.objects, force_permanent=True)
            self.processor_right.step(entry['imgRV'], entry['maskR'], objects=self.objects, force_permanent=True)

        _warm_start(
            self.processor_left,
            self.processor_right,
            self.video,
            self.reader_config,
            self.resume_frame,
            self.mask_w,
            self.mask_h,
            self.has_cuda,
            self.objects,
            self.mask_dir,
            self.prepare_frame,
            self.fix_mask2,
            self.warmup,
        )

    def get_mask(self, frame_idx: int, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self.processor_left is None or self.processor_right is None:
            raise RuntimeError('MatAnyOne backend not initialized')

        imgL = frame_rgb[:, :self.mask_w]
        imgR = frame_rgb[:, self.mask_w:]

        imgLV = self.prepare_frame(imgL, self.has_cuda)
        imgRV = self.prepare_frame(imgR, self.has_cuda)

        frame_match = False
        if self.mask_idx < len(self.prepared_masks):
            entry = self.prepared_masks[self.mask_idx]
            if ssim(entry['frameLGray'], cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)) > self.ssim_threshold:
                if ssim(entry['frameRGray'], cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)) > self.ssim_threshold:
                    frame_match = True

        if self.force_init_mask and frame_idx == 1:
            frame_match = True

        if frame_match:
            entry = self.prepared_masks[self.mask_idx]
            self.mask_idx += 1
            output_prob_L = self.processor_left.step(imgLV, entry['maskL'], objects=self.objects)
            output_prob_R = self.processor_right.step(imgRV, entry['maskR'], objects=self.objects)
            for _ in range(max(self.warmup, 1)):
                output_prob_L = self.processor_left.step(imgLV, first_frame_pred=self.mask_idx == 1)
                output_prob_R = self.processor_right.step(imgRV, first_frame_pred=self.mask_idx == 1)
        elif self.mask_idx > 0:
            output_prob_L = self.processor_left.step(imgLV)
            output_prob_R = self.processor_right.step(imgRV)
        else:
            return None

        mask_output_L = self.processor_left.output_prob_to_mask(output_prob_L)
        mask_output_R = self.processor_right.output_prob_to_mask(output_prob_R)

        mask_output_L_pha = (mask_output_L.unsqueeze(2).cpu().detach().numpy() * 255).astype(np.uint8)
        mask_output_R_pha = (mask_output_R.unsqueeze(2).cpu().detach().numpy() * 255).astype(np.uint8)

        if self.erode:
            mask_output_L_pha = cv2.erode(mask_output_L_pha, (3, 3), iterations=1)
            mask_output_R_pha = cv2.erode(mask_output_R_pha, (3, 3), iterations=1)

        combined_mask = cv2.hconcat([mask_output_L_pha, mask_output_R_pha])
        return combined_mask

    def finalize(self) -> None:
        self.processor_left = None
        self.processor_right = None


class Sam2Backend(MaskBackend):
    def __init__(
        self,
        *,
        video: str,
        resume_frame: int,
        video_width: int,
        video_height: int,
        infer_max_side: int,
        sam2_config_path: str,
        sam2_checkpoint_path: str,
        blur: bool = True,
        initial_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.video = video
        self.video_width = video_width
        self.video_height = video_height
        self.infer_max_side = infer_max_side
        self.sam2_config_path = sam2_config_path
        self.sam2_checkpoint_path = sam2_checkpoint_path
        self.blur = blur
        self.initial_mask = initial_mask
        self.predictor = None
        self.inference_state = None
        self.tracking_iter: Optional[Iterable[Tuple[int, List[int], torch.Tensor]]] = None
        self.mask_idx = 0
        self.start_frame_idx = max(resume_frame, 0)

    def _effective_image_size(self) -> Optional[int]:
        if not self.infer_max_side:
            return None
        max_side = max(self.video_width or 0, self.video_height or 0)
        if max_side <= 0:
            return None
        return min(self.infer_max_side, max_side)

    def warm_up(self) -> None:
        def _oom_safe_predictor() -> torch.nn.Module:
            candidates: List[Optional[int]] = []
            effective_size = self._effective_image_size()
            if effective_size:
                size = effective_size
                while size >= 512:
                    candidates.append(size)
                    next_size = size // 2
                    if next_size == size:
                        break
                    size = next_size
            candidates.append(None)

            devices: List[str] = []
            if torch.cuda.is_available():
                devices.append("cuda")
            devices.append("cpu")

            last_err: Optional[BaseException] = None
            for device in devices:
                for candidate in candidates:
                    hydra_overrides_extra = [f"++model.image_size={candidate}"] if candidate else []
                    try:
                        return build_sam2_video_predictor(
                            config_file=self.sam2_config_path,
                            ckpt_path=self.sam2_checkpoint_path,
                            device=device,
                            hydra_overrides_extra=hydra_overrides_extra,
                            apply_postprocessing=True,
                        )
                    except torch.cuda.OutOfMemoryError as err:
                        torch.cuda.empty_cache()
                        last_err = err
                        continue
                    except RuntimeError as err:
                        if "out of memory" in str(err).lower():
                            if device == "cuda":
                                torch.cuda.empty_cache()
                                last_err = err
                                continue
                        raise
            if last_err:
                raise last_err
            raise RuntimeError("failed to initialize SAM2 predictor: no viable device")

        self.predictor = _oom_safe_predictor()
        self.inference_state = self.predictor.init_state(self.video)
        video_h = self.inference_state.get('video_height', None)
        video_w = self.inference_state.get('video_width', None)
        if video_h is None or video_w is None:
            raise RuntimeError('SAM2 inference state missing video dimensions')
        if self.initial_mask is not None:
            mask_np = self.initial_mask
            if mask_np.shape[0] != video_h or mask_np.shape[1] != video_w:
                mask_np = cv2.resize(mask_np, (video_w, video_h), interpolation=cv2.INTER_NEAREST)
            init_mask = torch.from_numpy((mask_np > 127).astype(np.bool_))
        else:
            init_mask = torch.ones((video_h, video_w), dtype=torch.bool)
        self.predictor.add_new_mask(self.inference_state, frame_idx=0, obj_id=1, mask=init_mask)
        max_frame_idx = self.inference_state.get('num_frames', 0) - 1
        start_idx = min(self.start_frame_idx, max(max_frame_idx, 0))
        self.tracking_iter = self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx=start_idx,
        )

    def get_mask(self, frame_idx: int, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self.tracking_iter is None:
            raise RuntimeError('SAM2 backend not initialized')
        try:
            expected_frame = max(frame_idx - 1, 0)
            tracked_idx = -1
            mask_tensor = None
            while tracked_idx < expected_frame:
                tracked_idx, _, mask_tensor = next(self.tracking_iter)
        except StopIteration:
            return None
        except torch.cuda.OutOfMemoryError as err:
            torch.cuda.empty_cache()
            raise RuntimeError('SAM2 ran out of memory during propagation; reduce mask_infer_max_side or pin the backend to CPU') from err
        except RuntimeError as err:
            if 'out of memory' in str(err).lower():
                torch.cuda.empty_cache()
                raise RuntimeError('SAM2 ran out of memory during propagation; lower resolution or switch device') from err
            raise

        if tracked_idx != expected_frame:
            return None

        self.mask_idx += 1
        if mask_tensor is None:
            return None

        if isinstance(mask_tensor, torch.Tensor):
            mask_np = mask_tensor.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask_tensor)

        while mask_np.ndim > 2:
            mask_np = mask_np.squeeze(axis=0)
        if mask_np.ndim == 3:
            mask_np = mask_np.max(axis=0)
        mask_np = np.clip(mask_np, 0.0, 1.0)
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        if self.blur:
            mask_uint8 = cv2.GaussianBlur(mask_uint8, (5, 5), sigmaX=1.0)
        if mask_uint8.shape[:2] != frame_rgb.shape[:2]:
            mask_uint8 = cv2.resize(mask_uint8, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return mask_uint8

    def finalize(self) -> None:
        self.tracking_iter = None
        self.inference_state = None
        self.predictor = None


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


def _scan_mask_directory(mask_dir: Path) -> Tuple[List[Tuple[int, Path]], int, int, bool]:
    if not mask_dir.exists() or not mask_dir.is_dir():
        raise FileNotFoundError(f"mask directory does not exist: {mask_dir}")

    mask_candidates: List[Path] = sorted(mask_dir.glob('*.png'))
    if not mask_candidates:
        mask_candidates = sorted(mask_dir.glob('*.PNG'))
    indexed_masks: List[Tuple[int, Path]] = []
    for candidate in mask_candidates:
        try:
            frame_idx = int(candidate.stem)
        except ValueError:
            continue
        indexed_masks.append((frame_idx, candidate))

    if not indexed_masks:
        raise ValueError(f'no numeric mask files found in {mask_dir}')

    indexed_masks.sort(key=lambda item: item[0])
    sample = cv2.imread(str(indexed_masks[0][1]), cv2.IMREAD_UNCHANGED)
    if sample is None:
        raise ValueError(f'failed to read mask image: {indexed_masks[0][1]}')
    if sample.ndim == 3 and sample.shape[2] > 1:
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

    mask_h, total_w = sample.shape[:2]
    if total_w % 2 != 0:
        raise ValueError('expected stereo masks with even combined width')
    mask_w = total_w // 2

    zero_based = indexed_masks[0][0] == 0

    for _, path in indexed_masks[1:10]:
        probe = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if probe is None:
            raise ValueError(f'failed to read mask image: {path}')
        if probe.ndim == 3 and probe.shape[2] > 1:
            probe = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
        if probe.shape[0] != mask_h or probe.shape[1] != total_w:
            raise ValueError(f'inconsistent mask resolution in {mask_dir}')

    return indexed_masks, mask_w, mask_h, zero_based


def _normalize_frame_index(frame_idx: int, zero_based: bool) -> int:
    if zero_based:
        return frame_idx + 1
    return frame_idx if frame_idx > 0 else 1


def _mask_coverage_gaps(indexed_masks: List[Tuple[int, Path]], zero_based: bool,
                        total_frames: int) -> List[int]:
    available_frames = {
        _normalize_frame_index(frame_idx, zero_based)
        for frame_idx, _ in indexed_masks if frame_idx >= 0
    }
    missing = [frame for frame in range(1, total_frames + 1) if frame not in available_frames]
    return missing


def _prepare_mask_guides_from_directory(video: str, projection: str, indexed_masks: List[Tuple[int, Path]],
                                        zero_based: bool, mask_w: int, mask_h: int,
                                        stride: int = 1) -> List[Dict[str, Any]]:
    if stride < 1:
        raise ValueError('stride must be >= 1')

    guides: List[Dict[str, Any]] = []
    if projection == 'eq':
        filter_complex = (
            f"[0:v]split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; "
            f"[right]crop=ih:ih:ih:0[right_crop]; "
            f"[left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; "
            f"[right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; "
            f"[leftfisheye][rightfisheye]hstack,scale={2 * mask_w}:{mask_h}[v]"
        )
    else:
        filter_complex = None

    for ordinal, (frame_idx, path) in enumerate(indexed_masks):
        if stride > 1 and (ordinal % stride) != 0:
            continue

        frame_for_video = max(_normalize_frame_index(frame_idx, zero_based) - 1, 0)
        frame = FFmpegStream.get_frame(video, frame_for_video, filter_complex=filter_complex)
        if frame.shape[1] != 2 * mask_w or frame.shape[0] != mask_h:
            frame = cv2.resize(frame, (2 * mask_w, mask_h), interpolation=cv2.INTER_AREA)

        mask_frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask_frame is None:
            raise ValueError(f'failed to read mask image: {path}')
        if mask_frame.ndim == 3 and mask_frame.shape[2] > 1:
            mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        if mask_frame.shape[1] != 2 * mask_w or mask_frame.shape[0] != mask_h:
            mask_frame = cv2.resize(mask_frame, (2 * mask_w, mask_h), interpolation=cv2.INTER_NEAREST)

        maskL = Image.fromarray(mask_frame[:, :mask_w]).convert('L')
        maskR = Image.fromarray(mask_frame[:, mask_w:]).convert('L')

        frameL = frame[:, :mask_w]
        frameR = frame[:, mask_w:]
        frameLGray = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        frameRGray = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        guides.append({
            'maskL': maskL,
            'maskR': maskR,
            'frameL': frameL,
            'frameR': frameR,
            'frameLGray': frameLGray,
            'frameRGray': frameRGray,
        })

    return guides


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


def stitch_existing_masks(*, video: str, projection: str, mask_dir: str, crf: int,
                          helpers: Helpers, job_id: Optional[str] = None,
                          job_version: int = 0) -> str:
    set_status = helpers['set_status']

    mask_dir_path = Path(mask_dir)
    set_status('Validate existing masks...')
    indexed_masks, mask_w, mask_h, zero_based = _scan_mask_directory(mask_dir_path)

    video_info = FFmpegStream.get_video_info(video)
    missing_frames = _mask_coverage_gaps(indexed_masks, zero_based, video_info.length)
    if missing_frames:
        preview = ', '.join(f"{frame:06d}" for frame in missing_frames[:10])
        raise ValueError(f'missing masks for {len(missing_frames)} frame(s): {preview}')

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
            f"[leftfisheye][rightfisheye]hstack,scale={2 * mask_w}:{mask_h}[v]"
        )
    else:
        reader_config['video_filter'] = f'scale={2 * mask_w}:{mask_h}'

    job_identifier = job_id or _compute_job_id(video, projection_out)
    job_root = Path('process') / 'runs' / job_identifier
    (job_root / 'masks').mkdir(parents=True, exist_ok=True)
    (job_root / 'debug').mkdir(parents=True, exist_ok=True)

    metadata = {
        'job_id': job_identifier,
        'video': video,
        'projection': projection_out,
        'total_frames': video_info.length,
        'version': job_version,
    }
    progress = ProcessingProgress(str(job_root), metadata)
    progress.update(last_frame=video_info.length, mask_idx=len(indexed_masks), total_frames=video_info.length)

    tmp_name = f"{os.path.splitext(os.path.basename(video))[0]}_{projection_out.upper()}_alpha_tmp{os.path.splitext(os.path.basename(video))[1]}"
    result_name = f"{os.path.splitext(os.path.basename(video))[0]}_{projection_out.upper()}_alpha{os.path.splitext(os.path.basename(video))[1]}"

    set_status('Encode alpha using existing masks...')
    _combine_with_alpha(
        video=video,
        reader_config=reader_config,
        mask_dir=mask_dir_path,
        tmp_name=tmp_name,
        result_name=result_name,
        video_info=video_info,
        crf=crf,
        set_status=set_status,
    )

    progress.mark_completed(result_name)
    return result_name


def prepare_masks_from_directory(video: str, projection: str, mask_dir: str,
                                 stride: int = 1) -> List[Dict[str, Any]]:
    indexed_masks, mask_w, mask_h, zero_based = _scan_mask_directory(Path(mask_dir))
    guides = _prepare_mask_guides_from_directory(
        video,
        projection,
        indexed_masks,
        zero_based,
        mask_w,
        mask_h,
        stride=stride,
    )
    if not guides:
        raise ValueError('no mask frames selected with the provided stride')
    return guides


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
                  warmup: int = 4, ssim_threshold: float = 0.983,
                  mask_backend_cfg: Optional[MaskBackendConfig] = None) -> str:
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

    backend_cfg = mask_backend_cfg or MaskBackendConfig()
    prepared_masks = _prepare_manual_masks(masks, prepare_frame, fix_mask2, has_cuda)

    backend_choice = (backend_cfg.mask_backend or 'matanyone').lower()
    initial_mask_np = cv2.hconcat([
        np.array(masks[0]['maskL'], dtype=np.uint8),
        np.array(masks[0]['maskR'], dtype=np.uint8),
    ])
    if backend_choice == 'sam2':
        config_path = backend_cfg.sam2_config_path or 'sam2_configs/sam2_1_hiera_tiny.yaml'
        ckpt_path = backend_cfg.sam2_checkpoint_path or 'model/sam2_1_hiera_tiny.pt'
        backend: MaskBackend = Sam2Backend(
            video=video,
            resume_frame=resume_frame,
            video_width=video_info.width,
            video_height=video_info.height,
            infer_max_side=backend_cfg.mask_infer_max_side,
            sam2_config_path=config_path,
            sam2_checkpoint_path=ckpt_path,
            initial_mask=initial_mask_np,
        )
    else:
        backend = MatAnyOneBackend(
            prepared_masks=prepared_masks,
            mask_dir=mask_dir,
            mask_w=mask_w,
            mask_h=mask_h,
            reader_config=reader_config,
            has_cuda=has_cuda,
            resume_frame=resume_frame,
            initial_mask_idx=maskIdx,
            warmup=warmup,
            ssim_threshold=ssim_threshold,
            erode=erode,
            video=video,
            objects=[1],
            prepare_frame=prepare_frame,
            fix_mask2=fix_mask2,
            force_init_mask=force_init_mask,
        )

    backend.warm_up()

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

        combined_mask = backend.get_mask(current_frame, frame)
        if combined_mask is None:
            continue

        mask_path = mask_dir / f"{current_frame:06d}.png"
        cv2.imwrite(str(mask_path), combined_mask)

        progress.update(last_frame=current_frame, mask_idx=getattr(backend, 'mask_idx', maskIdx), total_frames=video_info.length)
        set_status(f"Create Mask {current_frame}/{video_info.length}")

    ffmpeg.stop()
    backend.finalize()

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

