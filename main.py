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
import torch
import cv2
import gc
import time
import threading
import queue
import subprocess
import random
from sam2_executor import GroundingDinoSAM2Segment, SAM2PointSegment
from PIL import Image

import torch.nn.functional as F
import numpy as np

from matanyone.model.matanyone import MatAnyone
from matanyone.inference.inference_core import InferenceCore

from data.ffmpegstream import FFmpegStream
from data.ArVideoWriter import ArVideoWriter
from video_process import ImageFrame

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

def prepare_frame(frame, has_cuda=True):
    vframes = torch.from_numpy(frame)
        
    if vframes.shape[-1] == 3:
        vframes = vframes.permute(2, 0, 1)
    
    if has_cuda:
         vframes =  vframes.cuda()

    image_input = vframes.float() / 255.0

    return image_input

def fix_mask2(mask):
    mask = np.array(mask)
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)
    mask = torch.from_numpy(mask)
    if torch.torch.cuda.is_available():
        mask = mask.cuda()

    return mask

@torch.no_grad()
def process(job_id, video, projection, maskL, maskR, crf = 16):
    global WORKER_STATUS

    _, mask_h = maskL.size

    maskL = fix_mask2(maskL)
    maskR = fix_mask2(maskR)

    original_filename = os.path.basename(video.name)
    file_name, file_extension = os.path.splitext(original_filename)
    file_name = str(job_id).zfill(4) + "_" + file_name
    
    video_info = FFmpegStream.get_video_info(video.name)
    
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

    # adopted from published repo
    WARMUP = 10
    has_cuda = torch.torch.cuda.is_available()

    ffmpeg = FFmpegStream(
        video_path = video.name,
        config = reader_config,
        skip_frames = 0
    )

    result_tmp_name = file_name.replace(' ', '_') + "_" + str(projection).upper() + "_alpha_tmp" + file_extension 
    result_name = file_name.replace(' ', '_') + "_" + str(projection).upper() + "_alpha" + file_extension 
    writer = ArVideoWriter(result_tmp_name, video_info.fps, crf)

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

    while ffmpeg.isOpen():
        img = ffmpeg.read()
        if img is None:
            break
        current_frame += 1

        img_scaled = cv2.resize(img, (2*mask_h, mask_h))

        _, width = img_scaled.shape[:2]
        imgL = img_scaled[:, :int(width/2)]
        imgR = img_scaled[:, int(width/2):]

        imgLV = prepare_frame(imgL, has_cuda)
        imgRV = prepare_frame(imgR, has_cuda)

        if current_frame == 1:
            imgLMask = maskL
            imgRMask = maskR

            output_prob_L = processor1.step(imgLV, imgLMask, objects=objects)
            output_prob_R = processor2.step(imgRV, imgRMask, objects=objects)
            
            for i in range(WARMUP):
                WORKER_STATUS = f"Warmup MatAnyone {i+1}/{WARMUP}"
                output_prob_L = processor1.step(imgLV, first_frame_pred=True)
                output_prob_R = processor2.step(imgRV, first_frame_pred=True)
        else:
            output_prob_L = processor1.step(imgLV)
            output_prob_R = processor2.step(imgRV)

        WORKER_STATUS = f"Create Mask {current_frame}/{video_info.length}"
        # print(WORKER_STATUS)

        mask_output_L = processor1.output_prob_to_mask(output_prob_L)
        mask_output_R = processor2.output_prob_to_mask(output_prob_R)

        mask_output_L_pha = mask_output_L.unsqueeze(2).cpu().detach().numpy()
        mask_output_R_pha = mask_output_R.unsqueeze(2).cpu().detach().numpy()

        mask_output_L_pha = (mask_output_L_pha*255).astype(np.uint8)
        mask_output_R_pha = (mask_output_R_pha*255).astype(np.uint8)

        mask_output_L_pha = cv2.erode(mask_output_L_pha, (3,3), iterations=1)
        mask_output_R_pha = cv2.erode(mask_output_R_pha, (3,3), iterations=1)

        combined_mask = cv2.hconcat([mask_output_L_pha, mask_output_R_pha])
        
        writer.add_frame(img, combined_mask)

        gc.collect()

    del processor1
    del processor2
    del matanyone1
    del matanyone2

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
        "-i", video.name,
        "-c", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        result_name
    ])

    os.remove(result_tmp_name)

    WORKER_STATUS = f"Convertion completed" 
    return (None, result_name)


job_id = 0
task_queue = queue.Queue()
mask_list = []
result_list = []

def background_worker():
    global file_list
    global WORKER_STATUS
    while True:
        job = task_queue.get()
        if job is None:
            break
        print("Start job", job['id'])
        (mask, result) = process(job['id'], job['video'], job['projection'], job['maskL'], job['maskR'], job['crf'])
        if mask is not None:
            mask_list.append(mask)
        if result is not None:
            result_list.append(result)
        task_queue.task_done()
        time.sleep(5)
        WORKER_STATUS = "Idle"

def add_job(video, projection, maskL, maskR, crf):
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
    task_queue.put({
        'id': job_id,
        'video': video,
        'projection': projection,
        'crf': crf,
        'maskL': Image.fromarray(maskL).convert('L'),
        'maskR': Image.fromarray(maskR).convert('L')
    })

    return None, None, None, None, None, None, None, None, None, None, None, None, None

def status_text():
    global task_queue
    return "Worker Status: " + WORKER_STATUS + "\n" \
        + f"Pending Jobs: {task_queue.qsize()}"

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
        add_button = gr.Button("Add Job")
        add_button.click(
            fn=add_job,
            inputs=[input_video, projection_dropdown, mergedMaskL, mergedMaskR, crf_dropdown],
            outputs=[input_video, framePreviewL, framePreviewR, maskPreviewL, mergedMaskL, maskPreviewR, mergedMaskR, maskL, maskR, maskSelectionL, maskSelectionR, previewMergedMaskL, previewMergedMaskR ]
        )

    with gr.Column():
        gr.Markdown("## Job Results")
        status = gr.Textbox(label="Status", lines=2)
        mask_videos = gr.File(value=[], label="Download Mask Videos", visible=True)
        output_videos = gr.File(value=[], label="Download AR Videos", visible=True)
        restart_button = gr.Button("Clear all jobs and data".upper())
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
    timer5.tick(lambda: mask_list, outputs=mask_videos)
    demo.load(fn=status_text, outputs=status)
    demo.load(fn=lambda: mask_list, outputs=mask_videos)
    demo.load(fn=lambda: result_list, outputs=output_videos)


if __name__ == "__main__":
    print("gradio version", gr.__version__)
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    worker_thread = threading.Thread(target=background_worker, daemon=True)
    worker_thread.start()
    demo.launch(server_name="0.0.0.0", server_port=7860)
    print("exit")
