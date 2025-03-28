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
import random
from sam2_executor import GroundingDinoSAM2Segment

import torch.nn.functional as F
import numpy as np

from matanyone.model.matanyone import MatAnyone
from matanyone.inference.inference_core import InferenceCore


def ffmpeg(cmd, progress, total_frames, process_start, process_end):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            
    for line in process.stdout:
        print(line.strip())
        if "frame=" in line:
            parts = line.split()
            try:
                current_frame = int(parts[1])
                progress(process_start + (current_frame / total_frames) * (process_end - process_start), desc="Converting")
            except ValueError:
                pass
    
    process.wait()
    
    if process.returncode != 0:
        print(cmd)
        return False
    
    return True

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

def fix_mask(img, mask):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)  # Add channel dimension [1, H, W]
        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask = mask[0]  # Remove batch dimension if present
            elif mask.shape[2] == 1:  # If [H, W, 1]
                mask = mask.permute(2, 0, 1)[0]  # Convert to [H, W]
    
    # Convert mask to numpy for processing
    mask_np = (mask.cpu().numpy() * 255.0).astype(np.float32)
    mask_np = gen_dilate(mask_np, 10, 10)
    mask_np = gen_erosion(mask_np, 10, 10)

    mask_tensor = torch.from_numpy(mask_np)
   
    if torch.torch.cuda.is_available():
        mask_tensor = mask_tensor.cuda()

    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor  # Keep as [H, W]
    elif mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
        mask_tensor = mask_tensor[0]  # Convert [1, H, W] to [H, W]

    img_h, img_w = img.shape[1], img.shape[2]
    mask_h, mask_w = mask_tensor.shape
    
    if mask_h != img_h or mask_w != img_w:
        # Add batch and channel dimensions for interpolation
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        mask_tensor = F.interpolate(mask_tensor, size=(img_h, img_w), mode='nearest')
        mask_tensor = mask_tensor[0, 0]  # Remove batch and channel dimensions

    return mask_tensor

def prepare_frame(frame, has_cuda=True):
    vframes = torch.from_numpy(frame)
        
    # Input is [B, H, W, C], convert to [B, C, H, W]
    if vframes.shape[-1] == 3:  # If channels are last
        vframes = vframes.permute(2, 0, 1)
    
    if has_cuda:
         vframes =  vframes.cuda()

    image_input = vframes.float() / 255.0

    return image_input

@torch.no_grad()
def process(video, projection, mode, progress=gr.Progress()):
    if video is None:
        return None, None, "No video uploaded"
    
    progress(0, desc="Starting conversion")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        original_filename = os.path.basename(video.name)
        file_name, file_extension = os.path.splitext(original_filename)
        
        temp_input_path = os.path.join(temp_dir, original_filename)
        shutil.copy(video.name, temp_input_path)

        cap = cv2.VideoCapture(temp_input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        output_filename = f"{file_name}-fisheye.{file_extension}"
        output_path = os.path.join(temp_dir, output_filename)
        # final_output_path = os.path.join(os.getcwd(), output_filename)

        if str(projection) == "eq":
            cmd = [
                "ffmpeg",
                "-i", temp_input_path,
                "-filter_complex",
                "[0:v]split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; [right]crop=ih:ih:ih:0[right_crop]; [left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; [right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; [leftfisheye][rightfisheye]hstack[v]",
                "-map", "[v]",
                "-c:a", "copy",
                "-crf", "16",
                output_path
            ]
            projection = "fisheye180"
            if not ffmpeg(cmd, progress, total_frames, 0.0, 0.2):
                return None, None, "Convertion 1 failed"
            
        else:
            output_path = temp_input_path

        progress(0.2, desc="Conversion 1 complete")


        cap = cv2.VideoCapture(output_path)

        mask_video = file_name + "-alpha.avi"
        out = cv2.VideoWriter(
            mask_video,
            cv2.VideoWriter_fourcc(*'MJPG'),
            cap.get(cv2.CAP_PROP_FPS), 
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            False
        )
        current_frame = 0

        objects = [1]

        WARMUP = 10
        has_cuda = torch.torch.cuda.is_available()

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            current_frame += 1

            oh, ow = img.shape[:2]
            img = cv2.resize(img, (2048, 1024))

            _, width = img.shape[:2]
            imgL = img[:, :int(width/2)]
            imgR = img[:, int(width/2):]

            imgLV = prepare_frame(imgL, has_cuda)
            imgRV = prepare_frame(imgR, has_cuda)

            if current_frame == 1:
                init_mask_gen = GroundingDinoSAM2Segment()

                (imgLOut, imgLMask) = init_mask_gen.predict([cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)], 0.3)
                (imgROut, imgRMask) = init_mask_gen.predict([cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)], 0.3)

                del init_mask_gen
                
                if torch.torch.cuda.is_available():
                    torch.cuda.empty_cache()

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

                imgLMask = fix_mask(imgLV, imgLMask[0])
                imgRMask = fix_mask(imgRV, imgRMask[0])

                output_prob_L = processor1.step(imgLV, imgLMask, objects=objects)
                output_prob_R = processor2.step(imgRV, imgRMask, objects=objects)
                
                for i in range(WARMUP):
                    progress(0.2 + (i / (total_frames + WARMUP)) * 0.6, desc=f"Warmup {i}/{WARMUP}")
                    output_prob_L = processor1.step(imgLV, first_frame_pred=True)
                    output_prob_R = processor2.step(imgRV, first_frame_pred=True)
            else:
                output_prob_L = processor1.step(imgLV)
                output_prob_R = processor2.step(imgRV)

            print(f"Converting {current_frame}/{total_frames}")
            progress(0.2 + ((current_frame + WARMUP) / (total_frames + WARMUP)) * 0.6, desc=f"Converting {current_frame}/{total_frames}")

            mask_output_L = processor1.output_prob_to_mask(output_prob_L)
            mask_output_R = processor2.output_prob_to_mask(output_prob_R)

            mask_output_L_pha = mask_output_L.unsqueeze(2).cpu().detach().numpy()
            mask_output_R_pha = mask_output_R.unsqueeze(2).cpu().detach().numpy()

            mask_output_L_pha = (mask_output_L_pha*255).astype(np.uint8)
            mask_output_R_pha = (mask_output_R_pha*255).astype(np.uint8)


            combined_image = cv2.hconcat([mask_output_L_pha, mask_output_R_pha])
            combined_image = cv2.resize(combined_image, (ow, oh))

            
            _, binary = cv2.threshold(combined_image, 127, 255, cv2.THRESH_BINARY)

            out.write(binary)
            gc.collect()


        cap.release()
        out.release()

        del processor1
        del processor2
        del matanyone1
        del matanyone2

        if torch.torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        print("FFmpeg Convert #2")

        result_name = file_name + "_" + projection + "_alpha" + file_extension 
        cmd = [
            "ffmpeg",
            "-i", output_path,
            "-i", mask_video,
            "-i", "mask.png",
            "-i", temp_input_path,
            "-filter_complex",
            "[1]scale=iw*0.4:-1[alpha];[2][alpha]scale2ref[mask][alpha];[alpha][mask]alphamerge,split=2[masked_alpha1][masked_alpha2]; [masked_alpha1]crop=iw/2:ih:0:0,split=2[masked_alpha_l1][masked_alpha_l2]; [masked_alpha2]crop=iw/2:ih:iw/2:0,split=4[masked_alpha_r1][masked_alpha_r2][masked_alpha_r3][masked_alpha_r4]; [0][masked_alpha_l1]overlay=W*0.5-w*0.5:-0.5*h[out_lt];[out_lt][masked_alpha_l2]overlay=W*0.5-w*0.5:H-0.5*h[out_tb]; [out_tb][masked_alpha_r1]overlay=0-w*0.5:-0.5*h[out_l_lt];[out_l_lt][masked_alpha_r2]overlay=0-w*0.5:H-0.5*h[out_tb_ltb]; [out_tb_ltb][masked_alpha_r3]overlay=W-w*0.5:-0.5*h[out_r_lt];[out_r_lt][masked_alpha_r4]overlay=W-w*0.5:H-0.5*h",
            "-c:v", "libx265", 
            "-crf", "16",
            "-preset", "veryfast",
            "-map", "3:a:?",
            "-c:a", "copy",
            result_name,
            "-y"
        ]

        if not ffmpeg(cmd, progress, total_frames, 0.8, 1.0):
            return None, None, "Convertion 2 failed"


        progress(1, desc="Conversion 2 complete")

    return mask_video, result_name, f"Conversion successful"

def process_video(video, projection, mode):
    mask_path, output_path, message = process(video, projection, mode)
    print("completed", mask_path, output_path, message)
    if mask_path and output_path:
        return gr.File(value=mask_path, visible=True), gr.File(value=output_path, visible=True), message
    else:
        return gr.File(visible=False), gr.File(visible=False), message

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Video VR2AR Converter")
    with gr.Column():
        input_video = gr.File(label="Upload Video (MKV or MP4)", file_types=["mkv", "mp4", "video"])
        with gr.Row():
            projection_dropdown = gr.Dropdown(choices=["eq", "fisheye180", "fisheye190", "fisheye200"], label="VR Video Format", value="eq")
            mode_dropdown = gr.Dropdown(choices=["method-01"], label="Method", value="method-01")
    with gr.Row():
        mask_video = gr.File(label="Download Mask Video", visible=False)
        output_video = gr.File(label="Download Converted Video", visible=False)
    convert_button = gr.Button("Convert")
    status = gr.Textbox(label="Status")
    
    convert_button.click(
        fn=process_video,
        inputs=[input_video, projection_dropdown, mode_dropdown],
        outputs=[mask_video, output_video, status]
    )

if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    demo.launch(server_name="0.0.0.0", server_port=7860)
