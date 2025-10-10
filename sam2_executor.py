import os
import gc
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import copy
import torch
import numpy as np
from PIL import Image
import logging
from torch.hub import download_url_to_file
from urllib.parse import urlparse
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from local_groundingdino.datasets import transforms as T
from local_groundingdino.util.utils import (
    clean_state_dict as local_groundingdino_clean_state_dict,
)
from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from local_groundingdino.models import build_model as local_groundingdino_build_model
import glob
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

logger = logging.getLogger("SAM2")

sam_model_dir_name = "sam2"
sam_model_list = {
    "sam2_hiera_tiny": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
    },
    "sam2_hiera_small.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
    },
    "sam2_hiera_base_plus.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
    },
    "sam2_hiera_large.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    },
    "sam2_1_hiera_tiny.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    },
    "sam2_1_hiera_small.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
    },
    "sam2_1_hiera_base_plus.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    },
    "sam2_1_hiera_large.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    },
}

groundingdino_model_dir_name = "grounding-dino"
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth",
    },
}


def get_bert_base_uncased_model_path():
    bert_model_base = os.path.join("model", "bert-base-uncased")
    if glob.glob(
        os.path.join(bert_model_base, "**/model.safetensors"), recursive=True
    ):
        # print("grounding-dino is using model/bert-base-uncased")
        return bert_model_base
    return "bert-base-uncased"


def load_sam_model(model_name):
    sam2_checkpoint_path = get_local_filepath(
        sam_model_list[model_name]["model_url"], sam_model_dir_name
    )
    model_file_name = os.path.basename(sam2_checkpoint_path)
    model_file_name = model_file_name.replace("2.1", "2_1")
    model_type = model_file_name.split(".")[0]

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    config_path = "sam2_configs"
    initialize(version_base=None, config_path=config_path)
    model_cfg = f"{model_type}.yaml"

    sam_device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = build_sam2(model_cfg, sam2_checkpoint_path, device=sam_device)
    sam.model_name = model_file_name
    return sam


def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    destination = os.path.join("model", local_file_name)
    if destination:
        # logger.warn(f"using extra model: {destination}")
        return destination

    folder = os.path.join("model", dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        logger.warn(f"downloading {url} to {destination}")
        download_url_to_file(url, destination)
    return destination


def load_groundingdino_model(model_name):
    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(
            groundingdino_model_list[model_name]["config_url"],
            groundingdino_model_dir_name,
        ),
    )

    if dino_model_args.text_encoder_type == "bert-base-uncased":
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()

    dino = local_groundingdino_build_model(dino_model_args)
    checkpoint = torch.load(
        get_local_filepath(
            groundingdino_model_list[model_name]["model_url"],
            groundingdino_model_dir_name,
        ),
    )
    dino.load_state_dict(
        local_groundingdino_clean_state_dict(checkpoint["model"]), strict=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino.to(device=device)
    dino.eval()
    return dino


def list_groundingdino_model():
    return list(groundingdino_model_list.keys())


def groundingdino_predict(dino_model, image, prompt, threshold):
    def load_dino_image(image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def get_grounding_output(model, image, caption, box_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        return boxes_filt.cpu()

    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt = get_grounding_output(dino_model, dino_image, prompt, threshold)
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt


def create_pil_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        output_masks.append(Image.fromarray(np.any(mask, axis=0)))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_images.append(Image.fromarray(image_np_copy))
    return output_images, output_masks


def create_tensor_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_image, output_mask = split_image_mask(Image.fromarray(image_np_copy))
        output_masks.append(output_mask)
        output_images.append(output_image)
    return (output_images, output_masks)


def split_image_mask(image):
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if "A" in image.getbands():
        mask = np.array(image.getchannel("A")).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image_rgb, mask)


def sam_segment(sam_model, image, boxes):
    if boxes.shape[0] == 0:
        return None
    predictor = SAM2ImagePredictor(sam_model)
    image_np = np.array(image)
    image_np_rgb = image_np[..., :3]
    predictor.set_image(image_np_rgb)
    # transformed_boxes = predictor.transform.apply_boxes_torch(
    #     boxes, image_np.shape[:2])
    sam_device = "cuda" if torch.cuda.is_available() else "cpu"
    masks, scores, _ = predictor.predict(
        point_coords=None, point_labels=None, box=boxes, multimask_output=False
    )
    # print("scores: ", scores)
    # print("masks shape before any modification:", masks.shape)
    if masks.ndim == 3:
        masks = np.expand_dims(masks, axis=0)
    # print("masks shape after ensuring 4D:", masks.shape)
    masks = np.transpose(masks, (1, 0, 2, 3))
    return create_tensor_output(image_np, masks, boxes)

def sam_segment_points(sam_model, image, prompt_points):
    predictor = SAM2ImagePredictor(sam_model)
    image_np = np.array(image)
    image_np_rgb = image_np[..., :3]
    predictor.set_image(image_np_rgb)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(prompt_points) != 0:
        points_value = [p for p, _ in prompt_points]
        points = torch.Tensor(points_value).to(device).unsqueeze(1)
        lables_value = [int(l) for _, l in prompt_points]
        labels = torch.Tensor(lables_value).to(device).unsqueeze(1)
        transformed_points = points
    else:
        transformed_points, labels = None, None

    input_points = np.array([(p[0][0], p[0][1]) for p in prompt_points])
    input_labels = np.array([p[1] for p in prompt_points])
    
    masks, scores, _ = predictor.predict(
        point_coords=input_points, point_labels=input_labels, box=None, multimask_output=False
    )
    
    mask = masks[0]    
    mask_all = np.ones((image_np_rgb .shape[0], image_np_rgb .shape[1], 3))
    # color_mask = np.random.random((1, 3)).tolist()[0]
    for i in range(3):
        mask_all[mask == True, i] = 1
        mask_all[mask == False, i] = 0.3
    img = image_np_rgb * mask_all / 255
    gc.collect()
    torch.cuda.empty_cache()
    
    mask_all = np.zeros((image_np_rgb.shape[0], image_np_rgb.shape[1]))
    mask_all[mask == True] = 1
    masked_image = np.zeros((image_np_rgb.shape[0], image_np_rgb.shape[1], 4))
    masked_image[:,:,0:3] = image_np_rgb  / 255
    masked_image[:,:,3] = mask_all

    return img, masks[0]


_grounding_segment_cache = None
_point_segment_cache = None


def get_grounding_segment() -> "GroundingDinoSAM2Segment":
    global _grounding_segment_cache
    if _grounding_segment_cache is None:
        print('[sam2] Loading GroundingDINO + SAM2 models...', flush=True)
        _grounding_segment_cache = GroundingDinoSAM2Segment()
        print('[sam2] Models ready', flush=True)
    return _grounding_segment_cache

def get_point_segment() -> "SAM2PointSegment":
    global _point_segment_cache
    if _point_segment_cache is None:
        print('[sam2] Loading SAM2 point-seg model...', flush=True)
        _point_segment_cache = SAM2PointSegment()
        print('[sam2] Point-seg model ready', flush=True)
    return _point_segment_cache


class GroundingDinoSAM2Segment:
    def __init__(self):
        self.grounding_dino_model = load_groundingdino_model("GroundingDINO_SwinB (938MB)")
        self.sam_model = load_sam_model("sam2_1_hiera_large.pt")

    def predict(self, image, threshold, prompt = "top person"):
        res_images = []
        res_masks = []
        for item in image:
            item = Image.fromarray(
                item
            ).convert("RGBA")
            boxes = groundingdino_predict(self.grounding_dino_model, item, prompt, threshold)
            if boxes.shape[0] == 0:
                break
            (images, masks) = sam_segment(self.sam_model, item, boxes)
            res_images.extend(images)
            res_masks.extend(masks)
        if len(res_images) == 0:
            print("WARN: Empty")
            _, height, width, _ = image.size()
            empty_mask = torch.zeros(
                (1, height, width), dtype=torch.uint8, device="cpu"
            )
            return (empty_mask, empty_mask)
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0)) # Image, Mask


class SAM2PointSegment:
    def __init__(self):
        self.sam_model = load_sam_model("sam2_1_hiera_large.pt")

    def predict(self, image, prompt_points):
        item = Image.fromarray(
            image
        ).convert("RGBA")
        return sam_segment_points(self.sam_model, item, prompt_points)

