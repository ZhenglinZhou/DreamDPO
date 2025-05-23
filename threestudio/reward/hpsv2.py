from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

import threestudio
from threestudio.utils.base import BaseObject

from verifiers.HPSv2.hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from verifiers.HPSv2.hpsv2.src.open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


class HPSv2Preprocess(nn.Module):
    def __init__(self, model):
        super().__init__()
        mean = getattr(model.visual, 'image_mean', OPENAI_DATASET_MEAN)
        if not isinstance(mean, (list, tuple)):
            mean = (mean,) * 3
        std = getattr(model.visual, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(std, (list, tuple)):
            std = (std,) * 3
        self.mean = mean
        self.std = std

        self.interpolation = tf.InterpolationMode.BICUBIC
        self.image_size = model.visual.image_size

    def __call__(self, img):
        # resize
        img = tf.resize(img, self.image_size, self.interpolation)
        # normalize
        mean = torch.as_tensor(self.mean, dtype=img.dtype, device=img.device)
        std = torch.as_tensor(self.std, dtype=img.dtype, device=img.device)
        img = img.sub_(mean.reshape(1, -1, 1, 1)).div_(std.reshape(1, -1, 1, 1))
        return img

@threestudio.register("hpsv2-score")
class HPSv2Score(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        hpsv2_path: str = "model_weights/HPSv2/HPS_v2.pt"
        clip_path: str = "model_weights/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading HPSv2 ...")
        hpsv2_path, clip_path = self.cfg.hpsv2_path, self.cfg.clip_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # initialize model
        self.tokenizer, self.model = self.init_model(hpsv2_path, clip_path)
        self.preprocess = HPSv2Preprocess(self.model)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.text_input = None

    def init_model(self, hpsv2_path, clip_path):
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
            checkpoint_local_path=clip_path,
        )
        hpsv2_cp = torch.load(hpsv2_path, map_location=self.device)
        model.load_state_dict(hpsv2_cp['state_dict'])
        hpsv2_tokenizer = get_tokenizer('ViT-H-14')
        hpsv2_model = model.to(self.device)
        hpsv2_model.eval()
        return hpsv2_tokenizer, hpsv2_model

    def __call__(
            self, image, prompt,
            prompt_utils=None,
            elevation=None,
            azimuth=None,
            camera_distances=None,
    ):
        if self.text_input is None:
            batch_size = image.shape[0]
            self.text_input = self.tokenizer([prompt] * batch_size).to(self.device, non_blocking=True)
            del self.tokenizer

        image = self.preprocess(image)
        outputs = self.model(image, self.text_input)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits_per_image = image_features @ text_features.T
        hps_score = torch.diagonal(logits_per_image)
        return hps_score
