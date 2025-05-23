import os
import numpy as np
import os.path as osp
from PIL import Image
from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *

import verifiers.Reward3D as r3d

try:
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class Reward3DPreprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __call__(self, img):
        # input image should be with [B, 3, 256, 256] in [0, 1]
        image = self.transform(img)
        return image


@threestudio.register("reward3d-score")
class Reward3DScore(BaseObject):
    @dataclass
    class Config:
        reward3d_path: str = "model_weights/Reward3D"
        med_config_path: str = "verifiers/Reward3D/scripts/med_config.json"
        alg_type: str = "Reward3D_Scorer"  # Reward3D_Scorer or Reward3D_CrossViewFusion

    cfg: Config

    def configure(self):
        # Model Initialization
        reward3d_path = osp.join(self.cfg.reward3d_path, f"{self.cfg.alg_type}.pt")
        med_config_path = self.cfg.med_config_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        Reward3D = self.init_model(reward3d_path, med_config_path)
        self.Reward3D = Reward3D.to(self.device)
        self.Reward3D.eval()

        self.preprocess = Reward3DPreprocess()
        for p in self.Reward3D.parameters():
            p.requires_grad_(False)

    def init_model(self, reward3d_path, med_config_path):
        state_dict = torch.load(reward3d_path)

        if self.cfg.alg_type == "Reward3D_Scorer":
            Reward3D = r3d.Reward3D_(device=self.device, med_config=med_config_path)
        elif self.cfg.alg_type == "Reward3D_CrossViewFusion":
            Reward3D = r3d.Reward3D(device=self.device, med_config=med_config_path)
        else:
            raise NotImplementedError

        msg = Reward3D.load_state_dict(state_dict, strict=False)
        print(msg)
        print(self.cfg.reward3d_path)
        return Reward3D

    def __call__(
            self, images, text,
            prompt_utils,
            elevation: Float[Tensor, "B"],
            azimuth: Float[Tensor, "B"],
            camera_distances: Float[Tensor, "B"],
    ):
        if isinstance(images, list):
            images = [torch.from_numpy(img) for img in images]
            images = torch.stack(images)

        if not hasattr(self, 'rm_input_ids'):
            self.rm_input_ids = []
            self.rm_attention_mask = []

            prompts_vds = prompt_utils.prompts_vd
            for idx in range(len(prompts_vds)):
                prompts_vd = prompts_vds[idx]
                g = self.Reward3D.blip.tokenizer(
                    prompts_vd,
                    padding='max_length',
                    truncation=True,
                    max_length=100,
                    return_tensors="pt"
                )
                self.rm_input_ids.append(g.input_ids)
                self.rm_attention_mask.append(g.attention_mask)
        bs = images.shape[0]
        images = images / 255.0
        images = self.preprocess(images).to(self.device)

        direction_idx = torch.zeros_like(elevation, dtype=torch.long)
        for d in prompt_utils.directions:
            direction_idx[
                d.condition(elevation, azimuth, camera_distances)
            ] = prompt_utils.direction2idx[d.name]
        rm_input_ids = torch.cat([self.rm_input_ids[idx] for idx in direction_idx]).to(self.device)
        rm_attention_mask = torch.cat([self.rm_attention_mask[idx] for idx in direction_idx]).to(self.device)

        rewards = self.Reward3D(images, rm_input_ids, rm_attention_mask)
        rewards = rewards[:, 0].repeat(bs)
        return rewards
