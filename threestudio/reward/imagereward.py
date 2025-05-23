import os.path as osp
from PIL import Image
import ImageReward as RM
from dataclasses import dataclass, field

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage

import threestudio
from threestudio.utils.base import BaseObject

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        ToPILImage(),
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


@threestudio.register("imagereward-score")
class ImageRewardScore(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        model_weights: str = "model_weights/ImageReward"

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading ImageReward ...")
        model_weights = self.cfg.model_weights

        model_path = osp.join(model_weights, 'ImageReward.pt')
        config_path = osp.join(model_weights, 'med_config.json')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # initialize model
        self.model = self.init_model(model_path, config_path)

        # preprocess
        self.image_size = 224
        # self.interpolation = tf.InterpolationMode.BICUBIC
        self.preprocess = _transform(self.image_size)

        self.text_input = None

    # def preprocess(self, image):
    #     # resize
    #     image = tf.resize(image, self.image_size, self.interpolation)
    #     image = tf.center_crop(image, self.image_size)
    #     # normalize
    #     mean = torch.as_tensor(OPENAI_DATASET_MEAN, dtype=image.dtype, device=image.device)
    #     std = torch.as_tensor(OPENAI_DATASET_STD, dtype=image.dtype, device=image.device)
    #     image = tf.normalize(image, mean, std)
    #     # image = image.sub_(mean.reshape(-1, 1, 1)).div_(std.reshape(-1, 1, 1))
    #     return image

    def init_model(self, model_path, config_path):
        state_dict = torch.load(model_path, map_location='cpu')
        med_config = config_path
        model = RM.ImageReward(device=self.device, med_config=med_config).to(self.device)
        msg = model.load_state_dict(state_dict, strict=False)
        print("checkpoint loaded")
        model.eval()
        return model

    def __call__(
            self, image, prompt,
            prompt_utils=None,
            elevation=None,
            azimuth=None,
            camera_distances=None,
    ):
        if self.text_input is None:
            self.text_input = self.model.blip.tokenizer(
                prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt"
            ).to(self.device)
            # del self.model.blip.tokenizer

        txt_set = []
        for img in image:
            img = self.preprocess(img)
            img = img.unsqueeze(0).to(self.device)
            image_embed = self.model.blip.visual_encoder(img)
            # text encode cross attention with image
            image_atts = torch.ones(image_embed.size()[:-1], dtype=torch.long).to(self.device)
            text_output = self.model.blip.text_encoder(self.text_input.input_ids,
                                                       attention_mask=self.text_input.attention_mask,
                                                       encoder_hidden_states=image_embed,
                                                       encoder_attention_mask=image_atts,
                                                       return_dict=True,
                                                       )
            txt_set.append(text_output.last_hidden_state[:, 0, :])

        txt_features = torch.cat(txt_set, 0).float()
        # txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.model.mlp(txt_features)
        rewards = (rewards - self.model.mean) / self.model.std

        return rewards.detach().squeeze()
