from dataclasses import dataclass, field

import os
import numpy as np
import os.path as osp
from kiui.utils import write_image

import torch

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject

from verifiers.Qwen.model import QwenModel
from verifiers.Qwen.promptv1 import PromptV1


@threestudio.register("qwen-score")
class QwenScore(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        # qwen-vl-plus / qwen2-vl-7b-instruct / qwen-vl-plus-latest / qwen-vl-max-latest
        qwen_model: str = 'qwen-vl-plus-latest'
        qwen_api_key: str = 'sk-0e5b8e14fc9b430e8ee5fb0cdd90bb24'
        temp_folder: str = "outputs/qwen_temp"

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Qwen ...")

        # qwen model: qwen2-vl-7b-instruct is free but not stability
        qwen_model = self.cfg.qwen_model
        qwen_api_key = self.cfg.qwen_api_key
        temp_folder = self.cfg.temp_folder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # initialize model
        self.tokenizer, self.model = self.init_model(qwen_model, qwen_api_key)

        self.text_input = None

        # initialize save folder
        if not osp.exists(temp_folder): os.makedirs(temp_folder)
        self.save_folder = temp_folder

    def preprocess(self, image):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # empirially to channel-last
        if len(image.shape) == 3 and image.shape[0] < image.shape[-1]:
            image = image.transpose(1, 2, 0)
        return image

    def init_model(self, qwen_model, qwen_api_key):
        mllm = QwenModel(api_key=qwen_api_key, model=qwen_model)
        prompter = PromptV1()
        return prompter, mllm

    def __call__(
            self, image, prompt,
            prompt_utils=None,
            elevation=None,
            azimuth=None,
            camera_distances=None,
    ):
        if self.text_input is None:
            self.text_input = self.tokenizer.get_prompt(prompt)

        scores = []
        for i, img in enumerate(image):
            img = self.preprocess(img)
            save_path = osp.join(self.save_folder, f"temp{i}.png")
            write_image(save_path, img)
            status, rsp = self.model.get_model_response(self.text_input, [save_path, ])
            scores.append(self.tokenizer.extract_score(rsp))
        print(scores)
        scores = torch.tensor(scores).to(self.device)
        return scores
