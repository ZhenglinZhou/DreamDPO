from dataclasses import dataclass
import numpy as np
from brisque import BRISQUE
from PIL import Image

import torch

import threestudio
from threestudio.utils.base import BaseObject


@threestudio.register("brique-score")
class BRISQUEScore(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pass

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading BRIQUEScore ...")
        self.model = BRISQUE(url=False)

    def preprocess_image(self, image):
        if isinstance(image, torch.Tensor):
            image = image.permute(0, 2, 3, 1)
            image = image.cpu().numpy()

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        return image

    def __call__(
            self, image, prompt,
            prompt_utils=None,
            elevation=None,
            azimuth=None,
            camera_distances=None,
    ):
        processed_image = self.preprocess_image(image)
        scores = []
        for i in range(processed_image.shape[0]):
            score = self.model.score(processed_image[i])
            scores.append((100 - score) / 100)
        return torch.tensor(scores)


if __name__ == '__main__':
    # wget https://www.mathworks.com/help/examples/images/win64/CalculateBRISQUEScoreUsingCustomFeatureModelExample_01.png
    img_path = "CalculateBRISQUEScoreUsingCustomFeatureModelExample_01.png"
    img = Image.open(img_path)
    img_rgb = img.convert('RGB')
    ndarray = np.array(img_rgb)
    obj = BRISQUE(url=False)
    score = obj.score(img=ndarray)
    print(score)
