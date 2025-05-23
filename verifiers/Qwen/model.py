"""
ref: https://github.com/TencentQQGYLab/AppAgent/blob/2c1900422caf6f9e94e96d5dd984b530e5a5fbf8/scripts/model.py#L73
"""
import dashscope
from typing import List
from http import HTTPStatus
from abc import abstractmethod


class BaseModel:
    def __init__(self):
        pass

    @abstractmethod
    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        pass


class QwenModel(BaseModel):
    def __init__(self, api_key: str, model: str):
        super().__init__()
        self.model = model
        dashscope.api_key = api_key

    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        content = [{
            "text": prompt
        }]
        for img in images:
            img_path = f"file://{img}"
            content.append({
                "image": img_path
            })
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        response = dashscope.MultiModalConversation.call(model=self.model, messages=messages)
        if response.status_code == HTTPStatus.OK:
            return True, response.output.choices[0].message.content[0]["text"]
        else:
            return False, response.message




if __name__ == '__main__':
    api_key = 'xx'
    model = "qwen-vl-plus"
    mllm = QwenModel(api_key=api_key, model=model)
    prompt = "Please describe this image."
    image = "/path/to/image.png"
    status, rsp = mllm.get_model_response(prompt, [image, ])
    print(rsp)
