<div align="center">
<h1>DreamDPO: Aligning Text-to-3D Generation with Human Preferences via Direct Preference Optimization</h1>

[**Zhenglin Zhou**](https://scholar.google.com/citations?user=6v7tOfEAAAAJ) · [**Xiaobo Xia<sup>*</sup>**](https://xiaoboxia.github.io/) · [**Fan Ma**](https://flowerfan.site/) · [**Hehe Fan**](https://hehefan.github.io/) · [**Yi Yang<sup>*</sup>**](https://scholar.google.com/citations?user=RMSuNFwAAAAJ) · [**Tat-Seng Chua**](https://www.chuatatseng.com/) 

<a href='https://zhenglinzhou.github.io/DreamDPO-ProjectPage/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2502.04370'><img src='https://img.shields.io/badge/Technique-Report-red'></a>

</div>

## Installation
1. Clone the repo and create conda environment
```shell
git clone https://github.com/ZhenglinZhou/DreamDPO.git
cd DreamDPO
conda create -n dreamdpo python=3.9
conda activate dreamdpo
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.23.post1
pip install ninja
git clone https://github.com/bytedance/MVDream extern/MVDream
pip install -e extern/MVDream 
pip install -r requirements.txt
```

2. Install the pretrained reward model

* Step1: Install the packages as following:

| Reward Model                                        | Package                                                         | Checkpoints                                              |
|-----------------------------------------------------|-----------------------------------------------------------------|----------------------------------------------------------|
| [HPSv2](https://github.com/tgxs002/HPSv2)           | `pip install hpsv2`                                             | `xswu/HPSv2`<br/>`laion/CLIP-ViT-H-14-laion2B-s32B-b79K` |
| [ImageReward](https://github.com/THUDM/ImageReward) | `pip install image-reward`                                      | `THUDM/ImageReward`                                      |
| [BRIQUE](https://pypi.org/project/brisque)          | `pip install brisque`<br/>`pip install libsvm-official==3.30.0` | -                                                        |
| [Reward3D](https://github.com/liuff19/DreamReward)  | `pip install fairscale`                                         | `yejunliang23/Reward3D`                                  |
| [Qwen](https://help.aliyun.com/zh/model-studio/obtain-api-key-app-id-and-workspace-id?spm=a2c4g.11186623.0.i1)                                            | `pip install kiui`<br/>`pip install dashcope`                       | -                                                        |
* Step2: Download model checkpoints:
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download ${Model CheckPoints} --local-dir model_weights/${Reward Model}
```
* Step3: Organize them look like this:
```shell
|-- model_weights
    |-- HPSv2 # HPSv2
        |-- HPS_v2.1_compressed.pt
    |-- ImageReward # ImageReward
        |-- ImageReward.pt
        |-- med_config.json
    |-- Reward3D # Reward3D
        |-- Reward3D_Scorer.pt
    |-- CLIP-ViT-H-14-laion2B-s32B-b79K # HPSv2
        |-- open_clip_pytorch_model.bin
```
* Step4 (Optional): Use large multi-module model (e.g., Qwen):
To use it, you should [create an Alibaba Cloud account](https://help.aliyun.com/zh/model-studio/obtain-api-key-app-id-and-workspace-id?spm=a2c4g.11186623.0.i1) and create a Dashscope API key to fill in the DASHSCOPE_API_KEY field in the config.yaml file. Change the MODEL field from OpenAI to Qwen as well.

## Usage
1. DreamDPO with different reward models, including `hpsv2-score`, `imagereward-score`, `brique-score`, `reward3d-score`:
```bash
python3 launch.py --config configs/dreamdpo/mvdream-sd21-reward.yaml \
--train --gpu 0 \
system.prompt_processor.prompt="A pair of hiking boots caked with mud at the doorstep of a cabin" \
system.guidance.beta_dpo=0.01 system.guidance.reward_model="hpsv2-score"
```
2. DreamDPO with large multi-module model (e.g., Qwen)
```bash
python3 launch.py --config configs/dreamdpo/mvdream-sd21-lmm.yaml \
--train --gpu 0 \
system.prompt_processor.prompt="A pair of hiking boots caked with mud at the doorstep of a cabin" \
system.guidance.ai_start_iter=1200 system.guidance.ai_prob=0.5
```

## Acknowledgments
* This code is built on the [threestudio-project](https://github.com/threestudio-project/threestudio), [MVDream](https://github.com/bytedance/MVDream-threestudio/tree/main), and [AppAgent](https://github.com/TencentQQGYLab/AppAgent). Thanks to the maintainers!
* We also borrow code and model from [HPSv2](https://github.com/tgxs002/HPSv2), [ImageReward](https://github.com/THUDM/ImageReward), [DreamReward](https://github.com/liuff19/DreamReward). Thanks to the authors for their human preference alignment models.



## Citation
If you find DreamDPO useful for your research and applications, please cite us using this BibTeX:
```bibtex
@inproceedings{zhou2025dreamdpo,
      title={DreamDPO: Aligning Text-to-3D Generation with Human Preferences via Direct Preference Optimization}, 
      author={Zhenglin Zhou and Xiaobo Xia and Fan Ma and Hehe Fan and Yi Yang and Tat-Seng Chua},
      booktitle={ICML},
      year={2025},
}
```