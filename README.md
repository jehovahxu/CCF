
# Conditional-Controllable-Image-Fusion

This is the official implementation of Conditional-Controllable-Image-Fusion(NeurIPS 2024)

![](imgs/frame.png)
## Abstract
Image fusion aims to integrate complementary information from multiple input images acquired through various sources to synthesize a new fused image. Existing methods usually employ distinct constraint designs tailored to specific scenes, forming fixed fusion paradigms. However, this data-driven fusion approach is challenging to deploy in varying scenarios, especially in rapidly changing environments. To address this issue, we propose a conditional controllable fusion (CCF) framework for general image fusion tasks without specific training. Due to the dynamic differences of different samples, our CCF employs specific fusion constraints for each individual in practice. Given the powerful generative capabilities of the denoising diffusion model, we first inject the specific constraints into the pre-trained DDPM as adaptive fusion conditions. The appropriate conditions are dynamically selected to ensure the fusion process remains responsive to the specific requirements in each reverse diffusion stage. Thus, CCF enables conditionally calibrating the fused images step by step. Extensive experiments validate our effectiveness in general fusion tasks across diverse scenarios against the competing methods without additional training.

## Environment Installation 
```
# create virtual environment
conda create -n CCF python=3.8.10
conda activate CCF
# select pytorch version yourself
# install CCF requirements
pip install -r requirements.txt
```

## Download pre-trained models
The pre-trained checkpoints "256x256_diffusion_uncond.pt" can download from [guided-diffusion](https://github.com/openai/guided-diffusion)

## Inference (Sampling)
```
python sample_LLVIP.py
```

## Citation
```
@article{cao2024conditional,
  title={Conditional Controllable Image Fusion},
  author={Cao, Bing and Xu, Xingxin and Zhu, Pengfei and Wang, Qilong and Hu, Qinghua},
  journal={arXiv preprint arXiv:2411.01573},
  year={2024}
}


````
