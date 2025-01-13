from functools import partial
import os
import argparse
import yaml
import torch

from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion_LLVIP_optim import create_sampler
from util.logger import get_logger
import cv2
import numpy as np
from PIL import Image
from util.wavalets import Wavalets
from conditions.conditions import Sobel
from torch.utils.tensorboard import SummaryWriter

import warnings

warnings.filterwarnings('ignore')


def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='configs/model_config_imagenet.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--input_dir', type=str, default='./input')
    parser.add_argument('--record', type=bool, default=True)
    args = parser.parse_args()

    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    diffusion_config['timestep_respacing'] = 200

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()
    params_stat = 0.
    for param in model.parameters():
        param.requires_grad = False
        params_stat += param.numel()
    print(params_stat)
    num_classes = 19

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model)

    # Working directory
    test_folder = args.input_dir
    out_path = args.output_dir
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['recon', 'progress', 'recon_color']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
    logs_dir = os.path.join(out_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(logs_dir)

    i = 0

    for img_name in sorted(os.listdir(os.path.join(test_folder, "ir"))):
        inf_img = image_read(os.path.join(test_folder, "ir", img_name), mode='RGB')
        vis_img = image_read(os.path.join(test_folder, "vi", img_name), mode='RGB')
        h, w = inf_img.shape[0], inf_img.shape[1]
        vis_rgb = vis_img
        scale = 32
        # 控制最大显存,短边为512
        if min(h, w) > 512:
            if w > h:
                inf_img = cv2.resize(inf_img, [int(512 * w / h), 512])
                vis_img = cv2.resize(vis_img, [int(512 * w / h), 512])
            else:
                inf_img = cv2.resize(inf_img, [512, int(512 * w / h)])
                vis_img = cv2.resize(vis_img, [512, int(512 * w / h)])

        new_h, new_w = inf_img.shape[0], inf_img.shape[1]
        pix_margin = new_h % 32
        if pix_margin != 0:
            new_h = new_h - pix_margin
        pix_margin = new_w % 32
        if pix_margin != 0:
            new_w = new_w - pix_margin
        inf_img = cv2.resize(inf_img, [new_w, new_h])
        vis_img = cv2.resize(vis_img, [new_w, new_h])
        vis_img = vis_img[np.newaxis, ...] / 255.0
        inf_img = inf_img[np.newaxis, ...] / 255.0

        inf_img = inf_img.transpose((0, 3, 1, 2))
        vis_img = vis_img.transpose((0, 3, 1, 2))
        inf_img = inf_img * 2 - 1
        vis_img = vis_img * 2 - 1
        inf_img = ((torch.FloatTensor(inf_img))).to(device)
        vis_img = ((torch.FloatTensor(vis_img))).to(device)

        logger.info(f"Inference for image {i}")

        # Sampling
        seed = 3407
        torch.manual_seed(seed)
        x_start = torch.randn(vis_img.shape, device=device)
        # add
        wavalets = Wavalets(J=1, devices=device)
        sobel = Sobel(device)
        sample = sample_fn(x_start=x_start, record=args.record, I=inf_img, V=vis_img, save_root=out_path,
                               img_index=os.path.splitext(img_name)[0], wavalets=wavalets, sobel=sobel, writer=writer)

        sample = sample.detach().cpu().squeeze().numpy()
        sample = np.transpose(sample, (1, 2, 0))
        sample_bak = sample
        sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
        sample = (sample * 255).astype(np.uint8)

        im = Image.fromarray(sample)
        im.save(os.path.join(os.path.join(out_path, 'recon'), "{}".format(img_name)))

        sample = cv2.cvtColor(sample_bak, cv2.COLOR_RGB2YCrCb)[..., 0]
        sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
        sample = (sample * 255).astype(np.uint8)
        sample = cv2.resize(sample, (w, h))
        cv2.imwrite(os.path.join(os.path.join(out_path, 'recon_color_1'), "{}".format(img_name)), sample)

        vis_rgb = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2YCrCb)
        new_fused_img = np.stack([sample, vis_rgb[..., 1], vis_rgb[..., 2]], -1)
        new_fused_img = cv2.cvtColor(new_fused_img, cv2.COLOR_YCrCb2RGB)
        sample = ((new_fused_img)).astype(np.uint8)
        cv2.imwrite(os.path.join(os.path.join(out_path, 'recon_color'), "{}".format(img_name)), cv2.cvtColor(new_fused_img, cv2.COLOR_RGB2BGR))
        i = i + 1
        writer.flush()
    writer.close()


