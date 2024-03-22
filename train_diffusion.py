import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    cuda_device_id = torch.cuda.current_device()
    print("using gpu: ", cuda_device_id)
    print("Current GPU device name:", torch.cuda.get_device_name(cuda_device_id))


    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config)

    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=2,3 python train_diffusion.py --config allweather.yml --resume "/root/autodl-tmp/ddpm/ckpts/111AllWeather_ddpm.pth.tar" --sampling_timesteps 25 --image_folder /root/autodl-tmp/ddpm/image/ --seed 61
# CUDA_VISIBLE_DEVICES=3 python train_diffusion.py --config ntire64_64.yml --resume "/root/autodl-tmp/ddpm/ckpts/size_64_ch_64_epochs_300.pth.tar" --sampling_timesteps 25 --image_folder /root/autodl-tmp/ddpm/image/ --seed 61
# CUDA_VISIBLE_DEVICES=1,2 python train_diffusion.py --config ntire64_96.yml --resume "/root/autodl-tmp/ddpm/ckpts/size_64_ch_96_epochs_300.pth.tar" --sampling_timesteps 25 --image_folder /root/autodl-tmp/ddpm/image/ --seed 61
# CUDA_VISIBLE_DEVICES=1,2,3,4 python train_diffusion.py --config ntire64_128.yml --resume "/root/autodl-tmp/ddpm/ckpts/size_64_ch_128_epochs_300.pth.tar" --sampling_timesteps 25 --image_folder /root/autodl-tmp/ddpm/image/ --seed 61
# CUDA_VISIBLE_DEVICES=0 python train_diffusion.py --config ntire96_64.yml --resume "/root/autodl-tmp/ddpm/ckpts/size_96_ch_64_epochs_300.pth.tar" --sampling_timesteps 25 --image_folder /root/autodl-tmp/ddpm/image/ --seed 61
# CUDA_VISIBLE_DEVICES=0,3 python train_diffusion.py --config ntire96_96.yml --resume "/root/autodl-tmp/ddpm/ckpts/size_96_ch_96_epochs_300.pth.tar" --sampling_timesteps 25 --image_folder /root/autodl-tmp/ddpm/image/ --seed 61
# CUDA_VISIBLE_DEVICES=1,2 python train_diffusion.py --config ntire96_128.yml --resume "/root/autodl-tmp/ddpm/ckpts/size_96_ch_128_epochs_300.pth.tar" --sampling_timesteps 25 --image_folder /root/autodl-tmp/ddpm/image/ --seed 61
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_diffusion.py --config ntire.yml --resume "/root/autodl-tmp/ddpm/ckpts/size_64_ch_128_epochs_2660.pth.tar" --sampling_timesteps 25 --image_folder /root/autodl-tmp/ddpm/image/ --seed 61
    