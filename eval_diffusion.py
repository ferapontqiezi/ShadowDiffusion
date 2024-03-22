import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from torchsummary import summary
#from models import DenoisingDiffusion, DiffusiveRestoration
from models import DenoisingDiffusion1, DiffusiveRestoration1, DenoisingDiffusion2, DiffusiveRestoration2


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Weather with Patch-Based Denoising Diffusion Models')
    #parser.add_argument("--config", type=str, required=True,
    #                    help="Path to the config file")
    parser.add_argument("--config1", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument("--config2", type=str, required=True,
                        help="Path to the config file")
    #parser.add_argument('--resume', default='', type=str,
    #                    help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument('--resume1', default='', type=str,
                    help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument('--resume2', default='', type=str,
                    help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width r that defines the overlap between patches")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps")
    parser.add_argument("--test_set", type=str, default='raindrop',
                        help="restoration test set options: ['raindrop', 'snow', 'rainfog']")
    parser.add_argument("--image_folder", default='./', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    #with open(os.path.join("configs", args.config), "r") as f:
    #    config = yaml.safe_load(f)
    #new_config = dict2namespace(config)
    with open(os.path.join("configs", args.config1), "r") as f:
        config1 = yaml.safe_load(f)
    new_config1 = dict2namespace(config1)
    with open(os.path.join("configs", args.config2), "r") as f:
        config2 = yaml.safe_load(f)
    new_config2 = dict2namespace(config2)

    #return args, new_config
    return args, new_config1, new_config2


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
    args, config1, config2 = parse_args_and_config()
    
    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config1.device = device
    config2.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data1 loading
    print("=> using dataset '{}'".format(config1.data.dataset))
    DATASET1 = datasets.__dict__[config1.data.dataset](config1)
    _, val_loader1 = DATASET1.get_loaders(parse_patches=False, validation=args.test_set)
    # data2 loading
    print("=> using dataset '{}'".format(config2.data.dataset))
    DATASET2 = datasets.__dict__[config2.data.dataset](config2)
    _, val_loader2 = DATASET2.get_loaders(parse_patches=False, validation=args.test_set)

    # create model
    print("=> creating denoising-diffusion model with wrapper...")


    #diffusion = DenoisingDiffusion(args, config)
    #model = DiffusiveRestoration(diffusion, args, config)  
    #model.restore(val_loader, validation=args.test_set, r=args.grid_r)


    # process model1
    diffusion1 = DenoisingDiffusion1(args, config1)
    model1 = DiffusiveRestoration1(diffusion1, args, config1)  
    model1.restore(val_loader1, validation=args.test_set, r=args.grid_r)

    # process model2
    diffusion2 = DenoisingDiffusion2(args, config2)
    model2 = DiffusiveRestoration2(diffusion2, args, config2)
    model2.restore(val_loader2, validation=args.test_set, r=args.grid_r)



if __name__ == '__main__':
    main()


# CUDA_VISIBLE_DEVICES=3 python eval_diffusion.py --config "allweather.yml" --resume '/root/autodl-tmp/ddpm/ckpts/AllWeather_ddpm_1600.pth.tar' --test_set 'snow' --sampling_timesteps 25 --grid_r 8
# CUDA_VISIBLE_DEVICES=2 python eval_diffusion.py --config "allweather.yml" --resume '/root/autodl-tmp/ddpm/ckpts/AllWeather_ddpm_1600.pth.tar' --test_set 'snow' --sampling_timesteps 25 --grid_r 4
# CUDA_VISIBLE_DEVICES=3 python eval_diffusion.py --config "ntire64_64.yml" --resume '/root/autodl-tmp/ddpm/ckpts/size_64_ch_64_epochs_300.pth.tar' --test_set 'snow' --sampling_timesteps 250 --grid_r 16
# CUDA_VISIBLE_DEVICES=2 python eval_diffusion.py --config "ntire.yml" --resume '/root/autodl-tmp/ddpm/ckpts/A' --test_set 'snow' --sampling_timesteps 250 --grid_r 16
# CUDA_VISIBLE_DEVICES=1 python eval_diffusion.py --config "ntire.yml" --resume '/root/autodl-tmp/ddpm/ckpts/size_64_ch_128_epochs_2660.pth.tar' --test_set 'snow' --sampling_timesteps 25 --grid_r 16
