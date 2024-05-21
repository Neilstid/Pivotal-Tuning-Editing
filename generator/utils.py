from typing import Any, Callable, List, Dict, Union
import os
import inspect

import torch
import numpy as np
from tqdm import tqdm
import torchvision
from PIL import Image

from decoratorhelper.type_decorator import load_configuration
from .stylegan2.model import (
    Generator as StyleGAN2_Generator,
    Discriminator as StyleGAN2_Discriminator,
)
from .stylegan3.model import Generator as StyleGAN3_Generator
from .stylegan2_ada.model import (
    Generator as StyleGAN2Ada_Generator,
    Discriminator as StyleGAN2Ada_Discriminator
)
from .styleganxl.model import SuperresGenerator as StyleGANXL_Generator
from .stylegan3 import dnnlib, legacy


def gen_w_space(
    generator: torch.nn.Module,
    need_c: bool = True,
    w_index: int = -1,
    device: Union[str, torch.device] = "auto",
    seed: int = None,
    latent: torch.Tensor = None,
    z_dim: int = 512,
    batch=1
):
    if device == "auto":
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    if need_c and latent is None:
        latent = [
            torch.from_numpy(np.random.RandomState(seed).randn(batch, z_dim)).to(device),
            torch.zeros([1, 0], device=device),
        ]
    elif latent is None:
        latent = torch.from_numpy(np.random.RandomState(seed).randn(1, z_dim)).to(
            device
        )

    # Generate the w space
    w = generator.mapping(
        *latent if need_c else latent
    )[w_index].detach()
    w.require_grad = False

    return w


def w_space_modifer(tensor):
    return tensor.squeeze(0)[0].detach()


def stylegan_img_convert(tensor, path):
    img = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(img[0].cpu().numpy(), "RGB").save(path)


def stylegan_invert_img_convert(tensor):
    img = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(img[0].cpu().numpy(), "RGB")


def keep_w(tensor: torch.Tensor, index: int = 0):
    return w_space_modifer(tensor[index]).cpu().numpy()


def keep_image(tensor: torch.Tensor, index: int = 2):
    return stylegan_invert_img_convert(tensor=tensor[index])


def keep_w_image(tensor: torch.Tensor, index_w: int = 0, index_image: int = 2):
    return keep_w(tensor, index_w), keep_image(tensor, index_image)


@load_configuration(
    r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\These\TheseUtils\generator\stylegan2\config_1024.json",
    "opts",
)
def load_stylegan2_generator(opts):
    generator = StyleGAN2_Generator(**opts["model_argument"])

    print("Loading StyleGAN2 from checkpoint: {}".format(opts["ckpt"]))
    checkpoint = torch.load(opts["ckpt"])
    generator.load_state_dict(checkpoint)

    device = opts["device"]
    generator.to(device)

    for param in generator.parameters():
        param.requires_grad = opts["require_grad"]

    generator.eval()

    return generator


@load_configuration(
    r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\These\TheseUtils\generator\stylegan3\config_t_ffhq_1024.json",
    "opts",
)
def load_stylegan3_generator(opts, from_pkl: bool = False):
    if from_pkl:
        with dnnlib.util.open_url(
            "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl"
        ) as f:
            generator = legacy.load_network_pkl(f)["G_ema"].to("cuda")

    else:
        generator = StyleGAN3_Generator(**opts["model_argument"])

        print("Loading StyleGAN3 from checkpoint: {}".format(opts["ckpt"]))
        checkpoint = torch.load(opts["ckpt"])
        generator.load_state_dict(checkpoint)

        device = opts["device"]
        generator.to(device)

        for param in generator.parameters():
            param.requires_grad = opts["require_grad"]

        generator.eval()

    return generator


@load_configuration(
    r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\These\TheseUtils\generator\stylegan2_ada\config_1024.json",
    "opts",
)
def load_stylegan2_ada_generator(opts: dict):
    generator = StyleGAN2Ada_Generator(**opts["model_argument"])

    print("Loading StyleGAN2-ADA from checkpoint: {}".format(opts["ckpt"]))
    checkpoint = torch.load(opts["ckpt"])
    generator.load_state_dict(checkpoint)

    device = opts["device"]
    generator.to(device)

    for param in generator.parameters():
        param.requires_grad = opts["require_grad"]

    generator.eval()

    return generator


@load_configuration(
    r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\These\TheseUtils\generator\styleganxl\config_r_256.json",
    "opts",
)
def load_styleganxl_generator(opts):
    generator = StyleGANXL_Generator(**opts["model_argument"])

    print("Loading StyleGAN-XL from checkpoint: {}".format(opts["ckpt"]))
    checkpoint = torch.load(opts["ckpt"])
    generator.load_state_dict(checkpoint)

    device = opts["device"]
    generator.to(device)

    for param in generator.parameters():
        param.requires_grad = opts["require_grad"]

    generator.eval()

    return generator
