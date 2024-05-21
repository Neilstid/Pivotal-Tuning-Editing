from typing import Any, Callable, List, Dict, Union
import os
import inspect

import torch
import numpy as np
from tqdm import tqdm
import torchvision
from PIL import Image

from decorator.type_decorator import load_configuration
from files.context import ContextualPath
from .stylegan2.model import Generator as StyleGAN2_Generator
from .stylegan3.model import Generator as StyleGAN3_Generator
from .stylegan2_ada.model import Generator as StyleGAN2Ada_Generator
from .styleganxl.model import SuperresGenerator as StyleGANXL_Generator
from .stylegan3 import dnnlib, legacy


def gen_wp_space(
    generator: torch.nn.Module, need_c: bool = True, w_index: int = -1,
    device: Union[str, torch.device] = "auto", dim=14, seed: Union[List[int], None] = None
):
    if seed is None:
        seed = np.random.randint(999999, size=(dim + 1))
    elif not isinstance(seed, list):
        seed = np.insert(np.random.randint(999999, size=(dim)), seed, 0)

    wplus = gen_w_space(
        generator=generator, need_c=need_c, w_index=w_index, device=device, seed=seed[0]
    )

    dim = max(wplus.shape[1], dim)
    for i in range(dim):
        try:
            seed_tmp = seed[i + 1]
        except IndexError:
            seed_tmp = np.random.randint(999999)

        wplus[:, i // dim] = gen_w_space(
            generator=generator, need_c=need_c, w_index=w_index, device=device, seed=seed_tmp
        )[:, 0]

    return wplus


def gen_w_space(
    generator: torch.nn.Module, need_c: bool = True, w_index: int = -1,
    device: Union[str, torch.device] = "auto", seed: int = None, latent: torch.Tensor = None
):
    if device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if need_c and latent is None:
        latent = [
            torch.from_numpy(np.random.RandomState(seed).randn(1, 512)).to(device),
            torch.zeros([1, 0], device=device)
        ]
    elif latent is None:
        latent = torch.from_numpy(np.random.RandomState(seed).randn(1, 512)).to(device)

    # Generate the w space
    w = generator(
        *latent if need_c else latent, input_is_latent=False,
        return_latents=True
    )[w_index].detach()
    w.require_grad = False

    return w


def w_space_modifer(tensor):
    return tensor.squeeze(0)[0].detach()


def wp_space_modifer(tensor):
    return tensor.squeeze(0).detach()


def z_space_modifer(tensor):
    return tensor.squeeze(0).detach().flatten()


def stylegan_img_convert(tensor, path):
    img = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(img[0].cpu().numpy(), 'RGB').save(path)


def stylegan_invert_img_convert(tensor):
    img = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(img[0].cpu().numpy(), 'RGB')


def keep_w(
    tensor: torch.Tensor, index: int = -1    
):
    return w_space_modifer(tensor[index]).cpu().numpy()


def keep_wp(
    tensor: torch.Tensor, index: int = -1    
):
    return wp_space_modifer(tensor[index]).cpu().numpy()


def keep_z(
    tensor: torch.Tensor, index: int = 0    
):
    return z_space_modifer(tensor[index]).cpu().numpy()


@load_configuration(r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\These\TheseUtils\generator\stylegan2\config_256.json", "opts")
def load_stylegan2_generator(opts):

    generator = StyleGAN2_Generator(
        **opts["model_argument"]
    )

    print('Loading StyleGAN2 from checkpoint: {}'.format(opts["ckpt"]))
    checkpoint = torch.load(opts["ckpt"])
    generator.load_state_dict(checkpoint)

    device = opts["device"]
    generator.to(device)

    for param in generator.parameters():
        param.requires_grad = opts["require_grad"]
    
    generator.eval()

    return generator


@load_configuration(r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\These\TheseUtils\generator\stylegan3\config_r_256.json", "opts")
def load_stylegan3_generator(opts, from_pkl: bool = False):


    if from_pkl:
        with ContextualPath(r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\These\TheseUtils\generator\stylegan3"):
            with dnnlib.util.open_url("https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl") as f:
                generator = legacy.load_network_pkl(f)['G_ema'].to("cuda")

    else:
        generator = StyleGAN3_Generator(
            **opts["model_argument"]
        )

        print('Loading StyleGAN3 from checkpoint: {}'.format(opts["ckpt"]))
        checkpoint = torch.load(opts["ckpt"])
        generator.load_state_dict(checkpoint)

        device = opts["device"]
        generator.to(device)

        for param in generator.parameters():
            param.requires_grad = opts["require_grad"]
        
        generator.eval()

    return generator


@load_configuration(r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\These\TheseUtils\generator\stylegan2_ada\config_1024.json", "opts")
def load_stylegan2_ada_generator(opts):


    generator = StyleGAN2Ada_Generator(
        **opts["model_argument"]
    )

    print('Loading StyleGAN2-ADA from checkpoint: {}'.format(opts["ckpt"]))
    checkpoint = torch.load(opts["ckpt"])
    generator.load_state_dict(checkpoint)

    device = opts["device"]
    generator.to(device)

    for param in generator.parameters():
        param.requires_grad = opts["require_grad"]
    
    generator.eval()

    return generator


@load_configuration(r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\These\TheseUtils\generator\styleganxl\config_r_256.json", "opts")
def load_styleganxl_generator(opts):


    generator = StyleGANXL_Generator(
        **opts["model_argument"]
    )

    print('Loading StyleGAN-XL from checkpoint: {}'.format(opts["ckpt"]))
    checkpoint = torch.load(opts["ckpt"])
    generator.load_state_dict(checkpoint)

    device = opts["device"]
    generator.to(device)

    for param in generator.parameters():
        param.requires_grad = opts["require_grad"]
    
    generator.eval()

    return generator


def modify_output(
    output: torch.Tensor, output_modification: Union[Callable, None, List[Callable]] = None,
    **kwargs
):
    
    if isinstance(output_modification, List):
        return [
            modify_output(latent, modifier, **kwargs)
            for latent, modifier in zip(output, output_modification)
        ]

    if output_modification is None:
        return output
    elif isinstance(output_modification, Callable):
        return output_modification(output, **kwargs)


def generate(
    model: Any, n: int = 100, generate_input: bool = True, 
    folders: List[str] = ["latent", "image"], return_result: bool = False,
    image_output: int = 1, model_parameter: Dict = {}, image_saver: Callable = None,
    input_generation: Union[Callable, List[Callable], List[Any], Dict[str, Callable], None] = None,
    input_generation_kwargs: Union[List[Dict], Dict] = {},
    output_modification: Union[Callable, None, List[Callable]] = None,
    output_modification_kwargs: Dict = {}
):

    if return_result:
        result = []

    if not isinstance(input_generation_kwargs, list) and isinstance(input_generation, list):
        input_generation_kwargs = [{}] * len(input_generation)

    for folder in folders:
        path: str = os.path.join("./", folder)
        os.makedirs(path, exist_ok=True)

    for i in tqdm(range(n), desc="Generating images..."):
        if generate_input:
            if isinstance(input_generation, Callable):
                input_value = input_generation(**input_generation_kwargs)
                input_name: tuple = list(inspect.signature(model.forward).parameters.keys())[0]
                input = {
                    input_name: input_value
                }
            elif isinstance(input_generation, list):
                if isinstance(input_generation[0], Callable):
                    input_value = [input(**input_generation_kwargs[i]) for i, input in enumerate(input_generation)]
                else:
                    input_value = input_generation

                input_name: tuple = list(inspect.signature(model.forward).parameters.keys())[:len(input_value)]
                input = {
                    name: value
                    for name, value in zip(input_name, input_value)
                }
            elif isinstance(input_generation, dict):
                input = {
                    key: value()
                    for key, value in input_generation.items()
                }
                input_value = input.values()
            else:
                raise AttributeError(f"The type of the argument input_generation is neither dict, list or callable")
        else:
            input = {}
            input_value = []

        input_tuple = tuple(input_value) if isinstance(input_value, list) else (input_value,)
        model_output = model(**input, **model_parameter)
        model_output = model_output if isinstance(model_output, tuple) else (model_output,)

        outputs = input_tuple + model_output
        outputs = modify_output(outputs, output_modification, **output_modification_kwargs)

        if not return_result:
            for output_index, (out, folder) in enumerate(zip(outputs, folders)):
                save_path = os.path.join("./", folder, f"output_{i}")

                if output_index == image_output:
                    if image_saver is None:
                        torchvision.utils.save_image(
                            out, save_path + ".jpg"
                        )
                    else:
                        image_saver(
                            out, save_path + ".jpg"
                        )
                else:
                    np.save(save_path + ".npy", out.detach().cpu().numpy())
        else:
            result.append(outputs)

    if return_result:
        return result
