from typing import Any, List, Union, Callable, Dict, Literal
from matplotlib import pyplot as plt
import torch
from PIL import Image
from rich.progress import Progress
from tqdm import tqdm
import numpy as np
from PIL import Image
import math
from invertor.loss import InvertionLoss
from invertor.utils import construct_loss_input, read_image, save_inversion_result, open_url
import copy
import torch.nn.functional as F
import PIL


def original_projection(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        synth_images = G.synthesis(w_opt.repeat([1, G.mapping.num_ws, 1]), noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    projected_w = w_out.repeat([1, G.mapping.num_ws, 1])

    synth_image = G.synthesis(projected_w[-1].unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'./proj.png')


def invert_StyleGAN_experiments(
    generator: torch.nn.Module, img_invert_path: Union[str, Image.Image],
    img_output_save_path: str = None, latent_output_save_path: str = None,
    invertion_iteration: int = 0, loss_function: Callable = None,
    initial_latent_function: Callable = None, initial_latent_function_kwargs: Dict = None,
    returned_information: Callable = None, device: Union[str, torch.device] = "auto",
    generator_kwargs: dict[str, Any] = None, context_pbar: Progress = None,
    space_modifier: Callable = lambda x: x,
    loss_argues: List[Literal["syn_image", "image", "latent_space"]] = ["syn_image", "image"],
    gradient_optimizer: torch.optim.Optimizer = torch.optim.Adam, opt_args: Dict = {},
    learning_rate: float = 0.1, other_optimize_param: Dict = None,
    lr_rampdown_length = 0.25, lr_rampup_length = 0.05, opt_noise_buffer: bool = False,
    update_lambdas: int = 100
):
    
    if device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    generator.eval().requires_grad_(False).to(device) # type: ignore

    if initial_latent_function_kwargs is None:
        initial_latent_function_kwargs = {}

    if loss_function is None:
        loss_function = InvertionLoss()
    else:
        loss_function = copy.deepcopy(loss_function)

    if generator_kwargs is None:
        generator_kwargs = {"input_is_latent": True, "return_latents": False}

    # Create the latent space -------------------------------------------------

    if isinstance(initial_latent_function, Callable):
        latent_space = initial_latent_function(
            generator=generator, **initial_latent_function_kwargs
        )
    elif isinstance(initial_latent_function, List):
        latent_space = [ltmp.clone().detach() for ltmp in initial_latent_function]
    else:
        latent_space = initial_latent_function.clone().detach()

    if isinstance(latent_space, torch.Tensor):
        latent_space.requires_grad = True
    elif isinstance(latent_space, list):
        for tensor in latent_space:
            tensor.requires_grad = True

    if opt_noise_buffer:
        noise_bufs = { name: buf for (name, buf) in generator.synthesis.named_buffers() if 'noise_const' in name }

    # Read the image ----------------------------------------------------------
    image = read_image(img_invert_path=img_invert_path, device=device)
    c = torch.zeros([1, 0], device=device)

    loss_history = []
    latent_history = []

    # Optimize the latent -----------------------------------------------------
    to_optimize = set(latent_space) if isinstance(latent_space, list) else {latent_space}
    if isinstance(latent_space, list):
        latent_space_saver = lambda tensor_list: [t.clone().detach() for t in tensor_list]
    else:
        latent_space_saver = lambda tensor: tensor.clone().detach()

    if other_optimize_param is not None:
        to_optimize.update(other_optimize_param)
    to_optimize.discard(None)

    if opt_noise_buffer:
        to_optimize.update(set(noise_bufs.values()))

    # Optimizer to change latent code in each backward step
    optimizer = gradient_optimizer(to_optimize, **opt_args, lr=learning_rate)

    # Init noise.
    if opt_noise_buffer:
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    # Invertion process -------------------------------------------------------

    if context_pbar is None:
        pbar = tqdm(range(invertion_iteration))
    elif isinstance(context_pbar, Progress):
        pbar = context_pbar.track(
            range(invertion_iteration),
            description="[cyan] Running invertion...",
        )
    else:
        pbar = range(invertion_iteration)

    for it in pbar:

        # Learning rate schedule.
        t = it / invertion_iteration

        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)

        syn_img = generator(space_modifier(latent_space), c, **generator_kwargs)
        syn_img = (syn_img.clamp(-1, 1) + 1) / 2

        # Compute loss
        loss = loss_function(*construct_loss_input(loss_argues, syn_img, image, latent_space)).sum()
        loss.backward()
        optimizer.step()

        # Normalize noise.
        if opt_noise_buffer:
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

        loss_history.append(loss.item())
        latent_history.append(latent_space_saver(latent_space))

        if isinstance(context_pbar, Progress) and (it + 1) % 10 == 0:
            pbar.set_description(f"Loss: {round(np.mean(loss_history[-9:]), 4)}")

    if isinstance(context_pbar, Progress):
        context_pbar.stop_task(context_pbar.task_ids[-1])
        context_pbar.update(context_pbar.task_ids[-1], visible=False)
    elif context_pbar is None:
        pbar.close()
        pbar.clear()

    # Save the results --------------------------------------------------------
    return save_inversion_result(
        syn_img=syn_img, img_output_save_path=img_output_save_path,
        latent_space=latent_space, latent_output_save_path=latent_output_save_path,
        loss_history=loss_history, latent_history=latent_history,
        returned_information=returned_information
    )


def multiphase_invert(
    generator: torch.nn.Module, img_invert_path: Union[str, Image.Image],
    img_output_save_path: str = None, latent_output_save_path: str = None,
    invertion_iteration: List[int] = 0, loss_function: Callable = None,
    initial_latent_function: Callable = None, initial_latent_function_kwargs: Dict = None,
    returned_information: Callable = None, device: Union[str, torch.device] = "auto",
    generator_kwargs: dict[str, Any] = None, context_pbar: Progress = None,
    space_modifier: Callable = lambda x: x,
    loss_argues: List[Literal["syn_image", "image", "latent_space"]] = ["syn_image", "image"],
    gradient_optimizer: torch.optim.Optimizer = torch.optim.Adam, opt_args: Dict = {},
    learning_rate: float = 0.1, other_optimize_param: Dict = None,
    lr_rampdown_length = 0.25, lr_rampup_length = 0.05, opt_noise_buffer: bool = False
):
    
    if device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    generator.eval().requires_grad_(False).to(device) # type: ignore

    if initial_latent_function_kwargs is None:
        initial_latent_function_kwargs = {}

    if loss_function is None:
        loss_function = InvertionLoss()
    else:
        loss_function = copy.deepcopy(loss_function)

    if generator_kwargs is None:
        generator_kwargs = {"input_is_latent": True, "return_latents": False}

    # Create the latent space -------------------------------------------------

    if isinstance(initial_latent_function, Callable):
        latent_space = initial_latent_function(
            generator=generator, **initial_latent_function_kwargs
        )
    else:
        latent_space = initial_latent_function.clone().detach()

    if opt_noise_buffer:
        noise_bufs = { name: buf for (name, buf) in generator.synthesis.named_buffers() if 'noise_const' in name }

    # Read the image ----------------------------------------------------------
    image = read_image(img_invert_path=img_invert_path, device=device)
    if image.shape[0] != latent_space.shape[0]:
        latent_space = latent_space.repeat(image.shape[0], 1, 1).detach()

    if isinstance(latent_space, torch.Tensor):
        latent_space.requires_grad = True
    elif isinstance(latent_space, list):
        for tensor in latent_space:
            tensor.requires_grad = True

    c = torch.zeros([1, 0], device=device)

    loss_history = []
    latent_history = []

    # Optimize the latent -----------------------------------------------------
    to_optimize = set(latent_space) if isinstance(latent_space, list) else {latent_space}
    if isinstance(latent_space, list):
        latent_space_saver = lambda tensor_list: [t.clone().detach() for t in tensor_list]
    else:
        latent_space_saver = lambda tensor: tensor.clone().detach()

    if other_optimize_param is not None:
        to_optimize.update(other_optimize_param)
    to_optimize.discard(None)

    if opt_noise_buffer:
        to_optimize.update(set(noise_bufs.values()))

    # Optimizer to change latent code in each backward step
    optimizer = gradient_optimizer(to_optimize, **opt_args, lr=learning_rate)

    # Init noise.
    if opt_noise_buffer:
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    # Invertion process -------------------------------------------------------

    if context_pbar is None:
        pbar = tqdm(range(sum(invertion_iteration)))
    else:
        pbar = context_pbar.track(
            range(sum(invertion_iteration)),
            description="[cyan] Running invertion...",
        )

    idx = 1
    invertion_iteration = [0] + invertion_iteration
    invertion_iteration = np.cumsum(invertion_iteration)

    for it in pbar:

        # Learning rate schedule.
        t = (it - invertion_iteration[idx-1]) / invertion_iteration[idx]

        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)

        syn_img = generator(space_modifier(latent_space), c, **generator_kwargs)
        syn_img = (syn_img.clamp(-1, 1) + 1) / 2

        # Compute loss
        loss = loss_function(*construct_loss_input(loss_argues, syn_img, image, latent_space)).sum()
        loss.backward()
        optimizer.step()

        # Normalize noise.
        if opt_noise_buffer:
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

        loss_history.append(loss.item())
        latent_history.append(latent_space_saver(latent_space))

        if context_pbar is None and (it + 1) % 10 == 0:
            pbar.set_description(f"Loss: {round(np.mean(loss_history[-9:]), 4)}")

    if context_pbar is not None:
        context_pbar.stop_task(context_pbar.task_ids[-1])
        context_pbar.update(context_pbar.task_ids[-1], visible=False)
    else:
        pbar.close()
        pbar.clear()

    # Save the results --------------------------------------------------------
    return save_inversion_result(
        syn_img=syn_img, img_output_save_path=img_output_save_path,
        latent_space=latent_space, latent_output_save_path=latent_output_save_path,
        loss_history=loss_history, latent_history=latent_history,
        returned_information=returned_information
    )
