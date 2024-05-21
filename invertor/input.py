import numpy as np
import torch
from typing import Union


def n_latents_init(
    generator: torch.nn.Module, n: int = 10000,
    device: Union[str, torch.device] = "auto"
) -> torch.Tensor:

    # Set the device to run on ------------------------------------------------
    if device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create the latent -------------------------------------------------------
    c = torch.zeros([1, 0], device=device)
    latent_space_z = torch.from_numpy(np.random.RandomState().randn(1, 512)).to(device)
    latent_space = generator.mapping(latent_space_z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False).to(device)

    for _ in range(n - 1):
        latent_space_z = torch.from_numpy(np.random.RandomState().randn(1, 512)).to(device)
        latent_space = generator.mapping(
            latent_space_z, c, truncation_psi=1, truncation_cutoff=None,
            update_emas=False
        ).to(device)

    latent_space = latent_space / n
    latent_space.requires_grad = True

    return latent_space
