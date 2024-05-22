from typing import List, Union
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F
from invertor.utils import open_url


PREWITT_KERNEL_X: torch.Tensor = torch.from_numpy(np.array([ # pylint: disable=E1101
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
]))


LOW_LEVEL_KERNEL_X: torch.Tensor = torch.from_numpy(np.array([ # pylint: disable=E1101
    [-1, 2, -1],
    [-1, 2, -1],
    [-1, 2, -1]
]))


PREWITT_KERNEL_Y: torch.Tensor = torch.from_numpy(np.array([ # pylint: disable=E1101
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
]))

LOW_LEVEL_KERNEL_Y: torch.Tensor = torch.from_numpy(np.array([ # pylint: disable=E1101
    [-1, -1, -1],
    [2, 2, 2],
    [-1, -1, -1]
]))



class InvertionLoss(torch.nn.Module):
    def __init__(self, device: Union[str, torch.device] = "auto") -> None:
        super().__init__()

        if device == "auto":
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Initialize the loss -----------------------------------------------------
        # Perceptual loss initialise object
        self.__perceptual = LpipsLoss(device=device)
        # MSE loss object
        self.__MSE_loss = torch.nn.MSELoss(reduction="mean")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.__MSE_loss(x, y) + self.__perceptual(x, y)


class InversionLossWithSpace(torch.nn.Module):
    def __init__(
        self, latents: List[torch.Tensor], lmbda: float = 0.5,
        device: Union[str, torch.device] = "auto"
    ) -> None:
        super().__init__()

        if device == "auto":
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if isinstance(latents[0][0], torch.Tensor):
            latents = latents[:,0,...].detach().cpu().numpy()
        else:
            latents = np.array(latents)

        self.__mean_latents = np.mean(latents, axis=0).flatten()
        self.__max_distance = np.max(cdist(
            [self.__mean_latents], latents.flatten().reshape(latents.shape[0], -1)
        ))
        self.__mean_latents = torch.Tensor(self.__mean_latents).to(device)
        self.__mean_latents.requires_grad = False

        self.__lmbda = lmbda

        # Initialize the loss -----------------------------------------------------
        # Perceptual loss initialise object
        # MSE loss object
        self.__MSE_loss = torch.nn.MSELoss(reduction="mean") 

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, x_latent: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(x_latent, torch.Tensor):
            pass
        elif isinstance(x_latent[0], torch.Tensor):
            x_latent = torch.Tensor(torch.cat([t.squeeze(0) for t in x_latent])) # pylint: disable=E1101

        dist_to_center = self.__MSE_loss(self.__mean_latents, x_latent.flatten())
        penalty = dist_to_center * self.__lmbda if dist_to_center >= self.__max_distance else 0

        return self.__MSE_loss(x, y) + penalty


class LpipsLoss(torch.nn.Module):
    def __init__(self, device: Union[str, torch.device] = "auto") -> None:
        super().__init__()

        if device == "auto":
            self.__device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.__device = device

        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with open_url(url) as f:
            self.vgg16 = torch.jit.load(f).eval().to(self.__device)


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.transform_img(x)
        y = self.transform_img(y)

        x_features = self.vgg16(x, resize_images=False, return_lpips=True)
        y_features = self.vgg16(y, resize_images=False, return_lpips=True)

        dist = (x_features - y_features).square().sum()

        return dist


    def transform_img(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) * (255/2)

        if x.ndim == 3:
            x = x.unsqueeze(0)

        if x.shape[2] != 256:
            x = F.interpolate(x, size=(256, 256), mode='area')

        return x
