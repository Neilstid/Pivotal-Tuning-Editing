import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import Any
import re
import urllib
import requests
import hashlib
import glob
import os
import html
import uuid
import io
import tempfile
import cv2



from typing import Callable, List, Union



def avg_latent(generator, w_avg_samples=10000):
    z_samples = np.random.RandomState(123).randn(w_avg_samples, generator.z_dim)
    w_samples = generator.mapping(torch.from_numpy(z_samples).to("cuda"), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]

    return torch.tensor(w_avg, dtype=torch.float32, device="cuda", requires_grad=True) 


def read_image(
    img_invert_path: Union[str, Image.Image], device: Union[str, torch.device] = "auto", 
    size=None
) -> torch.Tensor:
    
    if isinstance(img_invert_path, List):
        return torch.cat([read_image(path) for path in img_invert_path], dim=0)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(img_invert_path, str):
        with open(img_invert_path,"rb") as f:
            image = Image.open(f)
            image = image.convert("RGB")
    elif isinstance(img_invert_path, np.ndarray):
        image = Image.fromarray(img_invert_path[..., ::-1])
        image = image.convert("RGB")
    elif isinstance(img_invert_path, Image.Image):
        image = img_invert_path
        image = image.convert("RGB")
    elif isinstance(img_invert_path, torch.Tensor):
        return img_invert_path
    else:
        raise NotImplementedError(f"Type {type(img_invert_path)} is not supported!")

    if size is not None:
        image = image.resize(size=size)

    # Apply a transform form the PIL image
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device).to(torch.float32)

    return image


def save_inversion_result(
    syn_img: torch.Tensor, img_output_save_path: str, latent_space: torch.Tensor,
    latent_output_save_path: str, loss_history: list, latent_history: List,
    returned_information: Callable = None
):
    img = (syn_img.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)

    if isinstance(latent_space, list):
        latent_space_saver = lambda tensor_list: np.array([t.detach().cpu().numpy() for t in tensor_list])
    else:
        latent_space_saver = lambda tensor: tensor.detach().cpu().numpy()

    if img_output_save_path is not None:
        for single_img, single_img_path in zip(img, img_output_save_path):
            Image.fromarray(single_img.cpu().numpy(), 'RGB').save(single_img_path)

    if latent_output_save_path is not None:
        for single_latent, single_latent_path in zip(latent_space, latent_output_save_path):
            np.save(
                single_latent_path, latent_space_saver(single_latent)
            )

    if isinstance(returned_information, Callable):
        return returned_information(
            np.array(loss_history), [Image.fromarray(single_img.cpu().numpy(), 'RGB') for single_img in img],
            latent_space, latent_history
        )


def construct_loss_input(loss_argues, syn_image, image, latent):
    argues = []

    for args in loss_argues:
        if args == "syn_image":
            argues.append(syn_image)
        elif args == "image":
            argues.append(image)
        elif args == "latent":
            argues.append(latent)

    return argues


def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)


_dnnlib_cache_dir = None

def set_cache_dir(path: str) -> None:
    global _dnnlib_cache_dir
    _dnnlib_cache_dir = path


def make_cache_dir_path(*paths: str) -> str:
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)