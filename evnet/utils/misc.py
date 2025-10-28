import torch
import numpy as np
import random
import yaml
from pathlib import Path


def gaussian_kernel(
    sigma: float, k: float=1, size:float=15, norm:bool=False
    ) -> torch.tensor:
    """Returns a 2D Gaussian kernel.

    :param float sigma: Standard deviation of the Gaussian
    :param float k: Height of the Gaussian, defaults to 1
    :param float size: Kernel size, defaults to 15
    :param bool norm: Whether to normalize the Guassian kernel
    :return torch.tensor: Gaussian kernel
    """
    assert size % 2 == 1
    w = size // 2
    grid_val = torch.arange(-w, w+1, dtype=torch.float)
    x, y = torch.meshgrid(grid_val, grid_val, indexing='ij')
    gaussian = k * torch.exp(-(x**2 + y**2) / (2*(sigma)**2))
    if norm: gaussian /= torch.abs(gaussian.sum())
    return gaussian


def dog_kernel(
    sigma_c: float, sigma_s: float, k_c: float, k_s: float,
    polarity:int, size:int=27
    ) -> torch.tensor:
    """Returns a 2D Difference-of-Gaussians kernel.

    :param float sigma_c: Standard deviation of the center Gaussian
    :param float sigma_s: Standard deviation of the surround Gaussian
    :param float k_c: Peak sensitivity of the center
    :param float k_s: Peak sensitivity of the surround
    :param int polarity: Polarity of the center Gaussian (+1 or -1)
    :param int size: Kernel size, defaults to 27
    :return torch.tensor: DoG kernel
    """    
    assert size % 2 == 1
    assert polarity in [-1 , 1]
    center_gaussian = gaussian_kernel(sigma=sigma_c, k=k_c, size=size)
    surround_gaussian = gaussian_kernel(sigma=sigma_s, k=k_s, size=size)
    dog = polarity * (center_gaussian - surround_gaussian)
    dog /= torch.sum(dog)
    return dog


def gabor_kernel(frequency,  sigma_x, sigma_y, theta=0., offset=0., ks=61):
    """Returns Gabor kernel.

    :param float frequency: spatial frequency of gabor
    :param float sigma_x: standard deviation in x direction
    :param float sigma_y: standard deviation in y direction
    :param float theta: Angle theta, defaults to 0
    :param float offset: Offset, defaults to 0
    :param int ks: Kernel size, defaults to 61
    :return torch.tensor: Gabor kernel 
    """    
    w = ks // 2
    grid_val = torch.arange(-w, w+1, dtype=torch.float)
    x, y = torch.meshgrid(grid_val, grid_val, indexing='ij')
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)
    g = torch.zeros(y.shape)
    g[:] = torch.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= torch.cos(2 * np.pi * frequency * rotx + offset)

    return g

def generate_grating(
    size:float, radius:float, sf:float, theta:float=0, phase:float=0,
    contrast:float=1, gaussian_mask:bool=False
    ) -> torch.tensor:
    """Returns masked grating array.

    :param float size: kernel size
    :param float radius: standard deviation times sqrt(2) of the mask if gaussian_mask is True, and the radius if is false
    :param float sf: spatial frequency of the grating
    :param float theta: angle of the grating 
    :param float phase: phase of the grating
    :param bool gaussian_mask: mask is a Gaussian if true and a circle if false 
    :return torch.tensor: 2d masked grating array
    """
    grid_val = torch.linspace(-size//2, size//2, size, dtype=torch.float)
    X, Y = torch.meshgrid(grid_val, grid_val, indexing='ij')
    grating = torch.sin(2*np.pi*sf*(X*np.cos(theta) + Y*np.sin(theta)) + phase) * contrast
    mask = torch.exp(-((X**2 + Y**2)/(2*(radius/np.sqrt(2))**2))) if gaussian_mask else torch.sqrt(X**2 + Y**2) <= radius
    return grating * mask * .5 + .5


def sample_dist(hist, bins, ns, scale='linear'):
    """Samples distribution

    :param np.ndarray hist: Histogram
    :param np.ndarray bins: Bins
    :param int ns: Number of samples
    :param str scale: HIstogram scale, defaults to 'linear'
    :return np.ndarray: Random sample
    """    
    rand_sample = np.random.rand(ns)
    if scale == 'linear':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), bins)
    elif scale == 'log2':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), np.log2(bins))
        rand_sample = 2**rand_sample
    elif scale == 'log10':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), np.log10(bins))
        rand_sample = 10**rand_sample
    return rand_sample


def get_subcort_kwargs(variant):
    path = Path(__file__).parent / ".." / "configs" / "subcort_kwargs.yml"
    with open(path, "r") as f:
        subcort_kwargs = yaml.safe_load(f)[variant]
    return subcort_kwargs

def get_vone_kwargs(model_arch):
    path = Path(__file__).parent / ".." / "configs" / "vone_kwargs.yml"
    with open(path, "r") as f:
        vone_kwargs = yaml.safe_load(f)[model_arch]
    return vone_kwargs

def set_seed(seed):
    """Enforces deterministic behaviour and sets RNG seed for numpy and pytorch.

    :param int seed: seed
    """        
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
