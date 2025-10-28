import os
import glob
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms as T
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, List

from evnet.utils import get_human_object_recognition_categories


def get_data_norm(subcort_kwargs, force_no_norm):
    """Gets appropriate ImageNet dataset normalization.

    Args:
        subcort_kwargs (dict): SubcorticalBlock kwargs.
        force_no_norm (bool): Whether to force no data normalization
    Returns:
        mean (tupple of float), std (tupple of floats)
    """
    mean, std = .5, .5
    if force_no_norm:
        mean, std = 0., 1.
    if subcort_kwargs['with_subcorticalblock'] and subcort_kwargs['with_light_adapt']:
        mean, std = 0., 1.
    return [mean]*3, [std]*3


def get_imagenet(root, subcort_kwargs, train, force_no_norm, num_images=None):
    """Returns ImageNet Dataset.

    Args:
        root (str): ImageNet root directory.
        subcort_kwargs (dict): SubcorticalBlock kwargs.
        train (bool): Whether using training data.
        force_no_norm (bool): Whether to force no data normalization
        num_images (int, optional): Number of images to use in the dataset (`None` uses all images)
    """
    if train:
        root = os.path.join(root, 'train')
        transform = [T.RandomResizedCrop(224), T.RandomHorizontalFlip(p=.5), T.ToTensor()]
        transform.append(T.Normalize(*get_data_norm(subcort_kwargs, force_no_norm)))
        return datasets.ImageFolder(root, T.Compose(transform))
    root = os.path.join(root, 'val')
    transform = [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
    transform.append(T.Normalize(*get_data_norm(subcort_kwargs, False)))

    dataset = datasets.ImageFolder(root, T.Compose(transform))

    if num_images is None:
        return dataset
    
    stride = len(dataset) // num_images
    dataset.imgs = dataset.imgs[::stride]
    dataset.samples = dataset.samples[::stride]
    dataset.targets = dataset.targets[::stride]
    return dataset

def get_imagenet_c(root, subcort_kwargs, corruption, severity):
    """Returns ImageNet-C.

    Args:
        root (str): ImageNet root directory.
        subcort_kwargs (dict): SubcorticalBlock kwargs.
    Return:
        dataset (Dataset)
    """
    root = os.path.join(root, corruption, str(severity))
    if not os.path.isdir(root):
        raise ValueError(f'Directory does not exist: {root}')
    transform = [T.ToTensor()]
    transform.append(T.Normalize(*get_data_norm(subcort_kwargs, False)))
    return datasets.ImageFolder(root, T.Compose(transform))

def get_imagenet_ood(root, subcort_kwargs):
    """Returns ImageNet OOD dataset.

    Args:
        root (str): ImageNet root directory.
        subcort_kwargs (dict): SubcorticalBlock kwargs.
    Return:
        dataset (Dataset)
    """
    transform = [T.ToTensor(), T.Resize(256, antialias=True), T.CenterCrop(224)]
    transform.append(T.Normalize(*get_data_norm(subcort_kwargs, False)))
    return datasets.ImageFolder(root, T.Compose(transform))

def get_stylized16_imagenet(root, subcort_kwargs):
    """Returns ImageNet OOD dataset.

    Args:
        root (str): ImageNet root directory.
        subcort_kwargs (dict): SubcorticalBlock kwargs.
    Return:
        dataset (Dataset)
    """
    transform = [T.ToTensor()]
    transform.append(T.Normalize(*get_data_norm(subcort_kwargs, False)))
    return ModelVsHumanDataset(
            root=root, transform=T.Compose(transform),
            parametric=True, return_ood=False
            )

def get_imagenet_test_mask(test_path, imagenet_path='/path/to/imagenet'):
    """Returns ImageNet-X test set logit mask.

    Args:
        test_path (str): Test set path
        imagenet_path (str): ImageNet dataset path
    """
    imagenet_wnids = datasets.ImageFolder(os.path.join(imagenet_path, 'val')).classes
    test_set_wnids = datasets.ImageFolder(test_path).classes
    return [wnid in set(test_set_wnids) for wnid in imagenet_wnids]


class ModelVsHumanDataset(Dataset):
    def __init__(self, root, transform, parametric=True, return_ood=False):
        """
        Args:
            root (str): Root directory of dataset (e.g., ".../model-vs-human/rotation").
            transform (callable): Optional transform to be applied on a sample.
            parametric (bool, optional): Whether the dataset also has OOD parameters.
            return_ood (bool, optional): Whether to also return OOD parameter (e.g., rotation angle).
        """
        self.transform = transform
        self.return_ood = return_ood if parametric else False
        self.parametric = parametric
        self.root = os.path.join(root, "dnn", "session-1") if parametric else root
        self.label_map = {l:i for i, l in enumerate(get_human_object_recognition_categories())}
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, str, Optional[str]]]:
        """
        Loads all samples.

        Returns:
            List of (image_path, label_string, ood_value)
        """
        samples = []
        for root, _, files in os.walk(self.root):
            for fname in files:
                if fname.lower().endswith('.png'):
                    path = os.path.join(root, fname)
                    label_str, ood_value = self._parse_label_and_ood(fname, path)
                    samples.append((path, label_str, ood_value))
        return samples

    def _parse_label_and_ood(self, fname: str, full_path: str) -> Tuple[str, Optional[str]]:
        """
        Parses label and OOD parameter from filename or path.
        To be implemented by subclasses.
        """
        if self.parametric:
            parts = fname.split('_')
            ood_param = parts[3] if self.return_ood else ''
            return parts[4], ood_param
        else:
            rel_path = os.path.relpath(full_path, self.root)
            parts = rel_path.split(os.sep)
            return parts[0], ''

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            image (tensor), label (int), ood_param (str, optional)
        """
        path, label_str, ood_param = self.samples[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        if self.return_ood:
            return image, (self.label_map[label_str], ood_param)
        return image, self.label_map[label_str]

