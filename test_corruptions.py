import os
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms as T


from evnet import EVNet, get_subcort_kwargs, get_vone_kwargs
from evnet.utils import set_seed, get_experiment_name, get_model_ckpts
from testing_utils import ResultsData, load_model, test_corruption


parser = argparse.ArgumentParser(
    description="Testing on image corruptions (ImageNet-C)."
)
parser.add_argument(
    "--data_dir", default="/media/imagenet-c/", type=str, help="Path to dataset root directory"
)
parser.add_argument(
    "--out_dir", default='/media/generalstorage3/lapstorage/out', type=str,
    help="Output directory where CSV file with results will be created"
)
parser.add_argument(
    "--ckpt_dir", default='/media/generalstorage3/lapstorage/out', type=str,
    help="Directory where model checkpoints are saved"
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="How many subprocesses to use for data loading",
)
parser.add_argument(
    "--prefetch_factor", default=8, type=int, help="Number of batches loaded in advance by each worker"
)
parser.add_argument(
    "--no_amp", action="store_true", help="Whether to not to use automatic mixed precision"
)
parser.add_argument(
    "--with_prime", action="store_true", help="Whether to use PRIME data augmentation"
)
parser.add_argument(
    "--pretrained_ckpt", default=None, type=str, help="Checkpoint of pretrained model, when fine-tuning. `None` trains from scratch.",
)
parser.add_argument(
    "--seeds", type=int, default=None, nargs='+', help="Which seed of the model to evaluate the attack." 
)
parser.add_argument(
    "--model_arch", default='resnet-50', type=str, help="EVNet backend model architecture",
    choices=['resnet-50', 'efficientnet-b0', 'cornet-z']
)
parser.add_argument(
    "--subcorticalblock", default='', type=str, help="Name of the SubcorticalBlock variant"
)
parser.add_argument(
    "--with_voneblock", action="store_true", help="Whether to include a VOneBlock in the model"
)
parser.add_argument(
    "--ensemble_size", type=int, default=None, help="Inference ensembling size, if ensemble_size is not `None`"
)
parser.add_argument(
    "--ensemble_type", type=str, default=None, help="Which layer to use to average activations in inference ensemble",
    choices=['logits', 'embedding', 'bottleneck']
)
parser.add_argument(
    "--devices", default=[0], type=int, nargs='+', help="GPU IDs to use when training the model"
)

ARGS = parser.parse_args()

def main():
    subcort_kwargs = get_subcort_kwargs(ARGS.subcorticalblock)
    vone_kwargs = get_vone_kwargs(ARGS.model_arch)

    exp_name = get_experiment_name(ARGS)
    subdirs = get_model_ckpts(ARGS)

    output_file = os.path.join(ARGS.out_dir, f'{exp_name}-corr.csv')
    results = ResultsData.from_csv(output_file, 'corruptions')

    set_seed(42)

    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness'
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 
        ]

    torch.cuda.set_device(f'cuda:{ARGS.devices[0]}' if torch.cuda.is_available() else 'cpu')
    
    for seed, ckpt_path in subdirs:
        model = load_model(ARGS, subcort_kwargs, vone_kwargs, ckpt_path)
        for corruption in tqdm(corruptions, desc=f'Seed {seed}'):
            for severity in range(1, 6):
                top1_acc = test_corruption(ARGS, model, subcort_kwargs, corruption, severity)
                results.update(
                    exp_name=exp_name,
                    seed=seed,
                    severity=severity,
                    corruption=corruption,
                    top1_acc=top1_acc,
                    subcorticalblock=ARGS.subcorticalblock,
                    model_arch=ARGS.model_arch,
                    )
                results.to_csv(output_file)

if __name__ == "__main__":
    main()