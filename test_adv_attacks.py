import os
import random
import argparse
import tqdm
import torch
import yaml
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T, models

from evnet import EVNet, get_subcort_kwargs, get_vone_kwargs
from evnet.utils import set_seed, get_experiment_name, get_model_ckpts
from testing_utils import ResultsData, load_model, test_adversarial


parser = argparse.ArgumentParser(
    description="Testing adversarial attacks."
)
parser.add_argument(
    "--data_dir", default="/path/to/data_dir", type=str, help="Path to dataset root directory"
)
parser.add_argument(
    "--out_dir", default='/path/to/out_dir', type=str, help="Output root directory where CSV file with results will be created"
)
parser.add_argument(
    "--ckpt_dir", default='/path/to/ckpt_dir', type=str, help="Root directory where model checkpoints are saved"
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
    "--strength_idxs", type=int, default=[2,4,6,8], nargs='+', help="List of indices refering to the attack strengths to use." 
)
parser.add_argument(
    "--k", type=int, default=10, help="Number of models when using model ensemble." 
)
parser.add_argument(
    "--pgd_iterations", type=int, default=64, help="Number of PGD iterations to use in the attack."
)
parser.add_argument(
    "--eps_step_factor", type=int, default=32, help="Ratio between epsilon and step-size of each PGD iteration." 
)
parser.add_argument(
    "--num_images", type=int, default=5000, help="Number of images from the ImageNet Validation set to use." 
)
parser.add_argument(
    "--norms", type=str, default=['1', '2', 'inf'], nargs='+', help="List of attack norm to use." 
)
parser.add_argument(
    "--devices", default=[0], type=int, nargs='+', help="List of GPU IDs to use when training the model"
)

ARGS = parser.parse_args()

def main():
    subcort_kwargs = get_subcort_kwargs(ARGS.subcorticalblock)
    vone_kwargs = get_vone_kwargs(ARGS.model_arch)

    exp_name = get_experiment_name(ARGS)
    subdirs = get_model_ckpts(ARGS)

    output_file = os.path.join(ARGS.out_dir, f'{exp_name}-adv.csv')
    results = ResultsData.from_csv(output_file, 'adversarial')

    set_seed(42)

    norm_strength_mapping = {
        1: np.logspace(np.emath.logn(4, 10), np.emath.logn(4, 2560), num=9, base=4)[ARGS.strength_idxs],
        2: np.logspace(np.emath.logn(4, .0375), np.emath.logn(4, 9.6), num=9, base=4)[ARGS.strength_idxs],
        np.inf: np.logspace(np.emath.logn(4, 1/4080), np.emath.logn(4, 16/255), num=9, base=4)[ARGS.strength_idxs]
        }

    torch.cuda.set_device(f'cuda:{ARGS.devices[0]}' if torch.cuda.is_available() else "cpu")
    
    for seed, ckpt_path in subdirs:
        model = load_model(ARGS, subcort_kwargs, vone_kwargs, ckpt_path)
        for norm in map(float, ARGS.norms):
            for eps in norm_strength_mapping[norm]:
                top1_acc = test_adversarial(ARGS, model, subcort_kwargs, norm, eps, ARGS.k)
                results.update(
                    exp_name=exp_name,
                    seed=seed,
                    top1_acc=top1_acc,
                    subcorticalblock=ARGS.subcorticalblock,
                    model_arch=ARGS.model_arch,
                    norm=norm,
                    eps=eps,
                    epsilon_step_factor=ARGS.eps_step_factor,
                    iterations=ARGS.pgd_iterations,
                    ensemble_size=ARGS.k
                    )
                results.to_csv(output_file)

if __name__ == "__main__":
    main()