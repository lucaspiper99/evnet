#!/bin/bash

# Training EVCORnet-Z
python train.py --subcorticalblock standard --with_voneblock --model_arch cornet-z

# Training EVEfficientNet-50
python train.py --subcorticalblock standard --with_voneblock --model_arch efficientnet-b0

# Training EVResNet-50
python train.py --subcorticalblock standard --with_voneblock --model_arch resnet-50

# Training EVResNet-50 + PRIME (from pretrained checkpoint)
python train.py --subcorticalblock standard --with_voneblock --model_arch resnet-50 --seed 0 --with_prime --pretrained_ckpt /path/to/ckpt

# Training all EVResNet50 variants
python train.py --subcorticalblock no-p-cells --with_voneblock --model_arch resnet-50 # No P cells
python train.py --subcorticalblock no-m-cells --with_voneblock --model_arch resnet-50 # No M cells
python train.py --subcorticalblock no-light-adapt --with_voneblock --model_arch resnet-50 # No light adaptation
python train.py --subcorticalblock no-contrast-norm --with_voneblock --model_arch resnet-50 # No contrast normalization
python train.py --subcorticalblock no-noise --with_voneblock --model_arch resnet-50 # No subcortical noise
python train.py --subcorticalblock standard --model_arch resnet-50  # No VOneBlock
python train.py --subcorticalblock lgn-v2-connection --with_voneblock --model_arch resnet-50 # With LGN-V2 connection
