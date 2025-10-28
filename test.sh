#!/bin/bash

# Testing EVResNet-50 on image corruptions
python test_corruptions.py --subcorticalblock standard --with_voneblock --model_arch resnet-50

# Testing EVResNet-50 on all domain shifts datasets
python test_domain_shift.py --subcorticalblock standard --with_voneblock --model_arch resnet-50 --imagenet cartoon --data_dir /path/to/imagenet-cartoon
python test_domain_shift.py --subcorticalblock standard --with_voneblock --model_arch resnet-50 --imagenet drawing --data_dir /path/to/imagenet-drawing
python test_domain_shift.py --subcorticalblock standard --with_voneblock --model_arch resnet-50 --imagenet r --data_dir /path/to/imagenet-r
python test_domain_shift.py --subcorticalblock standard --with_voneblock --model_arch resnet-50 --imagenet sketch --data_dir /path/to/imagenet-sketch
python test_domain_shift.py --subcorticalblock standard --with_voneblock --model_arch resnet-50 --imagenet styized16 --data_dir /path/to/imagenet-stylized16

# Testing EVResNet-50 on adversarial attacks (full-strength attack set)
python test_adv_attacks.py --subcorticalblock standard --with_voneblock --model_arch resnet-50 --strength_idxs 2 4 6 8 --norms 1 2 inf --pgd_iterations 64 --eps_step_factor 32 --k 10 --num_images 5000

# Testing EVResNet-50 on adversarial attacks (reduced attack set)
python test_adv_attacks.py --subcorticalblock standard --with_voneblock --model_arch resnet-50 --strength_idxs 2 4 --norms 1 2 inf --pgd_iterations 64 --eps_step_factor 32 --k 10 --num_images 5000

# Testing EVResNet-50 with inference ensembling on image corruptions (same flags can be used for `test_domain_shift.py`)
python test_corruptions.py --subcorticalblock standard --with_voneblock --model_arch resnet-50 --ensemble_size 2 --ensemble_type bottleneck
python test_corruptions.py --subcorticalblock standard --with_voneblock --model_arch resnet-50 --ensemble_size 4 --ensemble_type embedding
python test_corruptions.py --subcorticalblock standard --with_voneblock --model_arch resnet-50 --ensemble_size 8 --ensemble_type logits
