# Explicitly Modeling Subcortical Vision with a Neuro-Inspired Front-End Improves CNN Robustness

[![arXiv](https://img.shields.io/badge/arXiv-2506.03089-b31b1b.svg)](https://arxiv.org/abs/2506.03089)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17465273.svg)](https://doi.org/10.5281/zenodo.17465273)

## Abstract

> Convolutional neural networks (CNNs) trained on object recognition achieve high task performance but continue to exhibit vulnerability under a range of visual perturbations and out-of-domain images, when compared with biological vision. Prior work has demonstrated that coupling a standard CNN with a front-end (VOneBlock) that mimics the primate primary visual cortex (V1) can improve overall model robustness. Expanding on this, we introduce Early Vision Networks (EVNets), a new class of hybrid CNNs that combine the VOneBlock with a novel SubcorticalBlock, whose architecture draws from computational models in neuroscience and is parameterized to maximize alignment with subcortical responses reported across multiple experimental studies. Without being optimized to do so, the assembly of the SubcorticalBlock with the VOneBlock improved V1 alignment across most standard V1 benchmarks, and better modeled extra-classical receptive field phenomena. In addition, EVNets exhibit stronger emergent shape bias and outperform the base CNN architecture by 9.3% on an aggregate benchmark of robustness evaluations, including adversarial perturbations, common corruptions, and domain shifts. Finally, we show that EVNets can be further improved when paired with a state-of-the-art data augmentation technique, surpassing the performance of the isolated data augmentation approach by 6.2% on our robustness benchmark. This result reveals complementary benefits between changes in architecture to better mimic biology and training-based machine learning approaches.

## Model Weights

We made four sets of model weights available in [Zenodo](https://zenodo.org/records/17465273):

- [EVResNet50](https://zenodo.org/records/17465273/files/evresnet-50.pth?download=1)
- [EVResNet50 + PRIME](https://zenodo.org/records/17465273/files/evresnet50+prime.pth?download=1)
- [EVEfficientNet-B0](https://zenodo.org/records/17465273/files/evefficientnet-b0.pth?download=1)
- [EVCORnet-Z](https://zenodo.org/records/17465273/files/evcornet-z.pth?download=1)

To load the checkpoints do:

```
import torch
checkpoint = torch.load('path/to/checkpoint')
model.load_state_dict(checkpoint["model"])
```

## Repository Structure

```
evnet/
├── configs/    # Configuration files
├── utils/      # Utilities
├── tuning/     # SubcorticalBlock hyperparameter tuning helpers and CSV files
└── prime/      # PRIME data augmentation module
```

## Requirements

- python >= 3.11
- torch
- torchvision
- numpy
- pandas
- yaml
- pillow
- tqdm
- scipy
- scikit-optimize
- adversarial-robustness-toolbox
- einops
- opt-einsum

## Datasets Used

| Dataset           | Perturbation             | Download Link                                                                      |
|-------------------|--------------------------|------------------------------------------------------------------------------------|
| ImageNet          | Adversarial (+ Clean)    | https://www.image-net.org/download.php                                             |
| ImageNet-C        | Common image corruptions | https://zenodo.org/records/2235448                                                 |
| ImageNet-Cartoon  | Domain shift             | https://zenodo.org/records/6801109                                                 |
| ImageNet-Drawing  | Domain shift             | https://zenodo.org/records/6801109                                                 |
| ImageNet-R        | Domain shift             | https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar                         |
| ImageNet-Sketch   | Domain shift             | https://www.kaggle.com/datasets/wanghaohan/imagenetsketch                          |
| ImageNet-Stylized | Domain shift             | https://github.com/bethgelab/model-vs-human/releases/download/v0.1/stylized.tar.gz |

## PRIME data augmentation

The PRIME data augmentation module was based on the code from the [the original repository](https://github.com/amodas/PRIME-augmentations).

## Installation

To install the `evnet` package do:
```
pip install git+https://github.com/lucaspiper99/evnet.git
```

To setup the repository locally do (UNIX):
```
git clone https://github.com/lucaspiper99/evnet.git
cd evnet
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requisites.txt
```

## Example Usage

```
from evnet import EVNet, get_subcort_kwargs, get_vone_kwargs

subcort_kwargs = get_subcort_kwargs(variant='standard')
vone_kwargs = get_vonekwargs(model_arch='resnet-50')
model = EVNet(with_voneblock=True, model_arch=True, **subcort_kwargs, **vone_kwargs)
```

The [`train.sh`](./train.sh) and [`test.sh`](./test.sh) bash scripts include several examples on training and testing EVNets.

## BibTeX

```
@inproceedings{piper2025explicitly,
    title={Explicitly Modeling Subcortical Vision with a Neuro-Inspired Front-End Improves {CNN} Robustness},
    author={Lucas Piper and Arlindo L. Oliveira and Tiago Marques},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025}
}
```

## Acknowledgements

This work was supported by the project Center for Responsible AI reference no. C628696807-00454142, and by national funds through Fundação para a Ciência e a Tecnologia, I.P. (FCT) under projects UID/50021/2025 and UID/PRR/50021/2025.
