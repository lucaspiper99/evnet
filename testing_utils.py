import os
import tqdm
import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier, EnsembleClassifier

from evnet.modules import EnsembleUpToLayer
from evnet import EVNet
from evnet.utils import ImageNetProbabilitiesTo16ClassesMapping
from data import get_imagenet_test_mask, get_imagenet, get_imagenet_c, get_imagenet_ood, get_stylized16_imagenet,\
get_data_norm


class ResultsData():
    def __init__(self, perturb):
        """initializes empty results data dictionary.

        Args:
            perturb (str): Type of perturbation (`corruptions`, `domain_shift`, `adversarial`)
        """
        self._check_pertub(perturb)
        
        self._results = {
            'exp_name': [],
            'seed': [],
            'model_arch': [],
            'subcorticalblock': [],
            'top1_acc': [],
        }
        if perturb=='corruptions':
            self._results['severity'] = []
            self._results['corruption'] = []
        elif perturb=='domain_shift':
            self._results['dataset'] = []
        elif perturb=='adversarial':
            self._results['eps'] = []
            self._results['norm'] = []
            self._results['epsilon_step_factor'] = []
            self._results['iterations'] = []
            self._results['ensemble_size'] = []

    @classmethod
    def from_csv(cls, root, perturb):
        cls._check_pertub(perturb)

        if not os.path.isfile(root):
            return cls(perturb)
        
        results = pd.read_csv(root, index_col=0).reset_index(drop=True)
        results = results.where(pd.notna(results), '').to_dict(orient="list")

        cls._check_dict(results, perturb)
        
        return cls(results) if results.shape[0] > 0 else cls(perturb)
    
    @staticmethod
    def _check_dict(data, perturb):
        cols = data.keys()
        keys = cls(perturb)._results.keys()
        if cols != keys:
            diff = (cols - keys) | (sets - keys)
            raise Exception(f'Invalid CSV output file. Keys do not match: {diff}')
    
    @staticmethod
    def _check_pertub(perturb):
        if perturb not in ['corruptions', 'domain_shift', 'adversarial']:
            raise ValueError(f'Invalid `perturb` parameter. Available options: `corruptions`, `dom_shift`, `adversarial`')

    def to_csv(self, root):
        pd.DataFrame(self._results).reset_index(drop=True).to_csv(root, index=True)
    
    def update(self, **result_kwargs):
        if result_kwargs.keys() != self._results.keys():
            raise KeyError(f'Not all columns are being updated.\nNew keys: {self._results.keys()}\nOld keys: {self._results.keys()}')
        for k, v in result_kwargs.items():
            self._results[k].append(v)


def load_model(args, subcort_kwargs, vone_kwargs, ckpt_path):
    """Loads models for inference, ensembling if needed.

    :param argparse.Namespace args: Script parsed arguments
    :param dict subcort_kwargs: SubcorticalBlock kwargs
    :param dict vone_kwargs: VOneBlock kwargs
    :param str ckpt_path: Path to root directory where model checkpoint is saved
    :return nn.Module: Model with weights loaded
    """    
    model = EVNet(
        with_voneblock=args.with_voneblock, model_arch=args.model_arch, 
        **vone_kwargs, **subcort_kwargs
    )
    ckpt = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(ckpt["model"])
    if hasattr(args, 'ensemble_type') and args.ensemble_type is not None:
        ensemble_layers = {
            'bottleneck': 'voneblock_bottleneck',
            'embedding': 'model.layer4.2',
            'logits': 'model.fc',
        }
        model = EnsembleUpToLayer(model, args.ensemble_size, ensemble_layers[args.ensemble_type])
    model.cuda()

    if len(args.devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.devices)
    
    return model


def test(args, model, dataloader, mask=None, aggregate=False):
    """Generic testing model function.

    :param argparse.Namespace args: Script parsed arguments
    :param nn.Sequential model: Model to test
    :param Dataloader dataloader: Dataloader to use for testing
    :param list mask: List of classes to consider, defaults to None
    :param bool aggregate: Whether to aggregate ImageNet classes into 16 classes, defaults to False
    :return np.ndarray: Top-1 accuracy
    """    
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.cuda()
            if aggregate:
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(not args.no_amp)):
                    softmax = nn.functional.softmax(model(x), dim=1).cpu().numpy()
                preds = ImageNetProbabilitiesTo16ClassesMapping().probabilities_to_decision(softmax)
                preds = torch.from_numpy(preds)
            else:
                y = y.cuda()
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(not args.no_amp)):
                    preds = model(x) if mask is None else model(x)[:, mask]
                preds = torch.argmax(preds, dim=1)
            predictions.append(preds.cpu().numpy())
            labels.append(y.cpu().numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    acc_top1 = (predictions == labels).sum() / labels.shape[0]
    return acc_top1


def test_corruption(args, model, subcort_kwargs, corruption, severity):
    """Tests model on image corruptions (ImageNet-C).

    :param argparse.Namespace args: Script parsed arguments
    :param nn.Sequential model: Model to be tested
    :param dict subcort_kwargs: SubcorticalBlock kwargs
    :param str corruption: ImageNet-C corruption folder name
    :param int severity: Corruption severity level (1-5)
    :return float: Top-1 accuracy
    """    
    dataset = get_imagenet_c(args.data_dir, subcort_kwargs, corruption, severity)
    loader = DataLoader(
        dataset, batch_size=256, num_workers=args.num_workers,
        pin_memory=True, prefetch_factor=args.prefetch_factor
        )
    return test(args, model, loader)


def test_domain_shift(args, model, subcort_kwargs, stylized16=False):
    """Test OOD ImageNet dataset (ImageNet-R, ImageNet-Cartoon, ImageNet-Drawing, ImageNet-Sketch ImageNet-Stylized16).
    
    :param argparse.Namespace args: Script parsed arguments
    :param nn.Sequential model: Model to be tested
    :param dict subcort_kwargs: SubcorticalBlock kwargs
    :param bool stylized16: Whether domain shift dataset is ImageNet-Stylized16, defaults to False
    :return float: Top-1 accuracy
    """    
    if stylized16:
        dataset = get_stylized16_imagenet(args.data_dir, subcort_kwargs)
    else:
        dataset = get_imagenet_ood(args.data_dir, subcort_kwargs)
    
    loader = DataLoader(
        dataset, batch_size=256, num_workers=args.num_workers,
        pin_memory=True, prefetch_factor=args.prefetch_factor
        )
    
    return test(args, model, loader, get_imagenet_test_mask(args.data_dir), aggregate=stylized16)
    

def test_adversarial(args, model, subcort_kwargs, norm, eps, k, input_shape=(3, 244, 244), num_classes=1000):
    """Test adversarial attacks under a single pair of norm constraint-strength.

    :param argparse.Namespace args: Script parsed arguments
    :param nn.Sequential model: Model to be tested
    :param dict subcort_kwargs: SubcorticalBlock kwargs
    :param float norm: Norm constraint of the perturbation budget
    :param float eps: Perturbation budget (epsilon)
    :param int k: Number of Monte Carlo gradient samples for generating attack (ensemble size)
    :param tuple input_shape: Input shape (channels first), defaults to (3, 244, 244)
    :param int num_classes: Number of classes, defaults to 1000
    :return float: Top-1 accuracy
    """
    def _ensemblize(classifier, k=10):
        if k==1:
            return classifier
        return EnsembleClassifier(
                classifiers=[classifier for _ in range(k)],
                channels_first=True,
                clip_values=(0, 1)
                )

    dataset = get_imagenet(args.data_dir, subcort_kwargs, False, True, num_images=args.num_images)
    loader = DataLoader(
        dataset, batch_size=min(256, args.num_images), shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=False, prefetch_factor=args.prefetch_factor
        )

    mean, std = get_data_norm(subcort_kwargs, False)

    classifier = PyTorchClassifier(
        model=model,
        preprocessing=(mean, std),
        clip_values=(0, 1),
        loss=nn.CrossEntropyLoss(),
        input_shape=input_shape,
        nb_classes=num_classes,
        optimizer=optim.SGD(model.parameters(), lr=.01),
        device_type='gpu',
        use_amp=args.no_amp
    )

    attack = ProjectedGradientDescent(
        estimator=_ensemblize(classifier, k=args.k),
        norm=norm,
        eps=eps,
        eps_step=eps/args.eps_step_factor,
        max_iter=args.pgd_iterations,
        num_random_init=0,
        verbose=False
    )
    
    predictions = []
    labels = []
    with tqdm.tqdm(total=len(loader)) as pbar:
        for x, y in loader:
            X_test_adv = attack.generate(x=x.numpy(), y=y.numpy())
            preds = classifier.predict(X_test_adv)
            predictions.append(np.argmax(preds, axis=1))
            labels.append(y.numpy())
            pbar.update(1)

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    return np.sum(predictions == labels) / len(labels)

