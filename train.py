import os
import datetime
import argparse
import tqdm
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from evnet import EVNet, get_subcort_kwargs, get_vone_kwargs
from evnet.utils import ModelSaver, set_seed, get_experiment_name
from data import get_imagenet, get_data_norm
from evnet.prime import get_prime


parser = argparse.ArgumentParser(
    description="ImageNet Training"
)
parser.add_argument(
    "--data_dir", default="/media/imagenet/", type=str, help="Path to ImageNet data directory"
)
parser.add_argument(
    "--use_ckpt", action="store_true", help="Whether to load the last checkpoint"
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
    "--seed", default=42, type=int, help="Seed for the GFB initialization and RNG.",
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
    "--out_dir", default='/media/generalstorage3/lapstorage/out', type=str,
    help="Output directory for model checkpoints and logs."
)
parser.add_argument(
    "--devices", default=[0], type=int, nargs='+', help="GPU IDs to use when training the model"
)

ARGS = parser.parse_args()


def train(
    model, criterion, optimizer, scheduler,
    loader_train, loader_val, epochs, out_dir, writer,
    last_epoch, prime
    ):
    """Train and validate model after each epoch.

    :param nn.Sequential model: Model to train
    :param torch.nn.Loss criterion: Criterion
    :param torch.optim.Optimizer optimizer: Optimizer
    :param torch.optim.lr_scheduler.LRScheduler scheduler: Learning Rate Scheduler
    :param torch.utils.data.DataLoader loader_train: Training DataLoader
    :param torch.utils.data.DataLoader loader_val: Validation DataLoader
    :param int epochs: Epochs to train
    :param str out_dir: Output directory
    :param torch.utils.tensorboard.SummaryWriter writer: Tensorboard SummeryWriter
    :param int last_epoch: Last trained epoch from checkpoint (when not training from scratch)
    :param nn.Module prime: PRIME data augmentation module
    """    
    scaler = torch.cuda.amp.GradScaler(enabled=(not ARGS.no_amp))
    model_saver = ModelSaver(model, optimizer, scheduler, out_dir)
    epoch_iterator = range(last_epoch + 1, epochs) if last_epoch is not None else range(epochs)
    
    for epoch in epoch_iterator:
        # Training
        model.train()
        pred_train, labels_train, running_loss = [], [], []
        with tqdm.tqdm(total=len(loader_train), desc=f"Epoch {epoch}") as pbar:
            for x, y in loader_train:
                x, y = x.cuda(), y.cuda()
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(not ARGS.no_amp)):
                    y_hat = model(prime(x) if ARGS.with_prime else x)
                    loss_train = criterion(y_hat, y)
                optimizer.zero_grad()
                scaler.scale(loss_train).backward()
                scaler.unscale_(optimizer)
                grads = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                prev_scale = scaler.get_scale()
                scaler.update()
                if ARGS.no_amp or prev_scale <= scaler.get_scale():
                    scheduler.step()
                running_loss.append(loss_train.item())
                pred_train.append(torch.argmax(y_hat, dim=1).cpu().numpy())
                labels_train.append(y.cpu().numpy())
                pbar.update(1)
                pbar.set_postfix(loss=running_loss[-1], grad=torch.max(grads).item())

            pred_train = np.concatenate(pred_train)
            labels_train = np.concatenate(labels_train)
            accuracy_train = (pred_train == labels_train).sum() / labels_train.shape[0]

            # Validation
            model.eval()
            pred_val, labels_val, losses = [], [], []
            with torch.no_grad():
                for x, y in loader_val:
                    x, y = x.cuda(), y.cuda()
                    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(not ARGS.no_amp)):
                        y_hat = model(x)
                        loss_val = criterion(y_hat, y)
                    prediction = torch.argmax(y_hat, dim=1)
                    losses.append(loss_val.item())
                    pred_val.append(prediction.cpu().numpy())
                    labels_val.append(y.cpu().numpy())

            pred_val = np.concatenate(pred_val)
            labels_val = np.concatenate(labels_val)
            loss_val = np.mean(losses)
            accuracy_val = (pred_val == labels_val).sum() / labels_val.shape[0]
            
            writer.add_scalar("Training Loss", np.nanmean(running_loss), epoch)
            writer.add_scalar("Training Accuracy", accuracy_train, epoch)
            writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("Validation Loss", loss_val, epoch)
            writer.add_scalar("Validation Accuracy", accuracy_val, epoch)
            writer.flush()

            model_saver.save(epoch, accuracy_val)


def main():
    # Configs
    hyperparam_tag = ARGS.model_arch
    if ARGS.with_prime:
        hyperparam_tag += '+prime'
    with open(f'evnet/configs/{hyperparam_tag}_train.yml', "r") as f:
        hyperparams = yaml.safe_load(f)

    subcort_kwargs = get_subcort_kwargs(ARGS.subcorticalblock)
    vone_kwargs = get_vone_kwargs(ARGS.model_arch)    

    set_seed(ARGS.seed)
    exp_name = get_experiment_name(ARGS)

    # Output directory
    out_dir = os.path.join(ARGS.out_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isdir(out_dir):
        if not ARGS.use_ckpt:
            out_dir = os.path.join(
                out_dir,
                (exp_name + "_" + str(ARGS.seed) + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            )
            os.makedirs(out_dir)
        else:
            subdir_count = 0
            for subdir in os.listdir(out_dir):
                if os.path.isdir(os.path.join(out_dir, subdir)) and int(os.path.basename(os.path.normpath(subdir)).split('_')[-2]) == ARGS.seed:
                    subdir_count += 1
                    assert subdir_count == 1, 'More than one possible directory to retrieve checkpoint!'
                    out_dir = os.path.join(out_dir, subdir)
            assert subdir_count, 'No checkpoint directory found!'


    torch.cuda.set_device(f'cuda:{ARGS.devices[0]}' if torch.cuda.is_available() else 'cpu')

    # Data
    data_train = get_imagenet(ARGS.data_dir, subcort_kwargs, True, ARGS.with_prime)
    data_val = get_imagenet(ARGS.data_dir, subcort_kwargs, False, False)
    loader_train = DataLoader(
        data_train, batch_size=hyperparams["batch_size"], shuffle=True,
        num_workers=ARGS.num_workers, pin_memory=True,
        drop_last=True, prefetch_factor=ARGS.prefetch_factor
        )
    loader_val = DataLoader(
        data_val, batch_size=hyperparams["batch_size"], shuffle=False,
        num_workers=ARGS.num_workers, pin_memory=True,
        drop_last=False, prefetch_factor=ARGS.prefetch_factor
        )

    # Model
    model = EVNet(
        with_voneblock=ARGS.with_voneblock, model_arch=ARGS.model_arch,
        gabor_seed=ARGS.seed, **subcort_kwargs, **vone_kwargs, 
    )
    model.cuda()
        
    # Criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=hyperparams['label_smoothing'])
    
    # Optimizer
    all_params = list(model.named_parameters())
    param_groups = [
        {'params': [v for k, v in all_params if ('bn' in k)], 'weight_decay': 0.},
        {'params': [v for k, v in all_params if not ('bn' in k)], 'weight_decay': hyperparams['weight_decay']}
        ]
    optimizer = getattr(optim, hyperparams['optimizer'])(
        param_groups, lr=hyperparams['lr'], **hyperparams.get('optimizer_kwargs', {})
        )
    
    # BatchNorm Momentum
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = hyperparams['bn_momentum']
    
    # Scheduler
    scheduler = getattr(optim.lr_scheduler, hyperparams['scheduler'])(
        optimizer, **hyperparams['scheduler_kwargs']
        )

    # Checkpoint
    if ARGS.use_ckpt and ARGS.pretrained_ckpt is not None:
        raise ValueError('Two possible model checkpoints given.')
    ckpt = None
    if ARGS.pretrained_ckpt is not None:
        ckpt = torch.load(ARGS.pretrained_ckpt, map_location='cuda')
        model.load_state_dict(ckpt["model"])
    elif ARGS.use_ckpt:
        ckpt = torch.load(os.path.join(out_dir, "ckpt_best.pth"), map_location='cuda')
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    
    # Logger
    writer = SummaryWriter(out_dir, exp_name)
    print('Output directory:', out_dir)
  
    if len(ARGS.devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=ARGS.devices)
    
    # PRIME
    prime =  None if not ARGS.with_prime else get_prime(T.Normalize(*get_data_norm(subcort_kwargs, False)))

    # Train model
    train(
        model, criterion, optimizer, scheduler, loader_train,
        loader_val, hyperparams['epochs'], out_dir,
        writer, (ckpt['epoch'] if ARGS.use_ckpt else None),
        prime
        )

    writer.close()

if __name__ == "__main__":
    main()