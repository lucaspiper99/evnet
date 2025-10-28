import os
import torch

class ModelSaver:
    def __init__(self, model, optimizer, scheduler, out_dir):
        self.best_acc = 0
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.out_dir = out_dir

    def save(self, epoch, acc):
        """Saves model checkpoint.

        Args:
            epochs (int): checkpoint epoch
            acc (float): ImageNet validation top-1 accuracy of the model
        """
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "val_accuracy": acc
        }
        torch.save(checkpoint, os.path.join(self.out_dir, "ckpt.pth"))
        if acc > self.best_acc:
            torch.save(checkpoint, os.path.join(self.out_dir, "ckpt_best.pth"))
            self.best_acc = acc


def get_experiment_name(args, init=True):
    """Creates experiment name based on architecture, data augmentation and inference ensembling.

    Args:
        args (obj): parsed training or testing arguments
        init (bool): Whether creating the name for the 1st time
    Returns:
        experiment name (str)
    """
    if getattr(args, 'model_ckpt', '') != '':
        return os.path.basename(args.model_ckpt.split('.')[0])

    exp_name = ''

    if args.with_voneblock and args.subcorticalblock == '':
        exp_name += 'vone'
    
    if args.subcorticalblock != '':
        exp_name += 'ev'
    
    exp_name += args.model_arch
    
    if args.subcorticalblock != '' and not args.with_voneblock:
        exp_name += '_no_v1'
    
    if args.subcorticalblock not in ('', 'standard'):
        exp_name += '_' + args.subcorticalblock

    if args.with_prime:
        exp_name += '+prime'

    if init and hasattr(args, 'ensemble_type') and args.ensemble_type is not None:
            exp_name += f'_ens_{args.ensemble_type}_{args.ensemble_size}'

    return exp_name


def get_model_ckpts(args):
    """Retrieves model checkpoints from output dir based on experiment name.

    Args:
        args (obj): parsed training or testing arguments
    Returns:
        List of tuples (seed (int), subdirectory (str))
    """
    directory = os.path.join(args.ckpt_dir, get_experiment_name(args, init=False))
    subdirs = []
    exp_name = get_experiment_name(args)
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path) and exp_name in subdir_path:
            seed = int(os.path.basename(os.path.normpath(subdir)).split('_')[-2])
            if args.seeds is not None and seed not in args.seeds:
                continue
            subdirs.append((
                    int(os.path.basename(os.path.normpath(subdir)).split('_')[-2]),
                    os.path.join(directory, subdir, 'ckpt_best.pth')
                    ))
    return subdirs