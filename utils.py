from typing import Tuple
from networks import *
import torch
import os
import shutil
import numpy as np
import random
import argparse


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators
    Args:
        seed (int): The random seed
        using_cuda (bool): Whether torch.cuda needs seed fixed
    """
    # Seed python RNG
    try:
        random.seed(seed)
    except:
        pass
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        torch.cuda.manual_seed(seed)
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def get_network_name(args):
    parts = []
    parts.append(args.net_type)
    if args.training_noise_type == "gaussian" and args.training_noise is None:
        parts.append("noise_std-[0.0]")
    elif args.training_noise_type == "uniform":
        parts.append("noise-uniform")
    else:
        parts.append("noise_std-{}".format(args.training_noise))
    if args.training_noise_mean is None:
        parts.append("noise_mean-[0.0]")
    else:
        parts.append("noise_mean-{}".format(args.training_noise_mean))

    parts.append("{}-{}".format(args.regularization_type, args.regularization))
    parts.append("dropout-{}".format(args.dropout_rate))
    parts.append("lr-{}".format(args.lr))
    parts.append("epochs-{}".format(args.num_epochs))
    parts.append("lrdecayepoch-{}".format(args.epochs_lr_decay))
    parts.append("deficitepoch-{}".format(args.deficit_epochs))
    if args.forward_samples != 1:
        parts.append("forward-{}".format(args.forward_samples))
    if args.optim_type == "EntropySGD":
        parts.append("entropySGD")
    if args.run_name:
        parts.append(args.run_name)

    file_name = "_".join(parts)
    return file_name


# Return network & file name
def _get_network(
    net_type: str,
    depth: int,
    dropout_rate: float,
    dataset: str,
    num_classes: int,
    widen_factor: int = 1,
    training_noise_type: str = "gaussian",
    training_noise: float = None,
):
    if net_type == "lenet":
        if dataset == "mnist":
            net = LeNet(num_classes, input_size=28, input_channel=1)
        elif dataset == "cifar10":
            net = LeNet(num_classes, input_size=32, input_channel=3)
        else:
            raise ValueError(f"Unrecognized dataset ({dataset}) for lenet")
    elif net_type == "resnet":
        if dataset == "mnist":
            net = ResNet(
                depth,
                num_classes,
                use_dropout=False,
                # dropout_rate=dropout_rate,
                in_channel=1,
            )
        else:
            net = ResNet(
                depth,
                num_classes,
                use_dropout=True,
                dropout_rate=dropout_rate,
                in_channel=3,
            )
    elif net_type == "wide_resnet":
        net = WideResNet(depth, widen_factor, dropout_rate, num_classes)
    elif net_type == "mlp":
        if dataset == "morse":
            net = MLP(20, 20, [20])
        else:
            raise ValueError(f"Unrecognized dataset ({dataset}) for mlp")
    else:
        raise ValueError(f"Unrecognized net_type ({net_type})")

    if training_noise_type == "gaussian" and training_noise is None:
        net.apply(set_gaussian_noise)
    elif training_noise_type == "uniform":
        net.apply(set_uniform_noise)
    else:
        net.apply(set_gaussian_noise)
    return net


def get_network(args, num_classes: int) -> Tuple[nn.Module, str]:
    """
    Get network model with args and number of classes to classify

    Args:
        args (:obj:`argparse.Namespace`):
            The arguments
        num_classes (:obj:`int`):
            The number of classes

    Return:
        network (:obj:`torch.nn.Module`):
            The network model
        file_name (:obj:`str`):
            A file name with the arguments composed in
    """
    net = _get_network(
        net_type=args.net_type,
        depth=args.depth,
        dropout_rate=args.dropout_rate,
        dataset=args.dataset,
        num_classes=num_classes,
        widen_factor=args.widen_factor,
        training_noise_type=args.training_noise_type,
        training_noise=args.training_noise,
    )
    file_name = get_network_name(args)
    return net, file_name


def create_or_clear_dir(path, force=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            # try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            # except Exception as e:
            # print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        if force:
            os.unlink(path)
            os.makedirs(path)
        else:
            raise NotADirectoryError(f"{path} is a file")


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is a file")


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def classwise_accuracy(
    output: torch.Tensor, target: torch.Tensor, num_classes: int = None, topk: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the precision@k for each class respectively

    Args:
        output (:obj:`torch.Tensor`): The logits or probs as the classifier output
        target (:obj:`torch.Tensor`): The ground truth labels
        num_classes (`int`): The number of classes
        topk (`int`): The k to computer precision@k for

    Returns:
        accuracy (:obj:`torch.Tensor`): The accuracy, i.e. the precision@k for
            each class
        counts (:obj:`torch.Tensor`): The number of data points of each class
    """
    _, pred = output.topk(topk, dim=1, largest=True, sorted=True)
    correct = pred.T.eq(target.view(-1)).sum(dim=0).bool()
    counts = target.bincount(minlength=num_classes)
    accuracy = target.bincount(weights=correct, minlength=num_classes) / counts
    accuracy[accuracy.isnan()] = 0 # handle zero-division
    # classes = torch.arange(counts.size(0))
    return accuracy, counts  # , classes


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def boolean_flag(
    parser: argparse.ArgumentParser, name: str, default: bool = False, help: str = None
) -> None:
    """Add a boolean flag to argparse parser.

    Parameters
    ----------
    parser (:obj:`argparse.ArgumentParser`):
        parser to add the flag to
    name (:obj:`str`):
        --<name> will enable the flag, while --no-<name> will disable it
    default (:obj:`bool`):
        default value of the flag
    help (:obj:`str`):
        help string for the flag
    """
    dest = name.replace("-", "_")
    parser.add_argument(
        "--" + name, action="store_true", default=default, dest=dest, help=help
    )
    parser.add_argument("--no-" + name, action="store_false", dest=dest)
