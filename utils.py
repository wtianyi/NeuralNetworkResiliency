from typing import Tuple
from networks import *
import config as cf
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, TensorDataset
import sys
import os
import shutil
import numpy as np
import random


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


def get_morse_datasets(mat_path: str) -> Tuple[Dataset, Dataset, int]:
    from scipy.io import matlab
    data = matlab.loadmat(mat_path)
    x = torch.from_numpy(data['images'].T.astype('float32'))
    y = torch.from_numpy(data['labels'].astype('int').ravel())
    x_test = torch.from_numpy(data['t_images'].T.astype('float32'))
    y_test = torch.from_numpy(data['t_labels'].astype('int').ravel())

    num_classes = torch.unique(y).nelement()

    train_set = TensorDataset(x, y)
    test_set = TensorDataset(x_test, y_test)

    return train_set, test_set, num_classes


def get_datasets(dataset: str):
    if dataset == 'morse':
        print("| Preparing Morse dataset...")
        sys.stdout.write("| ")
        return get_morse_datasets('./data/Morse_trainning_set.mat')
    transform_train_CIFAR = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
    ])  # meanstd transformation

    transform_train_MNIST = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
    ])  # meanstd transformation

    transform_test_CIFAR = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
    ])

    transform_test_MNIST = transforms.Compose([
        # transforms.Pad(padding=2, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
    ])
    if dataset == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train_CIFAR)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test_CIFAR)
        num_classes = 10
    elif dataset == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train_CIFAR)
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test_CIFAR)
        num_classes = 100
    elif dataset == 'mnist':
        print("| Preparing MNIST dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform_train_MNIST)
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform_test_MNIST)
        num_classes = 10
    else:
        raise ValueError(f"Unrecognized dataset ({dataset})")
    return trainset, testset, num_classes


def get_network_name(args):
    parts = []
    parts.append(args.net_type)
    if args.training_noise_type == 'gaussian' and args.training_noise is None:
        parts.append('noise_std-[0.0]')
    elif args.training_noise_type == 'uniform':
        parts.append('noise-uniform')
    else:
        parts.append('noise_std-{}'.format(args.training_noise))
    if args.training_noise_mean is None:
        parts.append('noise_mean-[0.0]')
    else:
        parts.append('noise_mean-{}'.format(args.training_noise_mean))

    parts.append('{}-{}'.format(args.regularization_type, args.regularization))
    parts.append('dropout-{}'.format(args.dropout_rate))
    parts.append('lr-{}'.format(args.lr))
    parts.append('epochs-{}'.format(args.num_epochs))
    parts.append('lrdecayepoch-{}'.format(args.epochs_lr_decay))
    if args.forward_samples != 1:
        parts.append('forward-{}'.format(args.forward_samples))
    if args.optim_type == "EntropySGD":
        parts.append('entropySGD')
    if args.run_name:
        parts.append(args.run_name)

    file_name = '_'.join(parts)
    return file_name


# Return network & file name
def _get_network(net_type: str, depth: int, dropout_rate: float, dataset: str, num_classes: int,
                 widen_factor: int = 1, training_noise_type: str = "gaussian", training_noise: float = None):
    if net_type == 'lenet':
        if dataset == 'mnist':
            net = LeNet(num_classes, input_size=28, input_channel=1)
        elif dataset == 'cifar10':
            net = LeNet(num_classes, input_size=32, input_channel=3)
        else:
            raise ValueError(f"Unrecognized dataset ({dataset}) for lenet")
    elif net_type == 'resnet':
        if dataset == 'mnist':
            net = ResNet(depth, num_classes, use_dropout=True,
                         dropout_rate=dropout_rate, in_channel=1)
        else:
            net = ResNet(depth, num_classes, use_dropout=True,
                         dropout_rate=dropout_rate, in_channel=3)
    elif net_type == 'wide_resnet':
        net = WideResNet(depth, widen_factor, dropout_rate, num_classes)
    elif net_type == 'mlp':
        if dataset == 'morse':
            net = MLP(20, 20, [20])
        else:
            raise ValueError(f"Unrecognized dataset ({dataset}) for mlp")
    else:
        raise ValueError(f"Unrecognized net_type ({net_type})")

    if training_noise_type == 'gaussian' and training_noise is None:
        net.apply(set_gaussian_noise)
    elif training_noise_type == 'uniform':
        net.apply(set_uniform_noise)
    else:
        net.apply(set_gaussian_noise)
    return net


def get_network(args, num_classes: int):
    net = _get_network(net_type=args.net_type, depth=args.depth, dropout_rate=args.dropout_rate,
                       dataset=args.dataset, num_classes=num_classes, widen_factor=args.widen_factor,
                       training_noise_type=args.training_noise_type, training_noise=args.training_noise)
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
