from typing import Tuple
from . import mnist
from . import cifar10
from . import cifar100
from torch.utils.data import DataLoader
import sys

helper_dict = {
    "mnist": mnist.get_dataloader_helper,
    "cifar10": cifar10.get_dataloader_helper,
    "cifar100": cifar100.get_dataloader_helper,
}


def get_dataloader(dataset: str, **kwargs) -> Tuple[DataLoader, DataLoader, int]:
    """
    Get train and test datasets by dataset name
    choices are 'mnist', 'cifar10', 'cifar100'

    Args:
        dataset (:obj:`str`): The dataset name
        **kwargs: other arguments will be passed to :obj:`DataLoader`

    Return:
        train_loader (:obj:`torch.utils.data.DataLoader`):
            The training data loader
        test_loader (:obj:`torch.utils.data.DataLoader`):
            The test data loader
        num_classes (:obj:`int`):
            The number of classes in the dataset
    """
    if dataset not in helper_dict:
        raise ValueError(f"Unknown dataset ({dataset})")
    print(f"| Preparing {dataset} dataset...")
    sys.stdout.write("| ")
    return helper_dict[dataset](**kwargs)