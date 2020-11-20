from torch.utils.data import DataLoader
from .utils import DeficitDataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

_transform_train_CIFAR_deficit = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize([8, 8]),
        transforms.Resize([32, 32]),
        #     transforms.GaussianBlur(7,5),
        transforms.ToTensor(),
        #     transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)

_transform_train_CIFAR = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
    ]
)  # meanstd transformation

_transform_test_CIFAR = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
    ]
)


def get_dataloader_helper(**kwargs):
    trainset = CIFAR10(
        root="./data", train=True, download=True, transform=_transform_train_CIFAR
    )
    deficit_trainset = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=_transform_train_CIFAR_deficit,
    )
    testset = CIFAR10(
        root="./data", train=False, download=True, transform=_transform_test_CIFAR
    )
    num_classes = 10

    train_loader = DeficitDataLoader(trainset, deficit_trainset, **kwargs)
    test_loader = DataLoader(testset, **kwargs)
    return train_loader, test_loader, num_classes
