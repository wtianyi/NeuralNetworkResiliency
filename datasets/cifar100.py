from torch.utils.data.dataloader import DataLoader
from .utils import DeficitDataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)

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
    trainset = CIFAR100(
        root="./data", train=True, download=True, transform=_transform_train_CIFAR
    )
    deficit_trainset = CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=_transform_train_CIFAR_deficit,
    )
    testset = CIFAR100(
        root="./data", train=False, download=True, transform=_transform_test_CIFAR
    )
    num_classes = 100

    train_loader = DeficitDataLoader(trainset, deficit_trainset, **kwargs)
    test_loader = DataLoader(testset, **kwargs)
    return train_loader, test_loader, num_classes
