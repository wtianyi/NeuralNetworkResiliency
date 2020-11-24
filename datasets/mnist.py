from torch.utils.data import DataLoader
from .utils import DeficitDataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

mean = (0.1307,)
std = (0.3081,)
meta = {}

_transform_train_MNIST_deficit = transforms.Compose(
    [
        transforms.Resize([7, 7]),
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ]
)  # meanstd transformation
_transform_train_MNIST = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ]
)  # meanstd transformation
_transform_test_MNIST = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ]
)
_transform_test_MNIST_deficit = transforms.Compose(
    [
        transforms.Resize([7, 7]),
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ]
)


def get_dataloader_helper(**kwargs):
    trainset = MNIST(
        root="./data", train=True, download=True, transform=_transform_train_MNIST
    )
    deficit_trainset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=_transform_train_MNIST_deficit,
    )
    testset = MNIST(
        root="./data", train=False, download=True, transform=_transform_test_MNIST
    )
    deficit_testset = MNIST(
        root="./data", train=False, download=True, transform=_transform_test_MNIST_deficit
    )
    num_classes = 10

    train_loader = DeficitDataLoader(trainset, deficit_trainset, **kwargs)
    test_loader = DeficitDataLoader(testset, deficit_testset, **kwargs)

    return train_loader, test_loader, num_classes
