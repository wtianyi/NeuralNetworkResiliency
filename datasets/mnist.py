from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

mean = (0.1307,)
std = (0.3081,)
meta = {}

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


def get_dataloader_helper(**kwargs):
    trainset = MNIST(
        root="./data", train=True, download=True, transform=_transform_train_MNIST
    )
    testset = MNIST(
        root="./data", train=False, download=True, transform=_transform_test_MNIST
    )
    num_classes = 10

    train_loader = DataLoader(trainset, **kwargs)

    test_loader = DataLoader(testset, **kwargs)
    return train_loader, test_loader, num_classes
