from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader

meta = {}
mat_path = "./data/Morse_trainning_set.mat"

def get_dataloader_helper(**kwargs) -> Tuple[DataLoader, DataLoader, int]:
    from scipy.io import matlab

    data = matlab.loadmat(mat_path)
    x = torch.from_numpy(data["images"].T.astype("float32"))
    y = torch.from_numpy(data["labels"].astype("int").ravel())
    x_test = torch.from_numpy(data["t_images"].T.astype("float32"))
    y_test = torch.from_numpy(data["t_labels"].astype("int").ravel())

    num_classes = torch.unique(y).nelement()

    train_set = TensorDataset(x, y)
    test_set = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_set, **kwargs)
    test_loader = DataLoader(test_set, **kwargs)
    return train_loader, test_loader, num_classes
