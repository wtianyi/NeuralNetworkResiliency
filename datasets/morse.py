from typing import Tuple
from torch.utils.data import Dataset, TensorDataset

mat_path = "./data/Morse_trainning_set.mat"

def get_morse_datasets() -> Tuple[Dataset, Dataset, int]:
    from scipy.io import matlab

    data = matlab.loadmat(mat_path)
    x = torch.from_numpy(data["images"].T.astype("float32"))
    y = torch.from_numpy(data["labels"].astype("int").ravel())
    x_test = torch.from_numpy(data["t_images"].T.astype("float32"))
    y_test = torch.from_numpy(data["t_labels"].astype("int").ravel())

    num_classes = torch.unique(y).nelement()

    train_set = TensorDataset(x, y)
    test_set = TensorDataset(x_test, y_test)

    return train_set, test_set, num_classes
