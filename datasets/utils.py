from torch.utils.data import Dataset, DataLoader


class DeficitDataLoader(DataLoader):
    """Switch from two given datasets.

    After switch, any created iterators will, of course, become invalid.
    """

    def __init__(self, original_dataset: Dataset, deficit_dataset: Dataset, **kwargs):
        # by default use the impaired dataset
        self.original_dataset = original_dataset
        self.deficit_dataset = deficit_dataset
        # self.state = "impaired"
        self.kwargs = kwargs
        self.dataset = self.deficit_dataset
        self.dataloader = DataLoader(self.deficit_dataset, **self.kwargs)

    @property
    def state(self):
        if self.dataset is self.original_dataset:
            return "cured"
        elif self.dataset is self.deficit_dataset:
            return "impaired"
        else:
            raise ValueError(f"Invalid state: {self.state}")

    def impair(self):
        if self.state != "impaired":
            self.dataset = self.deficit_dataset
            self.dataloader = DataLoader(self.deficit_dataset, **self.kwargs)

    def cure(self):
        if self.state != "cured":
            self.dataset = self.original_dataset
            self.dataloader = DataLoader(self.original_dataset, **self.kwargs)

    def switch(self):
        if self.state == "impaired":
            self.cure()
        elif self.state == "cured":
            self.impair()
        else:
            raise ValueError(f"Invalid state: {self.state}")

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return iter(self.dataloader)
