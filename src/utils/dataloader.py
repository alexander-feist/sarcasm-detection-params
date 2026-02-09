from torch.utils.data import DataLoader, random_split

from src.dataset.dataset import Dataset


def get_dataloaders(
    dataset: Dataset, train_test_split: float, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get train, valid, and test dataloaders."""
    dataset_size = len(dataset)
    train_size = int(train_test_split * dataset_size)
    remaining = dataset_size - train_size
    valid_size = remaining // 2
    test_size = remaining - valid_size
    train_d, valid_d, test_d = random_split(dataset, [train_size, valid_size, test_size])

    train_dl = DataLoader(train_d, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_d, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_d, batch_size=batch_size, shuffle=False)

    return train_dl, valid_dl, test_dl
