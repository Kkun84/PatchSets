from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from logging import getLogger


logger = getLogger(__name__)


def MNIST(root, transform=None, target_transform=None):
    train = MNISTdataset(root=root, type_='train', transform=transform, target_transform=target_transform)
    test = MNISTdataset(root=root, type_='test', transform=transform, target_transform=target_transform)
    return train, test


class MNISTdataset(torch.utils.data.Dataset):
    def __init__(self, root, type_, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.root = Path(root) / 'MNIST'
        self.type_ = type_

        self.data_property = pd.read_csv(self.root / 'data_property.csv')
        self.data = []
        for _, item in self.data_property.iterrows():
            image = Image.open(self.root / item.path)
            if self.type_ == item.type_:
                self.data.append((image, item.class_))
        assert len(self.data) > 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y
