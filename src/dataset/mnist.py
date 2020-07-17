from tqdm import tqdm
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from logging import getLogger


logger = getLogger(__name__)


def MNIST(root, transform=None, target_transform=None):
    logger.debug('MNIST({}, {}, {})'.format(root, transform, target_transform))
    train = MNISTdataset(root=root, type_='train', transform=transform, target_transform=target_transform)
    test = MNISTdataset(root=root, type_='test', transform=transform, target_transform=target_transform)
    return train, test


class MNISTdataset(torch.utils.data.Dataset):
    def __init__(self, root, type_, transform=None, target_transform=None):
        logger.debug('MNISTdataset({}, {}, {}, {})'.format(root, type_, transform, target_transform))
        self.transform = transform
        self.target_transform = target_transform
        self.root = Path(root) / 'MNIST'
        self.type_ = type_

        data_property = pd.read_csv(self.root / 'data_property.csv')
        self.data_property = data_property[data_property.type_ == self.type_]
        self.data = []
        for _, item in tqdm(self.data_property.iterrows(), total=len(self.data_property)):
            # logger.debug('Image.open({})'.format(self.root / item.path))
            image = Image.open(self.root / item.path)
            self.data.append((image, item.class_))
        assert len(self.data) > 0

    def set_transform(self, transform=None, target_transform=None):
        if transform is not None:
            self.transform = transform
        if target_transform is not None:
            self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        logger.debug(f'MNISTdataset.__getitem__({index})), {self.root}, {self.type_}, {self.transform}, {self.target_transform}')
        x, y = self.data[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        logger.debug(f'{type(x)}, {type(y)}')
        return x, y
