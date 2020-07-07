from logging import getLogger
import torch


logger = getLogger(__name__)


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform):
        self.transform = transform
        self.data_num = 1024
        self.x = torch.arange(self.data_num)
        self.y = x % 10

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        x = self.x
        y = self.y
        return x, y
