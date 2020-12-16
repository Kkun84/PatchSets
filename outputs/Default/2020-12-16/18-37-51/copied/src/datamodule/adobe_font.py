from typing import Callable, Any
import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import random

from src.dataset import AdobeFontDataset


class AdobeFontDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        upper: bool,
        lower: bool,
        data_split_num: int,
        data_use_num: int,
        seed: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.path = path
        self.upper = upper
        self.lower = lower

        assert 0 <= data_use_num < data_split_num
        self.data_split_num = data_split_num
        self.data_use_num = data_use_num
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.pin_memory = True
        self.dims = (1, 100, 100)

        self.transforms = transforms.ToTensor()
        self.target_transform = lambda x: x['alphabet']

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        def f(i):
            return round(len(dataset.unique_font) * i / (self.data_split_num + 1))

        dataset = AdobeFontDataset(
            self.path, self.transforms, self.target_transform, self.upper, self.lower
        )

        r = random.Random(self.seed)
        index = list(range(len(dataset.unique_font)))
        r.shuffle(index)

        if stage == 'fit' or stage is None:
            train_index = []
            for i in range(self.data_split_num):
                if i != self.data_use_num:
                    train_index += index[f(i) : f(i + 1)]
            val_index = index[f(self.data_use_num) : f(self.data_use_num + 1)]
            self.train_dataset = AdobeFontDataset(
                self.path,
                self.transforms,
                self.target_transform,
                self.upper,
                self.lower,
                train_index,
            )
            self.val_dataset = AdobeFontDataset(
                self.path,
                self.transforms,
                self.target_transform,
                self.upper,
                self.lower,
                val_index,
            )
        if stage == 'test' or stage is None:
            test_index = index[f(self.data_split_num) :]
            self.test_dataset = AdobeFontDataset(
                self.path,
                self.transforms,
                self.target_transform,
                self.upper,
                self.lower,
                test_index,
            )
#             self.test_dataset = [i for i in self.test_dataset] * 16

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    datamodule = AdobeFontDataModule(
        path='/dataset/AdobeFontCharImages',
        upper=True,
        lower=True,
        data_split_num=5,
        data_use_num=0,
        seed=0,
        batch_size=64,
        num_workers=0,
    )
    print(datamodule)
    datamodule.prepare_data()
    datamodule.setup()
