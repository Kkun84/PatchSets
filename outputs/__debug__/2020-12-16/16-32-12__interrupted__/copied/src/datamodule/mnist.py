from typing import Callable, Any
import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        data_split_num: int,
        data_use_num: int,
        seed: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.path = path

        assert 0 <= data_use_num < data_split_num
        self.data_split_num = data_split_num
        self.data_use_num = data_use_num
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.pin_memory = True
        self.dims = (1, 28, 28)

    def prepare_data(self) -> None:
        MNIST(self.path, train=True, download=False)
        MNIST(self.path, train=False, download=False)

    def setup(self, stage: str = None) -> None:
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.path, train=True, transform=transforms.ToTensor())

            def f(i):
                return round(len(mnist_full) * i / (self.data_split_num + 1))
            r = random.Random(self.seed)
            index = list(range(len(mnist_full)))
            r.shuffle(index)
            train_index = []
            for i in range(self.data_split_num):
                if i != self.data_use_num:
                    train_index += index[f(i) : f(i + 1)]
            val_index = index[f(self.data_use_num) : f(self.data_use_num + 1)]

            self.train_dataset = Subset(mnist_full, train_index)
            self.val_dataset = Subset(mnist_full, val_index)
        if stage == 'test' or stage is None:
            self.test_dataset = MNIST(
                self.path, train=False, transform=transforms.ToTensor()
            )

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
    datamodule = MNISTDataModule(
        path='/dataset/MNIST',
        data_split_num=12,
        data_use_num=0,
        seed=0,
        batch_size=64,
        num_workers=0
    )
    print(datamodule)
    datamodule.prepare_data()
