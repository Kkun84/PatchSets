from typing import Any, Dict, List
import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data

import src.set_module as sm


class Model(pl.LightningModule):
    def __init__(
        self,
        patch_size: int,
        hidden_n: int,
        output_n: int,
        pool_mode: str,
        patch_num_min: int,
        patch_num_max: int,
        seed: int,
        batch_size: int,
        num_workers: int,
        max_epochs: int,
        min_epochs: int,
        patience: int,
        optimizer: str,
        lr: float,
        data_split_num: int,
        data_use_num: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert (
            patch_num_min <= patch_num_max
        ), f'patch_num_min={patch_num_min}, patch_num_max={patch_num_max}'

        self.accuracy = pl.metrics.Accuracy()

        self.squeeze_from_set = sm.SqueezeFromSet()

        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(patch_size ** 2, hidden_n)

        self.set_pooling_keep = sm.SetPooling(pool_mode, keep_shape=True)

        self.linear_2_a = nn.Linear(hidden_n, hidden_n)
        self.linear_2_b = nn.Linear(hidden_n, hidden_n)

        self.linear_3_a = nn.Linear(hidden_n, hidden_n)
        self.linear_3_b = nn.Linear(hidden_n, hidden_n)

        self.set_pooling = sm.SetPooling(pool_mode)
        self.linear_4 = nn.Linear(hidden_n, hidden_n)
        self.linear_5 = nn.Linear(hidden_n, output_n)

    @auto_move_data
    def forward(self, batch_set: sm.BatchSetType) -> Tensor:
        x, index = self.squeeze_from_set(batch_set)
        x = self.flatten(x)
        x = self.linear_1(x).relu()
        x = (
            self.linear_2_a(x).relu()
            + self.linear_2_b(self.set_pooling_keep(x, index)[0]).relu()
        )
        x = (
            self.linear_3_a(x).relu()
            + self.linear_3_b(self.set_pooling_keep(x, index)[0]).relu()
        )
        x, index = self.set_pooling(x, index)
        x = self.linear_4(x).relu()
        x = self.linear_5(x)
        output = x
        return output

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = getattr(optim, self.hparams.optimizer)(
            self.parameters(), lr=self.hparams.lr
        )
        return optimizer

    def _step(self, batch: List[Tensor]) -> Dict[str, Any]:
        x, y = batch
        batch_size = len(y)
        patch_sets = sm.cutout_patch2d(
            x,
            torch.randint(
                self.hparams.patch_num_min, self.hparams.patch_num_max + 1, [batch_size]
            ),
            self.hparams.patch_size,
        )
        y_hat = self.forward(patch_sets)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        return {'batch_size': batch_size, 'loss': loss, 'accuracy': accuracy}

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        items = self._step(batch)
        loss = items['loss']
        accuracy = items['accuracy']
        self.log('train_loss', loss, on_step=True)
        self.log('train_accuracy', accuracy, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> Dict[str, Any]:
        items = self._step(batch)
        batch_size = items['batch_size']
        loss = items['loss'] * batch_size
        accuracy = items['accuracy'] * batch_size
        return {
            'batch_size': batch_size,
            'loss': loss,
            'accuracy': accuracy,
        }

    def test_step(self, batch: List[Tensor], batch_idx: int) -> Dict[str, Any]:
        items = self._step(batch)
        batch_size = items['batch_size']
        loss = items['loss'] * batch_size
        accuracy = items['accuracy'] * batch_size
        return {
            'batch_size': batch_size,
            'loss': loss,
            'accuracy': accuracy,
        }

    def _epoch_end(self, step_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        data_size = sum([i['batch_size'] for i in step_outputs])
        loss = sum([i['loss'] for i in step_outputs]) / data_size
        accuracy = sum([i['accuracy'] for i in step_outputs]) / data_size
        return {'loss': loss, 'accuracy': accuracy}

    def validation_epoch_end(self, step_outputs: List[Dict[str, Any]]):
        items = self._epoch_end(step_outputs)
        loss = items['loss']
        accuracy = items['accuracy']
        self.log('valid_loss', loss, prog_bar=True)
        self.log('valid_accuracy', accuracy, prog_bar=True)

    def test_epoch_end(self, step_outputs: List[Dict[str, Any]]):
        items = self._epoch_end(step_outputs)
        loss = items['loss']
        accuracy = items['accuracy']
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)


if __name__ == '__main__':
    image_size = 10
    patch_size = 3
    hidden_n = 64
    output_n = 2
    patch_num_min = 1
    patch_num_max = 5

    model = Model(patch_size, hidden_n, output_n, patch_num_min, patch_num_max)

    batch_size = 5
    image = (
        torch.arange(image_size ** 2)
        .float()
        .reshape(1, 1, *[image_size] * 2)
        .expand(batch_size, -1, -1, -1)
    )
    x = sm.cutout_patch2d(
        image,
        torch.randint(patch_num_min, patch_num_max + 1, [batch_size]),
        patch_size,
    )
    y = model(x)

    print('image', image.shape)
    print('x', *[i.shape for i in x], '', sep='\n')
    print('y', y.shape)
