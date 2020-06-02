from logging import getLogger
import hydra
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

from .patch import make_patch2d


logger = getLogger(__name__)


class Encoder(pl.LightningModule):

    def __init__(self, model_params):
        super().__init__()
        self.model_params = model_params
        self.hparams = {}

        self.input_n = np.prod(self.model_params.input_shape)
        hidden_n_0 = self.model_params.hidden_n_0
        self.output_n = self.model_params.output_n

        self.linear_0 = torch.nn.Linear(self.input_n, hidden_n_0)
        self.linear_1 = torch.nn.Linear(hidden_n_0, self.output_n)

    def forward(self, input):
        # [batch, sets, channels, x, y]
        logger.debug(f"input.shape={input.shape}")
        logger.debug([input.shape[0]*input.shape[1], self.input_n])
        x = input.reshape([input.shape[0]*input.shape[1], self.input_n])
        logger.debug(f"reshape-x.shape={x.shape}")
        x = self.linear_0(x)
        logger.debug(f"linear_0-x.shape={x.shape}")
        x = F.relu(x)
        x = self.linear_1(x)
        logger.debug(f"linear_1-x.shape={x.shape}")
        x = x.reshape([input.shape[0], input.shape[1], self.output_n])
        logger.debug(f"reshape-x.shape={x.shape}")
        x = x.mean(1)
        logger.debug(f"mean-x.shape={x.shape}")
        # [batch, lattent]
        return x


class Decoder(pl.LightningModule):

    def __init__(self, model_params):
        super().__init__()
        self.model_params = model_params
        self.hparams = {}

        self.input_n = self.model_params.input_n
        hidden_n_0 = self.model_params.hidden_n_0
        self.output_n = self.model_params.output_n

        self.linear_0 = torch.nn.Linear(self.input_n, hidden_n_0)
        self.linear_1 = torch.nn.Linear(hidden_n_0, self.output_n)

    def forward(self, input):
        # [batch, lattent]
        logger.debug(f"input.shape={input.shape}")
        x = self.linear_0(input)
        logger.debug(f"linear_0-x.shape={x.shape}")
        x = F.relu(x)
        x = self.linear_1(x)
        logger.debug(f"linear_1-x.shape={x.shape}")
        return x


class Model(pl.LightningModule):

    def __init__(self, model_params, hparams, optim, data_path):
        super().__init__()
        self.hparams = argparse.Namespace(**hparams)
        self.optim = optim
        self.data_path = data_path

        self.encoder = Encoder(model_params.encoder)
        self.decoder = Decoder(model_params.decoder)

        self.input_n = self.encoder.input_n
        self.output_n = self.decoder.output_n

    def forward(self, input):
        logger.debug(f"input.shape={input.shape}")
        z = self.encoder(input)
        logger.debug(f"z.shape={z.shape}")
        output = self.decoder(z)
        logger.debug(f"output.shape={output.shape}")
        return output

    def configure_optimizers(self):
        logger.debug('configure_optimizers')
        optimizer = hydra.utils.instantiate(self.optim.optimizer, self.parameters())
        if self.optim.lr_scheduler is None:
            return [optimizer], []
        lr_scheduler = hydra.utils.instantiate(self.optim.lr_scheduler, self.parameters())
        return [optimizer], [lr_scheduler]

    def prepare_data(self):
        logger.debug('prepare_data')
        train_dataset = MNIST(root=self.data_path, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = MNIST(root=self.data_path, train=False, download=True, transform=transforms.ToTensor())

        train_dataset, val_dataset = random_split(train_dataset, [55000, 5000])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        logger.debug('train_dataloader')
        batch_size = self.hparams.batch_size
        num_workers = self.hparams.num_workers
        return DataLoader(self.train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    def val_dataloader(self):
        logger.debug('val_dataloader')
        batch_size = self.hparams.batch_size
        num_workers = self.hparams.num_workers
        return DataLoader(self.val_dataset, batch_size=batch_size, num_workers=num_workers)

    def test_dataloader(self):
        logger.debug('test_dataloader')
        batch_size = self.hparams.batch_size
        num_workers = self.hparams.num_workers
        return DataLoader(self.test_dataset, batch_size=batch_size, num_workers=num_workers)

    def training_step(self, batch, batch_idx):
        logger.debug(f'training_step-{batch_idx}')
        x, y = batch
        y_hat = self(make_patch2d(x, self.hparams.patch_size, self.hparams.train_patch_n))
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        logger.debug(f'validation_step-{batch_idx}')
        x, y = batch
        y_hat = self(make_patch2d(x, self.hparams.patch_size, self.hparams.test_patch_n))
        correct = (y == y_hat.argmax(1)).float()
        return {'loss': F.cross_entropy(y_hat, y, reduction='sum'), 'correct': correct}

    def validation_epoch_end(self, outputs):
        logger.debug('validation_epoch_end')
        avg_loss = torch.stack([x['loss'] for x in outputs]).sum() / len(self.val_dataset)
        accuracy = torch.cat([x['correct'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_accuracy': accuracy}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        logger.debug(f'test_step-{batch_idx}')
        x, y = batch
        y_hat = self(make_patch2d(x, self.hparams.patch_size, self.hparams.test_patch_n))
        correct = (y == y_hat.argmax(1)).float()
        return {'loss': F.cross_entropy(y_hat, y, reduction='sum'), 'correct': correct}

    def test_epoch_end(self, outputs):
        logger.debug('test_epoch_end')
        avg_loss = torch.stack([x['loss'] for x in outputs]).sum() / len(self.test_dataset)
        accuracy = torch.cat([x['correct'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss.item(), 'test_accuracy': accuracy.item()}
        if self.logger is not None:
            self.logger.log_metrics(tensorboard_logs, self.global_step)
        return {'test_loss': avg_loss, 'log': tensorboard_logs}
