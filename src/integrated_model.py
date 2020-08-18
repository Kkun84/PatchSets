from logging import getLogger
import hydra
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import pytorch_lightning as pl

from src.patch import make_patch2d
from src.model import Encoder, Decoder


logger = getLogger(__name__)


class IntegratedModel(pl.LightningModule):

    def __init__(self, hparams, encoder, decoder, optim, dataset):
        logger.debug('Model({}, {}, {}, {}, {})'.format(hparams, encoder, decoder, optim, dataset))
        super().__init__()
        self.hparams = hparams
        self.optim = optim
        self.dataset = dataset

        assert isinstance(encoder, Encoder)
        self.encoder = encoder
        assert isinstance(decoder, Decoder)
        self.decoder = decoder

        self.input_n = self.encoder.input_n
        self.output_n = self.decoder.output_n

    def forward(self, input):
        logger.debug('forward({})'.format(input))
        logger.debug(f"input.shape={input.shape}")
        z = self.encoder(input)
        logger.debug(f"z.shape={z.shape}")
        z = self.pooling(z)
        output = self.decoder(z)
        logger.debug(f"output.shape={output.shape}")
        return output

    def pooling(self, x, mode=None):
        if mode is None:
            mode = self.hparams.pooling_mode
        if mode == 'sum':
            x = x.sum(1)
        elif mode == 'mean':
            x = x.mean(1)
        elif mode == 'max':
            x = x.max(1)[0]
        return x

    def configure_optimizers(self):
        logger.debug('configure_optimizers()')
        optimizer = hydra.utils.instantiate(self.optim.optimizer, self.parameters())
        if self.optim.lr_scheduler is None:
            return [optimizer], []
        lr_scheduler = hydra.utils.instantiate(self.optim.lr_scheduler, self.parameters())
        return [optimizer], [lr_scheduler]

    def prepare_data(self):
        logger.debug('prepare_data()')
        if isinstance(self.dataset, dict):
            self.train_dataset = self.dataset['train']
            self.val_dataset = self.dataset['valid']
            self.test_dataset = self.dataset['test']
        else:
            self.train_dataset, self.val_dataset, self.test_dataset = self.dataset
        logger.info(f'Size of training dataset: {len(self.train_dataset)}')
        logger.info(f'Size of validation dataset: {len(self.val_dataset)}')
        logger.info(f'Size of testing dataset: {len(self.test_dataset)}')
        return

    def train_dataloader(self):
        logger.debug('train_dataloader()')
        batch_size = self.hparams.batch_size
        num_workers = self.hparams.num_workers
        return DataLoader(self.train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    def val_dataloader(self):
        logger.debug('val_dataloader()')
        batch_size = self.hparams.batch_size
        num_workers = self.hparams.num_workers
        return DataLoader(self.val_dataset, batch_size=batch_size, num_workers=num_workers)

    def test_dataloader(self):
        logger.debug('test_dataloader()')
        batch_size = min(self.hparams.batch_size, 64)
        num_workers = self.hparams.num_workers
        return DataLoader(self.test_dataset, batch_size=batch_size, num_workers=num_workers)

    def training_step(self, batch, batch_idx):
        logger.debug('training_step({}, {})'.format(batch, batch_idx))
        logger.debug(f'training_step-{batch_idx}')
        loss = []
        for patch_n in self.hparams.train_patch_n:
            x, y = batch
            patch = make_patch2d(x, self.hparams.patch_size, patch_n)
            y_hat = self(patch)
            loss.append(F.cross_entropy(y_hat, y))
        total_loss = sum(loss) / len(loss)
        metrics = {f"train_loss_{n:05}": l for l, n in zip(loss, self.hparams.train_patch_n)}
        metrics["train_loss"] = total_loss
        return {'loss': total_loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        logger.debug('validation_step({}, {})'.format(batch, batch_idx))
        logger.debug(f'validation_step-{batch_idx}')
        loss = []
        correct = []
        for patch_n in self.hparams.valid_patch_n:
            x, y = batch
            patch = make_patch2d(x, self.hparams.patch_size, patch_n)
            y_hat = self(patch)
            loss.append(F.cross_entropy(y_hat, y, reduction='sum'))
            correct.append((y == y_hat.argmax(1)).float())
        return {'sum_loss': loss, 'correct': correct}

    def validation_epoch_end(self, outputs):
        logger.debug('validation_epoch_end({})'.format(outputs))
        loss = []
        accuracy = []
        for patch_n_i in range(len(self.hparams.valid_patch_n)):
            loss.append(torch.stack([x['sum_loss'][patch_n_i] for x in outputs]).sum() / len(self.val_dataset))
            accuracy.append(torch.cat([x['correct'][patch_n_i] for x in outputs]).mean())
        total_loss = sum(loss) / len(loss)
        total_accuracy = sum(accuracy) / len(accuracy)
        metrics = dict(
            **{'val_loss': total_loss, 'val_acc': total_accuracy},
            **{f"val_loss_{n:05}": l for l, n in zip(loss, self.hparams.test_patch_n)},
            **{f"val_acc_{n:05}": a for a, n in zip(accuracy, self.hparams.test_patch_n)}
        )
        return {'val_loss': total_loss, 'log': metrics}

    def test_step(self, batch, batch_idx):
        logger.debug('test_step({}, {})'.format(batch, batch_idx))
        loss = []
        correct = []
        for patch_n in self.hparams.test_patch_n:
            x, y = batch
            patch = make_patch2d(x, self.hparams.patch_size, patch_n)
            y_hat = self(patch)
            loss.append(F.cross_entropy(y_hat, y, reduction='sum'))
            correct.append((y == y_hat.argmax(1)).float())
        return {'sum_loss': loss, 'correct': correct}

    def test_epoch_end(self, outputs):
        logger.debug('test_epoch_end({})'.format(outputs))
        loss = []
        accuracy = []
        for patch_n_i in range(len(self.hparams.test_patch_n)):
            loss.append(torch.stack([x['sum_loss'][patch_n_i] for x in outputs]).sum() / len(self.test_dataset))
            accuracy.append(torch.cat([x['correct'][patch_n_i] for x in outputs]).mean())
        total_loss = sum(loss) / len(loss)
        total_accuracy = sum(accuracy) / len(accuracy)
        metrics = dict(
            **{'test_loss': total_loss.item(), 'test_acc': total_accuracy.item()},
            **{f"test_loss_{n:05}": l.item() for l, n in zip(loss, self.hparams.test_patch_n)},
            **{f"test_acc_{n:05}": a.item() for a, n in zip(accuracy, self.hparams.test_patch_n)}
        )
        self.log_metrics(metrics, self.global_step)
        logger.info(metrics)
        return {'test_loss': total_loss, 'log': metrics}

    def log_metrics(self, metrics, step=None):
        if self.logger is None:
            return
        self.logger.log_metrics(metrics, step)
        return

    def log_images(self, log_name, image, step=None):
        if self.logger is None:
            return
        if isinstance(self.logger, pl.loggers.base.LoggerCollection):
            for logger in self.logger:
                if hasattr(logger, 'log_image'):
                    logger.log_image(log_name, image, step)
        elif hasattr(self.logger, 'log_image'):
            logger.log_image(log_name, image, step)
        return
