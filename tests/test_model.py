from logging import getLogger
import torch

import pytest

from src.model import Encoder, Decoder, Model
from src.patch import make_patch2d


logger = getLogger(__name__)


def test_Encoder(batch_size, patch_n, model_params):
    assert issubclass(Encoder, torch.nn.Module)

    encoder = Encoder(model_params.encoder)
    assert hasattr(encoder, 'input_n')
    assert hasattr(encoder, 'output_n')

    x = torch.rand([batch_size, patch_n, encoder.input_n])
    z = encoder(x)
    assert z.shape[1] == encoder.output_n


def test_Decoder(batch_size, patch_n, model_params):
    assert issubclass(Decoder, torch.nn.Module)

    decoder = Decoder(model_params.decoder)
    assert hasattr(decoder, 'input_n')
    assert hasattr(decoder, 'output_n')

    x = torch.rand([batch_size, decoder.input_n])
    y = decoder(x)
    assert y.shape[1] == decoder.output_n


def test_Model(model_params, hparams, optim, data_path, batch):
    model = Model(model_params, hparams, optim, data_path)
    assert hasattr(model, 'input_n')
    assert hasattr(model, 'output_n')

    optimizer, scheduler = model.configure_optimizers()
    assert all([isinstance(i, torch.optim.Optimizer) for i in optimizer])
    if not scheduler:
        assert all([isinstance(i, torch.optim.lr_scheduler._LRScheduler) for i in scheduler])
    else:
        assert lr_scheduler is None

    x = make_patch2d(batch[0], hparams.patch_size, hparams.train_patch_n)
    logger.debug(f"x.shape={x.shape}")
    y = model(x)
    logger.debug(f"y.shape={y.shape}")
    assert y.shape[1] == model.output_n

    model.prepare_data()
    model.training_epoch_end([model.training_step(batch, None)])
    model.validation_epoch_end([model.validation_step(batch, None)])
    model.test_epoch_end([model.test_step(batch, None)])
