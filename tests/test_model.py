import torch

import pytest

from src.model import Model


def test_Model(model_params, hparams, optim, data_path, batch):
    model = Model(model_params, hparams, optim, data_path)
    assert hasattr(model, 'input_shape')
    assert hasattr(model, 'output_shape')

    optimizer, scheduler = model.configure_optimizers()
    assert all([isinstance(i, torch.optim.Optimizer) for i in optimizer])
    if not scheduler:
        assert all([isinstance(i, torch.optim.lr_scheduler._LRScheduler) for i in scheduler])

    assert list(model(batch[0]).shape[1:]) == list(model.output_shape)

    model.prepare_data()
    model.training_epoch_end([model.training_step(batch, None)])
    model.validation_epoch_end([model.validation_step(batch, None)])
    model.test_epoch_end([model.test_step(batch, None)])
