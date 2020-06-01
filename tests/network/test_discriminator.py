import torch

import pytest

from src.network import Discriminator


def test_Discriminator(batch, model_params):
    assert issubclass(Discriminator, torch.nn.Module)

    discriminator = Discriminator(model_params)
    assert hasattr(discriminator, 'input_shape')
    assert hasattr(discriminator, 'output_shape')

    assert list(discriminator(batch[0]).shape[1:]) == list(discriminator.output_shape)
