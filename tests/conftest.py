import hydra
import argparse
import torch

import pytest


@pytest.fixture(params=[1, 2])
def batch_size(request):
    return request.param


@pytest.fixture(params=[1, 2])
def channels(request):
    return request.param


@pytest.fixture(params=[0, 2])
def num_workers(request):
    return request.param


@pytest.fixture(params=[0.001])
def lr(request):
    return request.param


@pytest.fixture(params=[[1], [2, 3]])
def patch_n(request):
    return request.param


@pytest.fixture(params=[[1], [2, 3]])
def train_patch_n(request):
    return request.param


@pytest.fixture(params=[[1], [2, 3]])
def test_patch_n(request):
    return request.param


@pytest.fixture(params=[1, 2])
def patch_size(request):
    return request.param


@pytest.fixture(params=[1, 2])
def latent_dim(request):
    return request.param


@pytest.fixture(params=[0.5, 0, -1])
def n_pow(request):
    return request.param


@pytest.fixture(params=[
    # {
    #     'class': 'torch.optim.SGD',
    #     'params': {
    #         'lr': None,
    # }},
    {
        'class': 'torch.optim.Adam',
        'params': {
            'lr': None,
    }},
])
def optimizer(request, lr):
    x = request.param
    x['params']['lr'] = lr
    return x


@pytest.fixture(params=[
    {
        'class': 'torch.optim.lr_scheduler.StepLR',
        'params': {
            'step_size': 10,
            'gamma': 0.5,
    }},
    # {
    #     'class': 'torch.optim.lr_scheduler.MultiStepLR',
    #     'params': {
    #         'milestones': [5, 10, 20],
    #         'gamma': 0.5,
    # }},
    # {
    #     'class': 'torch.optim.lr_scheduler.ExponentialLR',
    #     'params': {
    #         'gamma': 0.9,
    # }},
    None
])
def scheduler(request):
    return request.param


@pytest.fixture
def batch(batch_size, channels, patch_size):
    x = torch.rand([batch_size, channels, *[patch_size + 8]*2])
    y = torch.randint(10, [batch_size])
    return (x, y)


@pytest.fixture
def model_params(channels, patch_size, n_pow):
    return hydra.utils.DictConfig(dict(
        # encoder=dict(input_n=channels * patch_size**2, hidden_n_0=11, output_n=13),
        encoder=dict(input_shape=[channels, patch_size, patch_size], hidden_n_0=11, hidden_n_1=13, output_n=17, n_pow=n_pow),
        decoder=dict(input_n=17, hidden_n_0=23, hidden_n_1=29, output_n=31),
    ))


@pytest.fixture
def hparams(batch_size, num_workers, lr, train_patch_n, test_patch_n, patch_size, latent_dim):
    return hydra.utils.DictConfig(dict(
        batch_size=batch_size,
        num_workers=num_workers,
        lr=lr,
        train_patch_n=train_patch_n,
        test_patch_n=test_patch_n,
        patch_size=patch_size,
        latent_dim=latent_dim,
    ))


@pytest.fixture
def optim(optimizer, scheduler):
    return hydra.utils.DictConfig({
        'optimizer': optimizer,
        'scheduler': scheduler,
    })


@pytest.fixture(scope='session')
def data_path(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp('tmp')
    # tmpdir = 'data'
    return tmpdir
