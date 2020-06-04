from logging import getLogger
import itertools
import numpy as np
import torch

import pytest

from src.patch import make_patch1d, make_patch2d, make_patch3d


logger = getLogger(__name__)


@pytest.mark.parametrize('margin', [0, 1, 2, 3])
def test_train_make_patch1d(channels, batch_size, margin, patch_size, patch_n):
    shape = [batch_size, channels, *[margin + patch_size]*1]
    x = torch.arange(np.prod(shape)).reshape(shape)
    for n in patch_n:
        y = make_patch1d(x, patch_size, n)
        logger.debug(f"x.shape={x.shape}")
        logger.debug(f"y.shape={y.shape}")
        assert list(y.shape) == [batch_size, n, channels, *[patch_size] * 1]

        for _x, _y in zip(x, y):
            for __y in _y:
                for index in itertools.product(*[range(i - patch_size) for i in _x.shape[1:]]):
                    if (_x[:, index[0]:index[0] + patch_size] == __y).all():
                        break
                else:
                    assert False, 'The pattern did not match.'
                break


@pytest.mark.parametrize('margin', [0, 1, 2, 3])
def test_train_make_patch2d(channels, batch_size, margin, patch_size, patch_n):
    shape = [batch_size, channels, *[margin + patch_size]*2]
    x = torch.arange(np.prod(shape)).reshape(shape)
    for n in patch_n:
        y = make_patch2d(x, patch_size, n)
        logger.debug(f"x.shape={x.shape}")
        logger.debug(f"y.shape={y.shape}")
        assert list(y.shape) == [batch_size, n, channels, *[patch_size] * 2]

        for _x, _y in zip(x, y):
            for __y in _y:
                for index in itertools.product(*[range(i - patch_size) for i in _x.shape[1:]]):
                    logger.debug(f"index={index}")
                    if (_x[:,
                            index[0]: index[0] + patch_size,
                            index[1]: index[1] + patch_size,
                            ] == __y).all():
                        break
                else:
                    assert False, 'The pattern did not match.'
                break


@pytest.mark.parametrize('margin', [0, 1, 2, 3])
def test_train_make_patch3d(channels, batch_size, margin, patch_size, patch_n):
    shape = [batch_size, channels, *[margin + patch_size]*3]
    x = torch.arange(np.prod(shape)).reshape(shape)
    for n in patch_n:
        y = make_patch3d(x, patch_size, n)
        logger.debug(f"x.shape={x.shape}")
        logger.debug(f"y.shape={y.shape}")
        assert list(y.shape) == [batch_size, n, channels, *[patch_size] * 3]

        for _x, _y in zip(x, y):
            for __y in _y:
                for index in itertools.product(*[range(i - patch_size) for i in _x.shape[1:]]):
                    if (_x[:,
                            index[0]: index[0] + patch_size,
                            index[1]: index[1] + patch_size,
                            index[2]: index[2] + patch_size,
                            ] == __y).all():
                        break
                else:
                    assert False, 'The pattern did not match.'
                break
