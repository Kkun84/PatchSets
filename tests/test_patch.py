from logging import getLogger
import itertools
import numpy as np
import torch

import pytest

from src.patch import _padding
from src.patch import make_patch1d, make_patch2d


logger = getLogger(__name__)


@pytest.mark.parametrize(['axis', 'padding', 'pad_value', 'input', 'output'], [
    [
        0, 2, 0,
        [1],
        [0, 0, 1, 0, 0],
    ],
    [
        [0], 2, 0,
        [1],
        [0, 0, 1, 0, 0],
    ],
    [
        1, 1, 1,
        [[2, 3],
         [4, 5]],
        [[1, 2, 3, 1],
         [1, 4, 5, 1]]
    ],
    [
        [0, 1], 1, 5,
        [[2, 3],
         [4, 5]],
        [[5, 5, 5, 5],
         [5, 2, 3, 5],
         [5, 4, 5, 5],
         [5, 5, 5, 5]]
    ],
    [
        [1, 2], 1, 0,
        [[[1]]],
        [[[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]]
    ],
    ])
def test__padding(axis, padding, pad_value, input, output):
    x = torch.Tensor(input)
    y = torch.Tensor(output)
    y_ = _padding(src=x, axis=axis, padding=padding, pad_value=pad_value)
    assert (y == y_).all()
    return


@pytest.mark.parametrize('padding', [0, 1, 2, 3])
def test_padding_make_patch1d(channels, batch_size, padding, patch_size, patch_n):
    shape = [batch_size, channels, *[patch_size]*1]
    x = torch.arange(np.prod(shape)).reshape(shape)
    padded_x = _padding(src=x, axis=[2], padding=padding, pad_value=0)
    for n in patch_n:
        y = make_patch1d(x, patch_size, n, padding=padding)
        logger.debug(f"x.shape={x.shape}")
        logger.debug(f"y.shape={y.shape}")
        assert list(y.shape) == [batch_size, n, channels, *[patch_size] * 1]

        if padding == 0:
            assert (x[:, None].repeat(1, n, 1, 1) == y).all()
        else:
            for _x, _y in zip(padded_x, y):
                for __y in _y:
                    for index in itertools.product(*[range(i - patch_size) for i in _x.shape[1:]]):
                        if (_x[:, index[0]:index[0] + patch_size] == __y).all():
                            break
                    else:
                        assert False, 'The pattern did not match.'
                    break
    return


@pytest.mark.parametrize('padding', [0, 1, 2, 3])
def test_padding_make_patch2d(channels, batch_size, padding, patch_size, patch_n):
    shape = [batch_size, channels, *[patch_size]*2]
    x = torch.arange(np.prod(shape)).reshape(shape)
    padded_x = _padding(src=x, axis=[2, 3], padding=padding, pad_value=0)
    for n in patch_n:
        y = make_patch2d(x, patch_size, n)
        logger.debug(f"x.shape={x.shape}")
        logger.debug(f"y.shape={y.shape}")
        assert list(y.shape) == [batch_size, n, channels, *[patch_size] * 2]

        if padding == 0:
            assert (x[:, None].repeat(1, n, 1, 1, 1) == y).all()
        else:
            for _x, _y in zip(padded_x, y):
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
    return


@pytest.mark.parametrize('margin', [0, 1, 2, 3])
def test_margin_make_patch1d(channels, batch_size, margin, patch_size, patch_n):
    shape = [batch_size, channels, *[margin + patch_size]*1]
    x = torch.arange(np.prod(shape)).reshape(shape)
    for n in patch_n:
        y = make_patch1d(x, patch_size, n)
        logger.debug(f"x.shape={x.shape}")
        logger.debug(f"y.shape={y.shape}")
        assert list(y.shape) == [batch_size, n, channels, *[patch_size] * 1]

        if margin == 0:
            assert (x[:, None].repeat(1, n, 1, 1) == y).all()
        else:
            for _x, _y in zip(x, y):
                for __y in _y:
                    for index in itertools.product(*[range(i - patch_size) for i in _x.shape[1:]]):
                        if (_x[:, index[0]:index[0] + patch_size] == __y).all():
                            break
                    else:
                        assert False, 'The pattern did not match.'
                    break
    return


@pytest.mark.parametrize('margin', [0, 1, 2, 3])
def test_margin_make_patch2d(channels, batch_size, margin, patch_size, patch_n):
    shape = [batch_size, channels, *[margin + patch_size]*2]
    x = torch.arange(np.prod(shape)).reshape(shape)
    for n in patch_n:
        y = make_patch2d(x, patch_size, n)
        logger.debug(f"x.shape={x.shape}")
        logger.debug(f"y.shape={y.shape}")
        assert list(y.shape) == [batch_size, n, channels, *[patch_size] * 2]

        if margin == 0:
            assert (x[:, None].repeat(1, n, 1, 1, 1) == y).all()
        else:
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
    return
