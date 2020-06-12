from logging import getLogger
import torch


logger = getLogger(__name__)


def padding(src, axis, padding, pad_value=0):
    if hasattr(axis, '__iter__'):
        dst = src
        for a in axis:
            dst = padding(dst, a, padding, pad_value=0)
        return dst
    logger.debug(f"{src.shape=}, {axis=}, {padding=}, {pad_value=}")
    pad_shape = [1] * src.ndim
    pad_shape[axis] = padding
    pad = torch.full(pad_shape, pad_value)
    dst = torch.cat([pad, src, pad], axis)
    return dst


def make_patch1d(src, patch_size, patch_n, padding):
    # src axis: [batch, channels, features]
    # dst axis: [batch, sets, channels, features]
    logger.debug(f"{src.shape=}, {patch_size=}, {patch_n=}, {padding=}")
    if src.shape[2] - patch_size + padding == 0:
        return src[:, None].repeat(1, patch_n, 1, 1)
    patch_index = [
        torch.arange(src.shape[0])[:, None, None, None],
        torch.arange(src.shape[1])[None, None, :, None],
        torch.randint(src.shape[2] - patch_size, [src.shape[0], patch_n, 1, 1]) + torch.arange(patch_size)[None, None, None, :],
    ]
    logger.debug(f"patch_index.shape={[i.shape for i in patch_index]}")
    dst = src[patch_index]
    logger.debug(f"{dst.shape=}")
    return dst


def make_patch2d(src, patch_size, patch_n, padding):
    # src axis: [batch, channels, height, width]
    # dst axis: [batch, sets, channels, height, width]
    logger.debug(f"{src.shape=}, {patch_size=}, {patch_n=}, {padding=}")
    if src.shape[2] - patch_size + padding == 0:
        return src[:, None].repeat(1, patch_n, 1, 1, 1)
    patch_index = [
        torch.arange(src.shape[0])[:, None, None, None, None],
        torch.arange(src.shape[1])[None, None, :, None, None],
        torch.randint(src.shape[2] - patch_size, [src.shape[0], patch_n, 1, 1, 1]) + torch.arange(patch_size)[None, None, None, :, None],
        torch.randint(src.shape[3] - patch_size, [src.shape[0], patch_n, 1, 1, 1]) + torch.arange(patch_size)[None, None, None, None, :],
    ]
    logger.debug(f"patch_index.shape={[i.shape for i in patch_index]}")
    dst = src[patch_index]
    logger.debug(f"{dst.shape=}")
    return dst
