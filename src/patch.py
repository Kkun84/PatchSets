from logging import getLogger
import torch


logger = getLogger(__name__)


def _padding(src, axis, padding, pad_value=0):
    logger.debug(f"src.shape={src.shape}, axis={axis}, padding={padding}, pad_value={pad_value}")
    if padding == 0:
        return src
    if hasattr(axis, '__iter__'):
        dst = src
        for a in axis:
            dst = _padding(src=dst, axis=a, padding=padding, pad_value=pad_value)
        return dst
    pad_shape = [padding if i == axis else s for i, s in enumerate(src.shape)]
    pad = torch.full(pad_shape, pad_value, dtype=src.dtype)
    logger.debug(f"pad.shape={pad.shape}")
    dst = torch.cat([pad, src, pad], axis)
    return dst


def make_patch1d(src, patch_size, patch_n, padding=0):
    # src axis: [batch, channels, features]
    # dst axis: [batch, sets, channels, features]
    logger.debug(f"src.shape={src.shape}, patch_size={patch_size}, patch_n={patch_n}, padding={padding}")
    dst = src
    dst = _padding(dst, 2, padding, pad_value=0)
    if dst.shape[2] - patch_size == 0:
        return dst[:, None].repeat(1, patch_n, 1, 1)
    patch_index = [
        torch.arange(dst.shape[0])[:, None, None, None],
        torch.arange(dst.shape[1])[None, None, :, None],
        torch.randint(dst.shape[2] - patch_size, [dst.shape[0], patch_n, 1, 1]) + torch.arange(patch_size)[None, None, None, :],
    ]
    logger.debug(f"patch_index.shape={[i.shape for i in patch_index]}")
    dst = dst[patch_index]
    logger.debug(f"dst.shape={dst.shape}")
    return dst


def make_patch2d(src, patch_size, patch_n, padding=0):
    # src axis: [batch, channels, height, width]
    # dst axis: [batch, sets, channels, height, width]
    logger.debug(f"src.shape={src.shape}, patch_size={patch_size}, patch_n={patch_n}, padding={padding}")
    dst = src
    dst = _padding(dst, [2, 3], padding, pad_value=0)
    if dst.shape[2] - patch_size == 0:
        return dst[:, None].repeat(1, patch_n, 1, 1, 1)
    patch_index = [
        torch.arange(dst.shape[0])[:, None, None, None, None],
        torch.arange(dst.shape[1])[None, None, :, None, None],
        torch.randint(dst.shape[2] - patch_size, [dst.shape[0], patch_n, 1, 1, 1]) + torch.arange(patch_size)[None, None, None, :, None],
        torch.randint(dst.shape[3] - patch_size, [dst.shape[0], patch_n, 1, 1, 1]) + torch.arange(patch_size)[None, None, None, None, :],
    ]
    logger.debug(f"patch_index.shape={[i.shape for i in patch_index]}")
    dst = dst[patch_index]
    logger.debug(f"dst.shape={dst.shape}")
    return dst
