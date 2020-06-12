from logging import getLogger
import torch


logger = getLogger(__name__)


def make_patch1d(input, patch_size, patch_n, padding):
    # intput axis: [batch, channels, features]
    # output axis: [batch, sets, channels, features]
    logger.debug(f"{input.shape=}, {patch_size=}, {patch_n=}, {padding=}")
    if input.shape[2] - patch_size + padding == 0:
        return input[:, None].repeat(1, patch_n, 1, 1)
    patch_index = [
        torch.arange(input.shape[0])[:, None, None, None],
        torch.arange(input.shape[1])[None, None, :, None],
        torch.randint(input.shape[2] - patch_size, [input.shape[0], patch_n, 1, 1]) + torch.arange(patch_size)[None, None, None, :],
    ]
    logger.debug(f"patch_index.shape={[i.shape for i in patch_index]}")
    output = input[patch_index]
    logger.debug(f"{output.shape=}")
    return output


def make_patch2d(input, patch_size, patch_n, padding):
    # intput axis: [batch, channels, height, width]
    # output axis: [batch, sets, channels, height, width]
    logger.debug(f"{input.shape=}, {patch_size=}, {patch_n=}, {padding=}")
    if input.shape[2] - patch_size + padding == 0:
        return input[:, None].repeat(1, patch_n, 1, 1, 1)
    patch_index = [
        torch.arange(input.shape[0])[:, None, None, None, None],
        torch.arange(input.shape[1])[None, None, :, None, None],
        torch.randint(input.shape[2] - patch_size, [input.shape[0], patch_n, 1, 1, 1]) + torch.arange(patch_size)[None, None, None, :, None],
        torch.randint(input.shape[3] - patch_size, [input.shape[0], patch_n, 1, 1, 1]) + torch.arange(patch_size)[None, None, None, None, :],
    ]
    logger.debug(f"patch_index.shape={[i.shape for i in patch_index]}")
    output = input[patch_index]
    logger.debug(f"{output.shape=}")
    return output
