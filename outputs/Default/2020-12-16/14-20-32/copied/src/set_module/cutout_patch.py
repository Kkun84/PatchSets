import torch
from torch import Tensor, nn
from typing import Iterable

from src.set_module.core import BatchSetType


def cutout_patch2d(
    batch: Tensor, patch_num: Iterable[int], patch_size: int, padding: int = 0
) -> BatchSetType:
    # input  axis: [batch, channels, height, width]
    # output axis: [batch, sets, channels, height, width]
    output = [
        image[
            [
                torch.arange(image.shape[0])[:, None, None, None],
                torch.arange(patch_size)[None, None, :, None]
                + torch.randint(image.shape[1] - patch_size, [pn, 1, 1, 1]),
                torch.arange(patch_size)[None, None, None, :]
                + torch.randint(image.shape[2] - patch_size, [pn, 1, 1, 1]),
            ]
        ]
        for image, pn in zip(batch, patch_num)
    ]
    # output = []
    # for image, pn in zip(batch, patch_num):
    #     shape = image.shape
    #     index = [
    #         torch.arange(shape[0])[:, None, None, None],
    #         torch.arange(patch_size)[None, None, :, None]
    #         + torch.randint(shape[1] - patch_size, [pn, 1, 1, 1]),
    #         torch.arange(patch_size)[None, None, None, :]
    #         + torch.randint(shape[2] - patch_size, [pn, 1, 1, 1]),
    #     ]
    #     output.append(image[index])
    return output


class CutoutPatch2d(nn.Module):
    def __init__(self, patch_size: int, padding: int = 0) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.padding = padding

    def forward(self, batch: Tensor, patch_num: Iterable[int]) -> BatchSetType:
        output = cutout_patch2d(batch, patch_num, self.padding)
        return output


if __name__ == '__main__':
    batch = torch.rand(4, 1, 10, 10)
    print('    batch', batch.shape, '', sep='\n')
    batch_set = cutout_patch2d(batch, range(1, len(batch) + 1), 3)
    print('batch_set', [i.shape for i in batch_set], '', sep='\n')
