import torch
from torch import Tensor, nn
from typing import Union, List, Tuple


BatchSetType = Union[List[Tensor], Tuple[Tensor]]


def squeeze_from_set(batch_set: BatchSetType) -> Tuple[Tensor, List[int]]:
    batch = torch.cat(batch_set, 0)
    index = [x.shape[0] for x in batch_set]
    return batch, index


def unsqueeze_to_set(batch: Tensor, index: List[int]) -> Tensor:
    batch_set = batch.split(index)
    return batch_set


class SqueezeFromSet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch_set: BatchSetType) -> Tuple[Tensor, List[int]]:
        output = squeeze_from_set(batch_set)
        return output


class UnsqueezeToSet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Tensor, index: List[int]) -> Tensor:
        output = unsqueeze_to_set(batch, index)
        return output


if __name__ == "__main__":
    batch_set = [torch.full([i, 2], i) for i in range(1, 5)]

    print('batch_set', batch_set, '', sep='\n')

    batch, index = squeeze_from_set(batch_set)

    print('batch', batch, '', sep='\n')
    print('index', index, '', sep='\n')

    splitted = unsqueeze_to_set(batch, index)

    print('splitted', splitted, '', sep='\n')
    print([i == j for i, j in zip(batch_set, splitted)])
