import torch
from torch import Tensor, nn
from typing import List, Tuple

from src.set_module.core import BatchSetType, squeeze_from_set, unsqueeze_to_set


def set_pooling(
    batch: Tensor, index: List[int], mode: str, keep_shape: bool = False
) -> Tuple[Tensor, List[int]]:
    x = unsqueeze_to_set(batch, index)
    if mode == "avg" or mode == "mean":
        x = [i.mean(dim=0, keepdim=True) for i in x]
    elif mode == "sum":
        x = [i.sum(dim=0, keepdim=True) for i in x]
    elif mode == "max":
        x = [i.max(dim=0, keepdim=True)[0] for i in x]
    elif mode == "min":
        x = [i.min(dim=0, keepdim=True)[0] for i in x]
    elif mode == "var":
        x = [i.var(dim=0, keepdim=True) for i in x]
    else:
        assert False, ""
    if keep_shape:
        x = [a.expand(i, *[-1] * (a.dim() - 1)) for a, i in zip(x, index)]
        output, new_index = squeeze_from_set(x)
        assert index == new_index, f"index={index}, new_index={new_index}"
    else:
        output, new_index = squeeze_from_set(x)
    return output, new_index


class SetPooling(nn.Module):
    def __init__(self, mode: str, keep_shape: bool = False) -> None:
        super().__init__()
        self.mode = mode
        self.keep_shape = keep_shape

    def forward(self, batch: Tensor, index: List[int]) -> Tensor:
        output = set_pooling(batch, index, self.mode, self.keep_shape)
        return output


if __name__ == "__main__":
    batch_set = [torch.full([i, 2], i, dtype=float) for i in range(1, 4)]

    print("batch_set", *batch_set, "", sep="\n")

    batch, index = squeeze_from_set(batch_set)

    mode = "sum"
    for keepdim in [False, True]:
        print(
            f"mode={mode}",
            f"keepdim={keepdim}",
            set_pooling(batch, index, mode, keepdim),
            "",
            sep="\n",
        )


# def set_pooling(batch_set: BatchSetType, mode: str) -> List[Tensor]:
#     x = batch_set
#     if mode == 'avg' or mode == 'mean':
#         x = [i.mean(dim=0, keepdim=True) for i in x]
#     elif mode == 'sum':
#         x = [i.sum(dim=0, keepdim=True) for i in x]
#     elif mode == 'max':
#         x = [i.max(dim=0, keepdim=True)[0] for i in x]
#     elif mode == 'min':
#         x = [i.min(dim=0, keepdim=True)[0] for i in x]
#     elif mode == 'var':
#         x = [i.var(dim=0, keepdim=True) for i in x]
#     else:
#         assert False, ''
#     output = x
#     return output


# class SetPooling(nn.Module):
#     def __init__(self, mode: str) -> None:
#         super().__init__()
#         self.mode = mode

#     def forward(self, batch_set: BatchSetType) -> List[Tensor]:
#         output = set_pooling(batch_set, self.mode)
#         return output


# if __name__ == '__main__':
#     batch_set = [torch.rand([i, 3]) for i in range(1, 4)]

#     print('batch_set', *batch_set, '', sep='\n')

#     for mode in ['avg', 'mean', 'sum', 'max', 'min', 'var']:
#         print(mode, set_pooling(batch_set, mode), '', sep='\n')
