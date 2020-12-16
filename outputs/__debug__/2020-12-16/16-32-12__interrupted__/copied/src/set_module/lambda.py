import torch
from torch import Tensor, nn
from typing import List, Callable, Union, Iterable

from .core import BatchSetType


def set_lambda(
    batch_set: BatchSetType,
    func: Callable[[Union[Tensor, Iterable[Tensor]]], Tensor],
) -> List[Tensor]:
    x = batch_set
    output = [func(i) for i in x]
    return output


class SetLambda(nn.Module):
    def __init__(
        self, func: Callable[[Union[Tensor, Iterable[Tensor]]], Tensor]
    ) -> None:
        super().__init__()
        self.func = func

    def forward(
        self,
        batch_set: BatchSetType,
    ) -> List[Tensor]:
        output = set_lambda(batch_set, self.func)
        return output


if __name__ == '__main__':
    batch_set_0 = [torch.rand([i, 3]) for i in range(1, 3)]
    batch_set_1 = [torch.rand([i, 3]) for i in range(1, 3)]

    print('batch_set_0', *batch_set_0, '', sep='\n')
    print('batch_set_1', *batch_set_1, '', sep='\n')

    for func in [
        lambda x: x[0] + x[1],
        lambda x: x[0] - x[1],
        lambda x: x[0] * x[1],
        lambda x: x[0] / x[1],
    ]:
        print(set_lambda(zip(batch_set_0, batch_set_1), func), '', sep='\n')
