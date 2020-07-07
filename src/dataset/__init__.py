from .MNIST import MNIST
from .my_dataset import MyDataset


def get_dataset(name):
    if name == 'MNIST':
        return MNIST
    elif name == 'MyDataset':
        return MyDataset
    assert False, f"Not found dataset {name}."
