from logging import getLogger
from torchvision.datasets import MNIST as original_MNIST


logger = getLogger(__name__)


def MNIST(root, transform=None, target_transform=None):
    download = False or True
    train = original_MNIST(root=root, train=True, transform=transform, target_transform=target_transform, download=download)
    test = original_MNIST(root=root, train=False, transform=transform, target_transform=target_transform, download=download)
    return train, test
