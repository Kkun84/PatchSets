import enum
from logging import getLogger
import itertools
from tqdm import tqdm
from pathlib import Path
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

from .utils import split_dataset


logger = getLogger(__name__)


def adobe_font_char_images(root, transform=None, target_transform=None):
    dataset = AdobeFontCharImages(root, transform=None, target_transform=None)
    unique_font = range(len(dataset.unique_font))
    train_font, test_font = train_test_split(unique_font, test_size=1/6, random_state=0)
    train_index = []
    test_index = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        if y['font'] in train_font:
            train_index.append(i)
        elif y['font'] in test_font:
            test_index.append(i)
        else:
            assert False
    train = AdobeFontCharImages(root, train_index, transform, target_transform)
    test = AdobeFontCharImages(root, test_index, transform, target_transform)
    return train, test


class AdobeFontCharImages(torch.utils.data.Dataset):
    def __init__(self, root, subset_index=None, transform=None, target_transform=None):
        logger.debug(f'AdobeFontCharImages({root}, {transform}, {target_transform})')
        self.transform = transform
        self.target_transform = target_transform

        root = Path(root) / 'AdobeFontCharImages'
        self.unique_font = sorted([i.name for i in (root / 'font').iterdir()])
        self.unique_alphabet = sorted([i.name for i in (root / 'alphabet').iterdir()])

        file_format = '{root}/font/{font}/{alphabet}_{font}.png'.format
        self.data = []
        self.path = []
        it = list(itertools.product(enumerate(self.unique_font), enumerate(self.unique_alphabet)))
        if subset_index is None:
            subset_index = range(len(it))
        for i in tqdm(subset_index):
            (fi, font), (ai, alphabet) = it[i]
            path = file_format(root=root, font=font, alphabet=alphabet)
            image = Image.open(path)
            self.path.append(path)
            self.data.append((image, dict(font=fi, alphabet=ai)))

    def set_transform(self, transform=None, target_transform=None):
        if transform is not None:
            self.transform = transform
        if target_transform is not None:
            self.target_transform = target_transform

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        x, y = self.data[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

class ExtractFont:
    def __call__(self, y):
        return y['font']

class ExtractAlphabet:
    def __call__(self, y):
        return y['alphabet']
