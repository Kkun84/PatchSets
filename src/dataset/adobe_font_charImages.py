from logging import getLogger
import itertools
from tqdm import tqdm
import pathlib
import torch
from PIL import Image


logger = getLogger(__name__)


class AdobeFontCharImages(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        base_path = pathlib.Path(root) / 'AdobeFontCharImages'
        self.unique_font = sorted([i.name for i in (base_path / 'font').iterdir()])
        self.unique_alphabet = sorted([i.name for i in (base_path / 'alphabet').iterdir()])

        self.path, self.font, self.alphabet = zip(*[[
                (base_path / 'font' / font / f"{alphabet}_{font}.png"), i, j
            ] for (i, font), (j, alphabet) in itertools.product(
                enumerate(self.unique_font), enumerate(self.unique_alphabet)
            )
        ])
        self.image = [Image.open(p) for p in tqdm(self.path)]

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        x = self.image[index]
        if self.transform is not None:
            x = self.transform(x)
        y = dict(font=self.font[index], alphabet=self.alphabet[index])
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

class ExtractFont:
    def __call__(self, y):
        return y['font']

class ExtractAlphabet:
    def __call__(self, y):
        print(y)
        return y['alphabet']
