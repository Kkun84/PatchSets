import enum
from logging import getLogger
import itertools
from tqdm import tqdm
from pathlib import Path
import torch
from PIL import Image
import pandas as pd
from typing import Iterable

logger = getLogger(__name__)


class AdobeFontDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        transform=None,
        target_transform=None,
        upper: bool = True,
        lower: bool = True,
        use_font_index: Iterable = None,
    ):
        self.transform = transform
        self.target_transform = target_transform

        root = Path(path)

        self.unique_font = sorted([i.name for i in (root / 'font').iterdir()])
        if use_font_index is not None:
            self.unique_font = [
                x for i, x in enumerate(self.unique_font) if i in use_font_index
            ]
        self.unique_alphabet = sorted([i.name for i in (root / 'alphabet').iterdir()])
        if upper == False:
            self.unique_alphabet = [i for i in self.unique_alphabet if i[:3] != 'cap']
        if lower == False:
            self.unique_alphabet = [i for i in self.unique_alphabet if i[:5] != 'small']

        df = pd.read_csv(root / 'list.csv').sort_values('FONT')
        self.unique_category = sorted(df['CATEGORY'].unique())
        self.unique_sub_category = sorted(df['SUB-CATEGORY'].unique())

        self.data = []
        self.path = []
        for (fi, font), (ai, alphabet) in tqdm(
            itertools.product(
                enumerate(self.unique_font), enumerate(self.unique_alphabet)
            ),
            total=len(self.unique_font) * len(self.unique_alphabet),
        ):
            path = root / 'font' / font / (alphabet + '_' + font + '.png')
            image = Image.open(path)
            self.path.append(path)

            # tmp = df[df['FONT'] == font]
            # category = tmp['CATEGORY'].iat[0]
            # sub_category = tmp['SUB-CATEGORY'].iat[0]
            self.data.append(
                (
                    image,
                    {
                        'font': fi,
                        'alphabet': ai,
                        # 'category': self.unique_category.index(category),
                        # 'sub_category': self.unique_sub_category.index(sub_category),
                    },
                )
            )

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        x, y = self.data[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


if __name__ == "__main__":
    path = '/dataset/AdobeFontCharImages'
    dataset = AdobeFontDataset(
        path=path,
        transform=None,
        target_transform=None,
        upper=True,
        lower=True,
    )
    print(len(dataset), dataset.unique_alphabet)
    dataset = AdobeFontDataset(
        path=path,
        transform=None,
        target_transform=None,
        upper=True,
        lower=False,
    )
    print(len(dataset), dataset.unique_alphabet)
    dataset = AdobeFontDataset(
        path=path,
        transform=None,
        target_transform=None,
        upper=False,
        lower=True,
    )
    print(len(dataset), dataset.unique_alphabet)
    dataset = AdobeFontDataset(
        path=path,
        transform=None,
        target_transform=None,
        upper=False,
        lower=False,
    )
    print(len(dataset), dataset.unique_alphabet)
