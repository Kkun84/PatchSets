from logging import getLogger
import torch
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold


logger = getLogger(__name__)


def split_dataset(dataset, n_splits, n):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    train_index, valid_index = list(skf.split(*zip(*dataset)))[n]
    train = torch.utils.data.Subset(dataset, train_index)
    valid = torch.utils.data.Subset(dataset, valid_index)
    logger.debug(f"len(train)={len(train)}")
    logger.debug(f"len(valid)={len(valid)}")
    return train, valid
