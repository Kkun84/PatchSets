from logging import getLogger
import hydra
import random
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold

from dataset import get_dataset
from model import Encoder, Decoder
from integrated_model import IntegratedModel


logger = getLogger(__name__)


def split_datset(dataset, n_splits, n):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    train_index, valid_index = list(kf.split(*zip(*dataset)))[n]
    train = torch.utils.data.Subset(dataset, train_index)
    valid = torch.utils.data.Subset(dataset, valid_index)
    return train, valid


@hydra.main(config_path='../conf/config.yaml')
def main(config):
    logger.info(f"\n{config.pretty()}")

    pl.seed_everything(config.hparams.seed)

    encoder = Encoder(**config.model_params.encoder)
    decoder = Decoder(**config.model_params.decoder)

    transform = torchvision.transforms.Compose([hydra.utils.instantiate(i) for i in config.dataset.transform]) if config.dataset.transform else None
    target_transform = torchvision.transforms.Compose([hydra.utils.instantiate(i) for i in config.dataset.target_transform]) if config.dataset.target_transform else None
    tmp_dataset, test_dataset = get_dataset(config.dataset.name)(**config.dataset.params, transform=transform, target_transform=target_transform)
    dataset = *split_datset(tmp_dataset, config.hparams.dataset_n_splits, config.hparams.dataset_n), test_dataset

    model = IntegratedModel(config.hparams, encoder, decoder, config.optim, dataset)

    trainer_params = {}

    if config.loggers is not None:
        logger.info('Setting logger.')
        trainer_params['logger'] = [hydra.utils.instantiate(i) for i in config.loggers]

    if config.callback.checkpoint is not None:
        logger.info('Setting checkpoint_callback.')
        trainer_params['checkpoint_callback'] = hydra.utils.instantiate(config.callback.checkpoint)

    if config.callback.early_stopping is not None:
        logger.info('Setting early_stop_callback.')
        trainer_params['early_stop_callback'] = hydra.utils.instantiate(config.callback.early_stopping)

    if config.callback.callbacks is not None:
        logger.info('Setting callbacks.')
        trainer_params['callbacks'] = [hydra.utils.instantiate(i) for i in config.callback.callbacks]

    trainer = pl.Trainer(
        **config.trainer,
        **trainer_params,
    )

    logger.info('Start trainer.fit().')
    trainer.fit(model)

    logger.info('Start trainer.test().')
    trainer.test()

    logger.info('All done.')


if __name__ == "__main__":
    main()
