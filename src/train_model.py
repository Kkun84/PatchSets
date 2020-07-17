from logging import getLogger
import hydra
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import os.path

from src.model import Encoder, Decoder
from src.integrated_model import IntegratedModel
from src.dataset import split_dataset


logger = getLogger(__name__)


@hydra.main(config_path='../config/config.yaml')
def main(config):
    logger.info(f"\n{config.pretty()}")

    pl.seed_everything(config.hparams.seed)

    encoder = Encoder(**config.model_params.encoder)
    decoder = Decoder(**config.model_params.decoder)

    transform = torchvision.transforms.Compose([hydra.utils.instantiate(i) for i in config.transform]) if config.transform else None
    target_transform = torchvision.transforms.Compose([hydra.utils.instantiate(i) for i in config.target_transform]) if config.target_transform else None

    tmp_dataset, test_dataset = hydra.utils.instantiate(config.dataset)
    tmp_dataset.set_transform(transform, target_transform)
    test_dataset.set_transform(transform, target_transform)

    dataset = *split_dataset(tmp_dataset, config.hparams.dataset_n_splits, config.hparams.dataset_n), test_dataset

    model = IntegratedModel(config.hparams, encoder, decoder, config.optim, dataset)

    trainer_params = {}

    if config.loggers is not None:
        logger.info('Setting logger.')
        loggers = [hydra.utils.instantiate(i) for i in config.loggers]
        logger.info(f"loggers={loggers}")
        trainer_params['logger'] = loggers

    if config.callback.checkpoint is not None:
        logger.info('Setting checkpoint_callback.')
        trainer_params['checkpoint_callback'] = hydra.utils.instantiate(config.callback.checkpoint)

    if config.callback.early_stopping is not None:
        logger.info('Setting early_stop_callback.')
        trainer_params['early_stop_callback'] = hydra.utils.instantiate(config.callback.early_stopping)

    if config.callback.callbacks is not None:
        logger.info('Setting callbacks.')
        callbacks = [hydra.utils.instantiate(i) for i in config.callback.callbacks]
        logger.info(f"callbacks={callbacks}")
        trainer_params['callbacks'] = callbacks

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
