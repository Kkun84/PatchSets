from logging import getLogger
import hydra
import pytorch_lightning as pl

from model import Model


logger = getLogger(__name__)


@hydra.main(config_path='../conf/config.yaml')
def main(cfg):
    logger.info(f"\n{cfg.pretty()}")

    model = Model(cfg.model_params, cfg.hparams, cfg.optim, hydra.utils.to_absolute_path('./data'))

    trainer_params = {}

    if cfg.loggers is not None:
        logger.info('Setting logger.')
        trainer_params['logger'] = [hydra.utils.instantiate(i) for i in cfg.loggers]

    if cfg.callback.checkpoint is not None:
        logger.info('Setting checkpoint_callback.')
        trainer_params['checkpoint_callback'] = hydra.utils.instantiate(cfg.callback.checkpoint)

    if cfg.callback.early_stopping is not None:
        logger.info('Setting early_stop_callback.')
        trainer_params['early_stop_callback'] = hydra.utils.instantiate(cfg.callback.early_stopping)

    if cfg.callback.callbacks is not None:
        logger.info('Setting callbacks.')
        trainer_params['callbacks'] = [hydra.utils.instantiate(i) for i in cfg.callback.callbacks]

    trainer = pl.Trainer(
        **cfg.trainer,
        **trainer_params,
    )

    logger.info('Start trainer.fit().')
    trainer.fit(model)

    logger.info('Start trainer.test().')
    trainer.test()

    logger.info('All done.')


if __name__ == "__main__":
    main()
