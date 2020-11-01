import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from logging import getLogger
from pathlib import Path

from src.model import Model


logger = getLogger(__name__)


@hydra.main(config_path='../config', config_name='config.yaml')
def main(config) -> None:
    all_done = False
    try:
        logger.info('\n' + OmegaConf.to_yaml(config))

        pl.seed_everything(config.hparams.seed)

        datamodule = hydra.utils.instantiate(config.data.datamodule)

        trainer = pl.Trainer(
            **config.trainer,
            checkpoint_callback=ModelCheckpoint(**config.model_checkpoint),
            callbacks=[EarlyStopping(**config.early_stopping)]
            + [hydra.utils.instantiate(i) for i in config.callbacks],
            logger=[hydra.utils.instantiate(i) for i in config.loggers],
            auto_lr_find=config.hparams.lr == 0,
        )

        model = Model(**config.hparams)

        trainer.tune(model, datamodule=datamodule)
        if config.debug == True:
            # fast_dev_runモードではauto_lr_findが失敗し，model.hparams.lrにNoneが代入される
            assert model.hparams.lr is None
            model.hparams.lr = 1
        assert model.hparams.lr > 0, f'model.hparams.lr > 0={model.hparams.lr > 0}'
        config.hparams.lr = model.hparams.lr

        # 更新したhparamsをロガーに出力するには↓がいる
        model = Model(**config.hparams)

        trainer.fit(model, datamodule=datamodule)
        trainer.test()

        logger.info('All done.')
        all_done = True
    finally:
        if all_done == False:
            path = Path.cwd()
            if 'outputs' in path.parts or 'multirun' in path.parts:
                logger.info(f'Rename directory name. "{path}" -> "{path}__interrupted"')
                path.rename(path.parent / (path.name + '__interrupted__'))


if __name__ == "__main__":
    main()
