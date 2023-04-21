import os
import hydra
import torch
import numpy as np
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging

from utils import dataloaders
from utils import preprocessors
from utils import saver

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig):
    print("Working directory : {}".format(os.getcwd()))

    #Get the model
    model = instantiate(cfg.model)

    #Get the trainer
    trainer = instantiate(cfg.trainer)
    trainer.FILE_EXTENSION = ".pt"
    
    #Get the dataset
    datamodule = dataloaders.ChallengeDataModule(
        root_path=cfg.paths.data,
        data_file=cfg.files.features_data,
        label_file=cfg.files.labels_data,
        batch_size=cfg.compilation.batch_size,
        splits=cfg.dataset.splits,
        crop_shape=cfg.dataset.crop_shape,
        resize_shape=cfg.dataset.resize_shape,
        samples=cfg.dataset.samples,
        num_workers=cfg.compilation.num_workers,
    )

    #Fit and validate the trainer (and the model)
    logger.info("training started")

    #trainer.fit(model, datamodule=datamodule)
    logger.info("training finished")
    

if __name__ == "__main__":
    main()