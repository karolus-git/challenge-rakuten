import os
import lightning.pytorch as pl
import torch

from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Any
import lightning.pytorch as pl

class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.FILE_EXTENSION = ".pb"




def to_pt(trainer, raw_checkpoint_path):

    raw_checkpoint = torch.load(raw_checkpoint_path)
    #raw_checkpoint[past_key] = raw_checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
    raw_checkpoint['hparams_type'] = 'Namespace'
    #raw_checkpoint[past_key]['batch_size'] = -17
    #del raw_checkpoint[pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
    # save back the checkpoint
    torch.save(raw_checkpoint, raw_checkpoint_path)


def make_archive():
    command = "torch-model-archiver \
        --model-name mnist \
        --version 1.0 \
        --model-file model.py \
        --serialized-file model.pt \
        --handler handler.py"

    res = os.system(command)
    #the method returns the exit status
    
    print("Returned Value: ", res)

if __name__ == "__main__":
    make_archive()
    import pdb;pdb.set_trace()