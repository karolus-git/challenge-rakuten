import torch
import torch.nn as nn
import lightning.pytorch as pl
import torchmetrics
import torch.nn.functional as F
from hydra.utils import instantiate
import logging
from collections import OrderedDict
import utils.metrics as metrics

from models import TextModelEmbeddingBag
from models import ImageModelMobileNetV2
from utils import graphics

logger = logging.getLogger(__name__)

class FusionModel(pl.LightningModule):
    def __init__(self, 
        loss_fn: torch.nn=None,
        optimizer_fc=None,
        num_labels:int=27,
        name:str="nonamemodel"
        ) -> None:
        
        super().__init__()

        self.validation_step_data = []

        self.acc = metrics.acc(num_labels)
        self.val_acc = metrics.val_acc(num_labels)
        self.loss = metrics.loss(num_labels)
        self.val_loss = metrics.val_loss(num_labels)

        model_image = ImageModelMobileNetV2.ImageModelMobileNetV2()
        model_image.load_state_dict(torch.load("./runs/ImageModelMobileNetV2/version_2/ImageModelMobileNetV2.pt"))
        model_image.eval()

        #Load the image model
        model_text = TextModelEmbeddingBag.TextModelEmbeddingBag()
        model_text.load_state_dict(torch.load("./runs/TextModelEmbeddingBag/version_0/TextModelEmbeddingBag.pt"))
        model_text.eval()

        #Freeze the layers
        for param in model_text.parameters():
            param.requires_grad = False
        for param in model_image.parameters():
            param.requires_grad = False

        self.model_image = model_image
        self.model_text = model_text
        self.classifier = nn.Linear(54, 27)

        self.loss_fn = loss_fn
        self.optimizer_fc = instantiate(optimizer_fc, params=self.parameters())

        #self.save_hyperparameters()

        logger.info(f"model {type(self).__name__} initialized")

    def configure_optimizers(self):
                
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, texts, offsets, images):

        x_text = self.model_text(texts, offsets)
        x_image = self.model_image(images)

        x = torch.cat((x_text, x_image), dim=1)
        x = self.classifier(x)


        return x

    def training_step(self, batch, batch_idx):

        labels = batch["labels"]
        images = batch["images"]
        texts = batch["input_ids_bag"]
        offsets = batch["offsets_bag"]

        y_probas = self.forward(texts, offsets, images)
        y_preds = y_probas.argmax(axis=1)
        loss = self.loss_fn(y_probas, labels)

        self.acc(y_preds, labels)
        self.loss(y_probas, labels)
        self.log('acc', self.acc, on_step=False, on_epoch=True)
        self.log('loss', self.loss, on_step=False, on_epoch=True)

        return loss
        
    def validation_step(self, batch, batch_idx):

        labels = batch["labels"]
        images = batch["images"]
        texts = batch["input_ids_bag"]
        offsets = batch["offsets_bag"]

        y_probas = self.forward(texts, offsets, images)
        y_preds = y_probas.argmax(axis=1)
        loss = self.loss_fn(y_probas, labels)

        self.val_acc(y_preds, labels)
        self.val_loss(y_probas, labels)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        self.log('val_loss', self.val_loss, on_step=False, on_epoch=True)

        self.validation_step_data.append(torch.stack([labels, y_preds]))

        return loss

    def on_validation_epoch_end(self):

        labels, y_preds = torch.hstack(self.validation_step_data)
        
        tensorboard = self.logger.experiment
        cm_fig = graphics.plot_cm(labels, y_preds)
        tensorboard.add_figure('confusion_matrix_validation', cm_fig, global_step=self.current_epoch)
  
        self.validation_step_data.clear()  # clean