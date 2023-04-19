import torch
import torch.nn as nn
import lightning.pytorch as pl
import torchmetrics
import torch.nn.functional as F
from hydra.utils import instantiate
import logging
from collections import OrderedDict

import utils.metrics as metrics
from utils import graphics

logger = logging.getLogger(__name__)

class ImageModelMobileNetV2(pl.LightningModule):
    def __init__(self, 
        loss_fn: torch.nn=None,
        optimizer_fc=None,
        resize_shape:int=224,
        num_labels:int=27,
        name:str="nonamemodel",
        ) -> None:
        
        super().__init__()

        self.validation_step_data = []

        self.acc = metrics.acc(num_labels)
        self.val_acc = metrics.val_acc(num_labels)
        self.loss = metrics.loss(num_labels)
        self.val_loss = metrics.val_loss(num_labels)

        self.num_features_fc1 = int(resize_shape / 2 * resize_shape / 2 * 6)

        #Load
        self.model = torch.hub.load(
            'pytorch/vision:v0.10.0', 
            'mobilenet_v2', 
            pretrained=True)

        #Freeze the layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        #Change the classifier
        self.model.classifier = nn.Sequential(OrderedDict([
            ('drop', nn.Dropout(p=0.2, inplace=False)), 
            ('dense', nn.Linear(1280, num_labels)), 
        ])) 

        self.loss_fn = loss_fn
        self.optimizer_fc = instantiate(optimizer_fc, params=self.parameters())

        #self.save_hyperparameters()

        logger.info(f"model {type(self).__name__} initialized")

    def configure_optimizers(self):
                
        return torch.optim.Adam(self.model.classifier.parameters(), lr=0.001)

    def forward(self, image):

        #image = batch["images"]

        x = self.model(image)

        return x

    def training_step(self, batch, batch_idx):

        labels = batch["labels"]
        images = batch["images"]

        y_probas = self.forward(images)
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

        y_probas = self.forward(images)
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