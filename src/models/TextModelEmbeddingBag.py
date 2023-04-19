import torch
import torch.nn as nn
import lightning.pytorch as pl
import torchmetrics
import torch.nn.functional as F
from hydra.utils import instantiate
import logging

import utils.metrics as metrics
from utils import graphics

logger = logging.getLogger(__name__)

class TextModelEmbeddingBag(pl.LightningModule):
    def __init__(self, 
        loss_fn: torch.nn=None,
        optimizer_fc=None,
        vocab_size:int=32005,
        embedding_dim:int=200,
        num_labels:int=27, 
        name:str="nonamemodel",
        tokenizer=None,
        ) -> None:

        super().__init__()

        self.validation_step_data = []

        self.acc = metrics.acc(num_labels)
        self.val_acc = metrics.val_acc(num_labels)
        self.loss = metrics.loss(num_labels)
        self.val_loss = metrics.val_loss(num_labels)

        self.vocab_size = vocab_size

        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=False)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(embedding_dim, num_labels)
        self.drop = nn.Dropout(p=0.2)
 
        self.loss_fn = loss_fn
        self.optimizer_fc = instantiate(optimizer_fc, params=self.parameters())

        #self.save_hyperparameters()
        logger.info(f"model {type(self).__name__} initialized")

    def configure_optimizers(self):
                
        return self.optimizer_fc

    def forward(self, texts_ids, offsets):

        x = self.embedding(texts_ids, offsets)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.classifier(x)
        #x = self.softmax(x)

        return x

    def training_step(self, batch, batch_idx):

        labels = batch["labels"]
        texts = batch["input_ids_bag"]
        offsets = batch["offsets_bag"]

        y_probas = self.forward(texts, offsets)
        y_preds = y_probas.argmax(axis=1)
        loss = self.loss_fn(y_probas, labels)

        self.acc(y_preds, labels)
        self.loss(y_probas, labels)
        self.log('acc', self.acc, on_step=False, on_epoch=True)
        self.log('loss', self.loss, on_step=False, on_epoch=True)

        return loss
        
    def validation_step(self, batch, batch_idx):

        labels = batch["labels"]
        texts = batch["input_ids_bag"]
        offsets = batch["offsets_bag"]

        y_probas = self.forward(texts, offsets)
        y_preds = y_probas.argmax(axis=1)
        loss = self.loss_fn(y_probas, labels)

        self.val_acc(y_preds, labels)
        self.val_loss(y_probas, labels)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        self.log('val_loss', self.val_loss, on_step=False, on_epoch=True)

        #Add the labels and the predictions to validation_step_data
        self.validation_step_data.append(torch.stack([labels, y_preds]))

        return loss

    def on_validation_epoch_end(self):

        #Get the labels and predictions during validation
        labels, y_preds = torch.hstack(self.validation_step_data)
        
        #Create a confusion matrix plot
        cm_fig = graphics.plot_cm(labels, y_preds)

        #Add it to the logger as figure
        self.logger.experiment.add_figure('confusion_matrix_validation', cm_fig, global_step=self.current_epoch)

        #Reset the tuple
        self.validation_step_data.clear()

    def on_fit_end(self):
        
        #Add the embedding layer to the logger at end of training
        self.logger.experiment.add_embedding(
            self.embedding.weight,
            tag="embedding")