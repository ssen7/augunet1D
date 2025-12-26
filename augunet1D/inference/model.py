
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from watermark import watermark
import torch.optim
import torch.utils.data

from unet.unet import UNet1D
from losses.dice import DiceLoss
# from losses.focal import FocalLoss

from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
import time
import os
import torch.nn.functional as F

import torchvision
import torchaudio
# from ecg_augmentations.ecg_augmentations import *
from torchvision.transforms import Compose

# https://github.com/santurini/cosine-annealing-linear-warmup/tree/main/cosine-warmup
from cosine_warmup.scheduler import CosineAnnealingLinearWarmup

torch.set_float32_matmul_precision('high')

class LightningModel(L.LightningModule):

    def __init__(self, model, model_lr):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])
        self.model=model
        self.model_lr=model_lr

        self.loss_fn = DiceLoss(mode='binary', from_logits=True)

        self.val_acc = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        
        self.test_acc = BinaryAccuracy()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()

    def forward(self, ecog):
        return self.model(ecog)

    def training_step(self, batch, batch_idx):
        label, ecog = batch
        label = label.unsqueeze(1)

        output = self(ecog)
        # preds = torch.argmax(output,axis=1)
        loss = self.loss_fn(output, label)

        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        label, ecog = batch
        label = label.unsqueeze(1)

        output = self(ecog)
        prob_mask = output.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        
        loss = self.loss_fn(output, label)

        self.val_acc(pred_mask, label)
        self.val_precision(pred_mask, label)
        self.val_recall(pred_mask, label)
        self.val_f1(pred_mask, label)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_prec", self.val_precision, prog_bar=True)
        self.log("val_recall", self.val_recall, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=False)

    def test_step(self, batch, batch_idx):
        label, ecog = batch
        label = label.unsqueeze(1)

        output = self(ecog)

        prob_mask = output.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        
        self.test_acc(pred_mask, label)
        self.test_precision(pred_mask, label)
        self.test_recall(pred_mask, label)
        self.test_f1(pred_mask, label)
        self.log("accuracy", self.test_acc, prog_bar=True)
        self.log("test_precision", self.test_precision, prog_bar=True)
        self.log("test_recall", self.test_recall, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.model_lr
        )
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            # "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=6),
            "scheduler": CosineAnnealingLinearWarmup(optimizer = optimizer,min_lrs = [ 1e-5 ], first_cycle_steps = 1000, warmup_steps = 500, gamma = 0.9),
            "monitor": "val_loss",
            "frequency": 1,
            "interval":"epoch"
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
            },
        }
