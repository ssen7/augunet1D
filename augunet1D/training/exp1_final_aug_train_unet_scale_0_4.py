## Notes
# No mixed precision
# Augmentations applied during training
# Using Dice Loss (not weighted)
# Residual UNet1D model
# No weighted dataloader
# Batch size 32
# Learning rate 1e-3
# Cosine Annealing with Linear Warmup LR Scheduler
# Early Stopping with patience 10
# Train-validation split 95-5%

from dataset import MiceData,MiceDataSegmented, custom_collate_fn, custom_collate_fn_V2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


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
from ecg_augmentations.ecg_augmentations import *
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


if __name__=='__main__':

    print(watermark(packages="torch,pytorch_lightning,transformers", python=True), flush=True)
    print("Torch CUDA available?", torch.cuda.is_available(), flush=True)

    # Load Data
    # df_path='/project/GutIntelligenceLab/ss4yd/mice_data/processed_data/processed_data.csv'
    # df_path='/scratch/ss4yd/SWDUNet/final_train_test_split.csv'
    df_path='/home/ss4yd/time_segmentation/SWDUNet/final_train_test_split.csv'
    orig_freq=1000
    down_freq=100
    split_size=2000
    
    batch_size=32
    num_workers=5
    
    generator = torch.Generator().manual_seed(42)
    unaug_transforms=Compose(transforms=[])
    transforms = [
        RandomApply([Scale()], p=0.4),
        GaussianNoise(max_snr=0.005),
        RandomApply([Invert()], p=0.2),
        # RandomApply([Permute()], p=0.6),
        # RandomApply([Reverse()], p=0.2),
        # RandomApply([TimeWarp()], p=0.2),
        # RandomApply([RandMask()], p=0.2),
        # RandomApply([Reverse()], p=0.2),
        # RandomCrop(n_samples=1000),
        # RandomApply([PRMask(sample_rate=100)], p=0.4),
        # RandomApply([QRSMask(sample_rate=100)], p=0.4),
    ]

    transform = Compose(transforms=transforms)
    # unaug_dataset=MiceDataSegmented(df_path, down_freq=down_freq, split_size=split_size, dtype='train', process_dir='/scratch/ss4yd/mice_data/processed_all_data/')
    aug_dataset=MiceDataSegmented(df_path, down_freq=down_freq, split_size=split_size, transform=transforms, data_prop=1.0, dtype='train', process_dir='/scratch/ss4yd/mice_data/processed_all_data/')
    
    # total_dataset=torch.utils.data.ConcatDataset([aug_dataset, unaug_dataset])
    train_dataset, val_dataset=torch.utils.data.random_split(aug_dataset, [0.95, 0.05], generator=generator)
    # val_dataset.dataset.dtype = 'val'
    
    testds = MiceDataSegmented(df_path, down_freq=down_freq, split_size=split_size, dtype='test', process_dir='/scratch/ss4yd/mice_data/processed_all_data/')
    # testds400 = MiceDataSegmented(df_path, orig_freq=400, new_freq=100, split_size=split_size, dtype='test400')
    
    # testds = torch.utils.data.ConcatDataset([testds1000, testds400])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               num_workers=num_workers,
                                            #    collate_fn=custom_collate_fn_V2
                                               )
    
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=False, 
                                               num_workers=num_workers,
                                            #    collate_fn=custom_collate_fn_V2
                                               )
    
    test_loader1 = torch.utils.data.DataLoader(testds, 
                                               batch_size=batch_size, 
                                               shuffle=False, 
                                               num_workers=num_workers)
    
    # test_loader2 = torch.utils.data.DataLoader(testds400, 
    #                                            batch_size=batch_size, 
    #                                            shuffle=False, 
    #                                            num_workers=num_workers)
    
    
    # Initialize Model
    model = UNet1D(
        normalization='batch',
        preactivation=False,
        residual=True,
        out_classes=1,
        num_encoding_blocks=5,
        encoder_kernel_sizes=[5,5,5,5,5],
        decoder_kernel_sizes=[3,3,3,3],
    )

    # TODO
    # model = model.double()

    # Initialize model hyperparameters
    epochs=50
    model_lr=1e-3


    # lightning configuration
    lightning_model = LightningModel(model, model_lr=model_lr)

    
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="min", monitor="val_loss", filename='{epoch}-{val_loss:.2f}-{step:.2f}'),  # save top 1 model
        # ModelCheckpoint(save_last=True, filename='{epoch}-{val_bleu:.2f}-{step:.2f}'),  # save last model
        EarlyStopping(monitor="val_loss", min_delta=0.000, patience=10, verbose=False, mode="min"),
        # StochasticWeightAveraging(swa_lrs=1e-2)
    ]

    csv_logger = CSVLogger(save_dir="/home/ss4yd/time_segmentation/SWDUNet/checkpoints/", name=f"exp1_final_aug_scale_0_4")
    # ten_logger = TensorBoardLogger(save_dir="/scratch/ss4yd/logs_only_vit_gpt_bert/", name=f"my_model")

    trainer = L.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        # precision='16-mixed',
        logger=csv_logger,
        # log_every_n_steps=100,
        # deterministic=False,
        # gradient_clip_val=0.5,
        # gradient_clip_algorithm="norm",
        # accumulate_grad_batches=16,
        # detect_anomaly=True,
        # limit_train_batches=0.1, 
        # limit_val_batches=1.0,
        # limit_test_batches=0.1
    )

    start = time.time()
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

    test_bleu1 = trainer.test(lightning_model, test_loader1, ckpt_path="best")
    # test_bleu2 = trainer.test(lightning_model, test_loader2, ckpt_path="best")

    with open(os.path.join(trainer.logger.log_dir, "outputs.txt"), "w") as f:
        f.write((f"Time elapsed {elapsed/60:.2f} min\n"))
        f.write(f"Test Accuracy: {test_bleu1}\n")
        # f.write(f"Test Accuracy 400Hz: {test_bleu2}")