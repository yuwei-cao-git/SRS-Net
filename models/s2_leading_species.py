import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassF1Score,
    ConfusionMatrix,
    MulticlassAccuracy,
)

from .blocks import MF
from .unet import UNet
from .ResUnet import ResUnet
from .TransResUnet import FusionBlock
from .loss import apply_mask, focal_loss_multiclass


# Updating UNet to incorporate residual connections and MF module
class Model(pl.LightningModule):
    def __init__(self, config, vis):
        super(Model, self).__init__()
        self.config = config
        self.test_csv_written = False
        self.sample_id_offset = 0
        self.vis_mode = vis
        self.fusion_mode = self.config["fusion_mode"]
        self.use_fuse = False
        self.network = self.config["network"]
        self.loss = self.config["loss"]
        self.leading_loss = self.config["leading_loss"]
        
        self.remove_bands = self.config["remove_bands"]
        if self.config["resolution"] == "10m":
            if self.remove_bands:
                self.n_bands = 9
            else:
                self.n_bands = 12
        else:
            self.n_bands = 9
        self.season = self.config["season"]
        if self.config["season"] == "2seasons":
            self.num_season = 2
            input_channels = [self.n_bands, self.n_bands]
        elif self.config["season"] == "4seasons":
            self.num_season = 4
            input_channels = [self.n_bands, self.n_bands, self.n_bands, self.n_bands]
        elif self.config["season"] == "cli4seasons":
            self.num_season = 5
            input_channels = [
                self.n_bands,
                self.n_bands,
                self.n_bands,
                self.n_bands,
                36,
            ]
        elif (
            self.config["season"] == "dem4seasons"
            or self.config["season"] == "ph4seasons"
        ):
            self.num_season = 5
            input_channels = [self.n_bands, self.n_bands, self.n_bands, self.n_bands, 1]
        elif self.config["season"] == "all":
            self.num_season = 7
            input_channels = [
                self.n_bands,
                self.n_bands,
                self.n_bands,
                self.n_bands,
                1,
                36,
                1,
            ]
        else:
            self.num_season = 1
            
        if self.num_season != 1:
            # MF Module for seasonal fusion (each season has `n_bands` channels)
            if self.fusion_mode == "sf":
                self.mf_module = FusionBlock(
                    n_inputs=self.num_season, in_chs=input_channels, n_filters=64
                )
                total_input_channels = 64
                self.use_fuse = True
            elif self.fusion_mode == "stack":
                if self.num_season < 5:
                    total_input_channels = (
                        self.n_bands * self.num_season
                    )  # If no MF module, concatenating all seasons directly
                elif self.num_season == 5:
                    if self.config["season"] == "cli4seasons":
                        total_input_channels = (
                            self.n_bands * 4 + 36  # all seasons + climate (36)
                        )
                    else:
                        total_input_channels = (
                            self.n_bands * 4 + 1  # all seasons + dem (1) / ph (1)
                        )
                else:
                    total_input_channels = (
                        self.n_bands * 4
                        + 38  # all seasons + dem (1) + climate (36) + ph (1)
                    )  # If no MF module, concatenating all seasons directly
            else:
                if self.config["season"] == "cli4seasons":
                    add_channel = 36
                elif self.config["season"] == "all":
                    add_channel = 38
                else:
                    add_channel = 1
                self.mf_module = MF(
                    channels=self.n_bands,
                    seasons=self.num_season,
                    rest_channel=add_channel,
                    spatial_att=self.fusion_mode == "cs_mf",
                )
                total_input_channels = (
                    16
                    * self.num_season  # MF module outputs 64 channels after processing four seasons
                )
                self.use_fuse = True
        else:
            total_input_channels = self.n_bands
        # Define the U-Net architecture with or without Residual connections
        if self.network == "resunet":
            # Using ResUNet
            self.model = ResUnet(
                n_channels=total_input_channels, n_classes=self.config["n_classes"]
            )
        else:
            # Using standard UNet
            self.model = UNet(
                n_channels=total_input_channels, n_classes=self.config["n_classes"]
            )

        # Loss function
        self.criterion = nn.MSELoss()
        self.weights = self.config["prop_weights"]

        # Metrics
        self.train_f1 = MulticlassF1Score(
            num_classes=self.config["n_classes"], average="weighted"
        )
        self.val_f1 = MulticlassF1Score(
            num_classes=self.config["n_classes"], average="weighted", ignore_index=255
        )
        self.val_oa = MulticlassAccuracy(
            num_classes=self.config["n_classes"], average="micro", ignore_index=255
        )
        self.test_f1 = MulticlassF1Score(
            num_classes=self.config["n_classes"], average="weighted", ignore_index=255
        )
        self.test_oa = MulticlassAccuracy(
            num_classes=self.config["n_classes"], average="micro", ignore_index=255
        )
        self.confmat = ConfusionMatrix(
            task="multiclass", num_classes=self.config["n_classes"], ignore_index=255
        )

        # Optimizer and scheduler settings
        self.optimizer_type = self.config["optimizer"]
        self.learning_rate = self.config["learning_rate"]
        self.scheduler_type = self.config["scheduler"]

        # Containers for validation predictions and true labels
        self.best_test_outputs = None
        self.best_val_metric = None
        self.validation_step_outputs = []

        self.save_hyperparameters()

    def forward(self, inputs):
        # Optionally pass inputs through MF module
        if self.use_fuse:
            # Apply the MF module first to extract features from input
            fused_features = self.mf_module(inputs)
        else:
            if self.num_season != 1:
                # Concatenate all seasons directly if no MF module
                fused_features = torch.cat(inputs, dim=1)
            else:
                fused_features = inputs
        logits, _ = self.model(fused_features)  # ([16, 9, 128, 128])
        logits = F.log_softmax(logits, dim=1)  # ([16, 9, 128, 128])
        return logits

    def compute_loss_and_metrics(self, pixel_logits, targets, img_masks, stage="val"):
        """
        Computes the masked loss, R² score, and logs the metrics.

        Args:
        - outputs: Predicted values (batch_size, num_channels, H, W)
        - targets: Ground truth values (batch_size, num_channels, H, W)
        - masks: Boolean mask indicating NoData pixels (batch_size, H, W)
        - stage: One of 'train', 'val', or 'test', used for logging purposes.

        Returns:
        - loss: The computed masked loss.
        """
        logs = {}
        # Convert outputs and targets to leading class labels by taking argmax
        pred_lead_pixel_labels = torch.argmax(pixel_logits, dim=1)  # ([16, 128, 128])
        true_lead_pixel_labels = torch.argmax(targets, dim=1)  # ([16, 128, 128])
        valid_pixel_lead_preds, valid_pixel_lead_true = apply_mask(
            pred_lead_pixel_labels,
            true_lead_pixel_labels,
            img_masks,
            multi_class=False,
            keep_shp=True,
        )
        if self.config["loss"] == "wce" and stage == "train":
            loss_pixel_leads = F.nll_loss(
                pixel_logits,
                valid_pixel_lead_true,
                weight=self.weights.to(pixel_logits.device),
                ignore_index=255,
            )
        else:
            loss_pixel_leads = F.nll_loss(
                pixel_logits,
                valid_pixel_lead_true,
                ignore_index=255,
            )

        # Calculate R² and F1 score for valid pixels
        if stage == "val":
            f1 = self.val_f1(valid_pixel_lead_preds, valid_pixel_lead_true)
            oa = self.val_oa(valid_pixel_lead_preds, valid_pixel_lead_true)
        elif stage == "test":
            f1 = self.test_f1(valid_pixel_lead_preds, valid_pixel_lead_true)
            oa = self.test_oa(valid_pixel_lead_preds, valid_pixel_lead_true)

        # Compute RMSE
        rmse = torch.sqrt(loss_pixel_leads)
        # Log metrics
        logs.update(
            {
                f"{stage}_loss": loss_pixel_leads,
                f"{stage}_rmse": rmse,
            }
        )
        if stage != "train":
            logs.update(
                {
                    f"{stage}_f1": f1,
                    f"{stage}_oa": oa,
                }
            )

        if stage == "val":
            self.validation_step_outputs.append(
                {
                    "val_target": valid_pixel_lead_true,
                    "val_pred": valid_pixel_lead_preds,
                }
            )

        # Log val_loss if in validation stage for ModelCheckpoint
        sync_state = True
        self.log(
            f"{stage}_loss",
            loss_pixel_leads,
            logger=True,
            sync_dist=sync_state,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            f"{stage}_rmse",
            rmse,
            logger=True,
            sync_dist=sync_state,
            on_step=True,
            on_epoch=(stage != "train"),
        )
        if stage != "train":
            self.log(
                f"{stage}_f1",
                f1,
                logger=True,
                prog_bar=True,
                sync_dist=sync_state,
                on_step=True,
                on_epoch=(stage != "train"),
            )
        if stage != "train":
            self.log(
                f"{stage}_oa",
                oa,
                logger=True,
                sync_dist=sync_state,
                on_epoch=True,
            )
        if stage == "test":
            if self.vis_mode:
                return (
                    valid_pixel_lead_true,
                    valid_pixel_lead_preds,
                    loss_pixel_leads,
                )
            else:
                return (
                    valid_pixel_lead_true.view(-1),
                    valid_pixel_lead_preds.view(-1),
                    loss_pixel_leads,
                )
        else:
            return loss_pixel_leads

    def training_step(self, batch, batch_idx):
        inputs, targets, masks, _ = batch
        outputs = self(inputs)  # Forward pass #[batch_size, n_classes, height, width]

        return self.compute_loss_and_metrics(outputs, targets, masks, stage="train")

    def validation_step(self, batch, batch_idx):
        inputs, targets, masks, _ = batch
        outputs = self(inputs)  # Forward pass

        return self.compute_loss_and_metrics(outputs, targets, masks, stage="val")

    def on_validation_epoch_end(self):
        sys_f1 = self.val_f1.compute()

        # Concatenate all predictions and true labels
        test_true = torch.cat(
            [output["val_target"] for output in self.validation_step_outputs], dim=0
        )
        test_pred = torch.cat(
            [output["val_pred"] for output in self.validation_step_outputs], dim=0
        )
        self.log("sys_f1", sys_f1, sync_dist=True)
        # Determine if current epoch has the best validation metric
        is_best = False
        if self.best_val_metric is None or sys_f1 > self.best_val_metric:
            is_best = True
            self.best_val_metric = sys_f1

        if is_best:
            print(f"F1 Score:{sys_f1}")
            cm = self.confmat(test_pred, test_true)
            print(f"OA Score:{self.val_oa.compute()}")
            print(cm)
        # Clear buffers for the next epoch
        self.validation_step_outputs.clear()
        self.val_f1.reset()
        self.val_oa.reset()

    def test_step(self, batch, batch_idx):
        if self.vis_mode:
            inputs, targets, masks, tile_names = batch
        else:
            inputs, targets, masks, _ = batch
        outputs = self(inputs)  # Forward pass

        labels, preds, loss = self.compute_loss_and_metrics(
            outputs, targets, masks, stage="test"
        )
        if self.vis_mode:
            self.save_to_img(preds, tile_names)
        else:
            self.save_to_csv(labels, preds)
        return loss

    def save_to_csv(self, labels, outputs):
        # Convert tensors to numpy arrays or lists as necessary
        labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        outputs = (outputs.cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs)
        num_samples = labels.shape[0]

        # Use a running counter to avoid resetting SampleID
        data = {"SampleID": np.arange(self.sample_id_offset, self.sample_id_offset + num_samples)}
        self.sample_id_offset += num_samples  # Update for the next batch

        data["True"] = labels[:]
        data["Pred"] = outputs[:]

        df = pd.DataFrame(data)

        output_dir = os.path.join(
            self.config["save_dir"], 
            self.config["log_name"],
            "outputs"
        )
        os.makedirs(output_dir, exist_ok=True)
        # Save DataFrame to a CSV file
        df.to_csv(
            os.path.join(output_dir, f"{self.config['log_name']}_test_outputs.csv"),
            mode="a",
            header=not self.test_csv_written,  # Only write header on first write
            index=False,
        )
        self.test_csv_written = True  # Flip the flag
    
    def save_to_img(self, preds, tile_names):
        import rasterio
        
        # Output path
        output_dir = os.path.join(
            self.config["save_dir"], 
            self.config["log_name"],
            "outputs", "lsc_predictions"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        preds = preds.detach().cpu().numpy()
        for i, tile_name in enumerate(tile_names):
            # preds[i]: Tensor of shape (n_classes, H, W)
            # Path to the original label to get spatial info
            label_path = os.path.join(
                self.config["data_dir"],
                self.config["resolution"],
                "labels/tiles_128",
                tile_name,
            )

            with rasterio.open(label_path) as src:
                meta = src.meta.copy()

            # Update metadata for prediction output
            meta.update({
                "count": 1,
                "dtype": np.uint8,
                "driver": "GTiff"
            })

            out_path = os.path.join(output_dir, tile_name)

            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(preds[i], 1)  # Write to band 1

    def configure_optimizers(self):
        # Choose the optimizer based on input parameter
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08
            )
        elif self.optimizer_type == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        # Configure the scheduler based on the input parameter
        if self.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, factor=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",  # Reduce learning rate when 'val_loss' plateaus
                },
            }
        elif self.scheduler_type == "asha":
            return optimizer
        elif self.scheduler_type == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler_type == "cosinewarmup":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
