import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms.v2 as transforms

from .blocks import MF
from .unet import UNet
from .ResUnet import ResUnet
from .loss import calc_loss, mask_output
from .loss import cal_leading_loss

from torchmetrics.classification import ConfusionMatrix
from torchmetrics.regression import R2Score
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassAccuracy,
)
from torchmetrics import MetricCollection
from torchmetrics.wrappers import MultitaskWrapper


# Updating UNet to incorporate residual connections and MF module
class Model(pl.LightningModule):
    def __init__(self, config):
        """
        Args:
            n_bands (int): Number of input channels (bands) for each season.
            n_classes (int): Number of output classes.
            use_mf (bool): Whether to use the MF module.
            use_residual (bool): Whether to use Residual connections in U-Net blocks.
            optimizer_type (str): Type of optimizer ('adam', 'sgd', etc.).
            learning_rate (float): Learning rate for the optimizer.
            scheduler_type (str): Type of scheduler ('plateau', etc.).
            scheduler_params (dict): Parameters for the scheduler (e.g., 'patience', 'factor' for ReduceLROnPlateau).
        """
        super(Model, self).__init__()
        self.config = config
        self.use_mf = self.config["use_mf"]
        self.spatial_attention = self.config["spatial_attention"]
        self.use_residual = self.config["use_residual"]
        self.loss = self.config["loss"]
        self.leading_loss = self.config["leading_loss"]
        self.season = self.config["season"]
        self.num_season = 1
        if self.config["season"] == "2seasons":
            self.num_season = 2
        if self.config["season"] == "4seasons":
            self.num_season = 4
        if self.config["resolution"] == "10m":
            self.n_bands = 12
        else:
            self.n_bands = 9
        if self.num_season != 1:
            if self.use_mf:
                # MF Module for seasonal fusion (each season has `n_bands` channels)
                self.mf_module = MF(
                    channels=self.n_bands,
                    seasons=self.num_season,
                    spatial_att=self.spatial_attention,
                )
                total_input_channels = (
                    64  # MF module outputs 64 channels after processing four seasons
                )
            else:
                total_input_channels = (
                    self.n_bands * self.num_season
                )  # If no MF module, concatenating all seasons directly
                self.spatial_attention = False
        else:
            total_input_channels = self.n_bands
            self.use_mf = False
        # Define the U-Net architecture with or without Residual connections
        if self.use_residual:
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
        metrics = MultitaskWrapper(
            {
                "Classification": MetricCollection(
                    MulticlassF1Score(
                        num_classes=self.config["n_classes"], average="weighted"
                    ),
                    MulticlassAccuracy(
                        num_classes=self.config["n_classes"], average="micro"
                    ),
                ),
                "Regression": MetricCollection(
                    R2Score(),
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.confmat = ConfusionMatrix(
            task="multiclass", num_classes=self.config["n_classes"]
        )

        # Optimizer and scheduler settings
        self.optimizer_type = self.config["optimizer"]
        self.learning_rate = self.config["learning_rate"]
        self.scheduler_type = self.config["scheduler"]

    def forward(self, inputs):
        if self.use_mf:
            # Apply the MF module first to extract features from input
            fused_features = self.mf_module(inputs)
        else:
            # Concatenate all seasons directly if no MF module
            fused_features = torch.cat(inputs, dim=1)
        logits, _ = self.model(fused_features)
        return logits

    def training_step(self, batch, batch_idx):
        inputs, targets, masks = batch
        outputs = self(inputs)  # Forward pass #[batch_size, n_classes, height, width]
        # Compute the masked loss
        self.weights = self.weights.to(outputs.device)
        loss = calc_loss(self.loss, outputs, targets, masks, self.weights)

        valid_outputs, valid_targets = mask_output(
            outputs, targets, masks, stage="train"
        )
        if self.leading_loss:
            leading_loss = cal_leading_loss(
                valid_targets["train_Classification"],
                valid_outputs["train_Classification"],
                alpha_leading=0.2,
            )
            loss += leading_loss
        self.log(
            "train_loss", loss, logger=True, sync_dist=True, on_step=True, on_epoch=True
        )
        # compute metrics
        metrics = self.train_metrics(valid_outputs, valid_targets)

        self.log_metrics(metrics)

    def validation_step(self, batch, batch_idx):
        inputs, targets, masks = batch
        outputs = self(inputs)  # Forward pass
        valid_outputs, valid_targets = mask_output(outputs, targets, masks, stage="val")
        # Compute the masked loss
        loss = self.criterion(
            valid_outputs["val_Regression"], valid_targets["val_Regression"]
        )
        # Compute RMSE
        rmse = torch.sqrt(loss)
        self.log(
            "val_loss", loss, logger=True, sync_dist=True, on_step=True, on_epoch=True
        )
        self.log("val_rmse", rmse, sync_dist=True, on_step=True, on_epoch=True)

        # compute metrics
        self.valid_metrics.update(valid_outputs, valid_targets)

    def on_validation_epoch_end(self):
        output = self.valid_metrics.compute()
        # Log each metric individually
        self.log_metrics(output)
        # remember to reset metrics at the end of the epoch
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        inputs, targets, masks = batch
        outputs = self(inputs)  # Forward pass
        valid_outputs, valid_targets = mask_output(
            outputs, targets, masks, stage="test"
        )
        # Compute the masked loss
        loss = self.criterion(
            valid_outputs["test_Regression"], valid_targets["test_Regression"]
        )
        # Compute RMSE
        rmse = torch.sqrt(loss)
        self.log("test_rmse", rmse, sync_dist=True, on_step=False, on_epoch=True)

        output = self.test_metrics(valid_outputs, valid_targets)
        self.log_metrics(output)

    def log_metrics(self, output_metrics):
        for task, metrics in output_metrics.items():
            if isinstance(metrics, dict):  # Handle nested dictionaries
                for metric_name, metric_value in metrics.items():
                    full_metric_name = f"{task}_{metric_name}"
                    if (
                        isinstance(metric_value, torch.Tensor)
                        and metric_value.numel() == 1
                    ):
                        metric_value = metric_value.item()  # Convert tensor to scalar
                    self.log(
                        full_metric_name,
                        metric_value,
                        sync_dist=True,
                        logger=True,
                        on_epoch=True,
                    )
            else:  # Handle non-nested metrics (if any exist)
                if isinstance(metrics, torch.Tensor) and metrics.numel() == 1:
                    metrics = metrics.item()
                    self.log(
                        task,
                        metrics,
                        logger=True,
                        on_epoch=True,
                        sync_dist=True,
                    )

    def configure_optimizers(self):
        # Choose the optimizer based on input parameter
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08
            )
        elif self.optimizer_type == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
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
                optimizer,
                T_max=50,
                eta_min=0,
                last_epoch=-1,
                verbose=False,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
