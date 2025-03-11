import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms.v2 as transforms

from .blocks import MF
from .unet import UNet
from .ResUnet import ResUnet
from .TransResUnet import FusionBlock

from torchmetrics.regression import R2Score
from torchmetrics.classification import (
    MulticlassF1Score,
    ConfusionMatrix,
    MulticlassAccuracy,
)
from .loss import calc_masked_loss


# Updating UNet to incorporate residual connections and MF module
class Model(pl.LightningModule):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.fusion_mode = self.config["fusion_mode"]
        self.use_fuse = True
        self.network = self.config["network"]
        self.loss = self.config["loss"]
        self.leading_loss = self.config["leading_loss"]
        self.season = self.config["season"]
        self.remove_bands = self.config["remove_bands"]
        self.num_season = 1
        if self.config["season"] == "2seasons":
            self.num_season = 2
        if self.config["season"] == "4seasons":
            self.num_season = 4
        if self.config["season"] == "all":
            self.num_season = 5
        if self.config["resolution"] == "10m":
            if self.remove_bands:
                self.n_bands = 9
            else:
                self.n_bands = 12
        else:
            self.n_bands = 9
        if self.num_season != 1:
            # MF Module for seasonal fusion (each season has `n_bands` channels)
            if self.fusion_mode == "sf":
                self.mf_module = FusionBlock(
                    n_inputs=self.num_season, in_ch=self.n_bands, n_filters=64
                )
                total_input_channels = 64
            elif self.fusion_mode == "stack":
                if self.num_season != 5:
                    total_input_channels = (
                        self.n_bands * self.num_season
                    )  # If no MF module, concatenating all seasons directly
                else:
                    total_input_channels = (
                        self.n_bands * 4
                        + 38  # all seasons + dem (1) + climate (36) + ph (1)
                    )  # If no MF module, concatenating all seasons directly
                self.use_fuse = False
            else:
                self.mf_module = MF(
                    channels=self.n_bands,
                    seasons=self.num_season,
                    spatial_att=self.fusion_mode == "cs_mf",
                )
                total_input_channels = (
                    16
                    * self.num_season  # MF module outputs 64 channels after processing four seasons
                )
        else:
            total_input_channels = self.n_bands
            self.use_fuse = False
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
        self.train_r2 = R2Score()
        self.train_f1 = MulticlassF1Score(
            num_classes=self.config["n_classes"], average="weighted"
        )

        self.val_r2 = R2Score()
        self.val_f1 = MulticlassF1Score(
            num_classes=self.config["n_classes"], average="weighted"
        )

        self.test_r2 = R2Score()
        self.test_f1 = MulticlassF1Score(
            num_classes=self.config["n_classes"], average="weighted"
        )
        self.test_oa = MulticlassAccuracy(
            num_classes=self.config["n_classes"], average="micro"
        )
        self.confmat = ConfusionMatrix(
            task="multiclass", num_classes=self.config["n_classes"]
        )

        # Optimizer and scheduler settings
        self.optimizer_type = self.config["optimizer"]
        self.learning_rate = self.config["learning_rate"]
        self.scheduler_type = self.config["scheduler"]

        # Containers for validation predictions and true labels
        self.best_test_outputs = None
        self.best_val_metric = None

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
        logits, _ = self.model(fused_features)
        preds = F.softmax(logits, dim=1)
        return preds

    def apply_mask(self, outputs, targets, mask, multi_class=True):
        """
        Applies the mask to outputs and targets to exclude invalid data points.

        Args:
            outputs: Model predictions (batch_size, num_classes, H, W) for images or (batch_size, num_points, num_classes) for point clouds.
            targets: Ground truth labels (same shape as outputs).
            mask: Boolean mask indicating invalid data points (True for invalid).

        Returns:
            valid_outputs: Masked and reshaped outputs.
            valid_targets: Masked and reshaped targets.
        """
        # Expand the mask to match outputs and targets
        if multi_class:
            expanded_mask = mask.unsqueeze(1).expand_as(
                outputs
            )  # Shape: (batch_size, num_classes, H, W)
            num_classes = outputs.size(1)
        else:
            expanded_mask = mask

        # Apply mask to exclude invalid data points
        valid_outputs = outputs[~expanded_mask]
        valid_targets = targets[~expanded_mask]
        # Reshape to (-1, num_classes)
        if multi_class:
            valid_outputs = valid_outputs.view(-1, num_classes)
            valid_targets = valid_targets.view(-1, num_classes)

        return valid_outputs, valid_targets

    def compute_loss_and_metrics(self, outputs, targets, masks, stage="val"):
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
        valid_outputs, valid_targets = self.apply_mask(
            outputs, targets, masks, multi_class=True
        )

        # Convert outputs and targets to leading class labels by taking argmax
        pred_labels = torch.argmax(outputs, dim=1)
        true_labels = torch.argmax(targets, dim=1)
        # Apply mask to leading species labels
        valid_preds, valid_true = self.apply_mask(
            pred_labels, true_labels, masks, multi_class=False
        )
        if stage == "train":
            # Compute the masked loss
            self.weights = self.weights.to(outputs.device)
            loss = calc_masked_loss(
                self.loss, valid_outputs, valid_targets, self.weights
            )
        else:
            loss = self.criterion(valid_outputs, valid_targets)

        if self.leading_loss and stage == "train":
            correct = (valid_preds.view(-1) == valid_true.view(-1)).float()
            loss_pixel_leads = 1 - correct.mean()  # 1 - accuracy as pseudo-loss
            loss += loss_pixel_leads

        # Round outputs to two decimal place
        valid_outputs = torch.round(valid_outputs, decimals=1)

        # Calculate R² and F1 score for valid pixels
        if stage == "train":
            r2 = self.train_r2(valid_outputs.view(-1), valid_targets.view(-1))
        elif stage == "val":
            r2 = self.val_r2(valid_outputs.view(-1), valid_targets.view(-1))
            f1 = self.val_f1(valid_preds, valid_true)
        else:
            r2 = self.test_r2(valid_outputs.view(-1), valid_targets.view(-1))
            f1 = self.test_f1(valid_preds, valid_true)
            oa = self.test_oa(valid_preds, valid_true)

        # Compute RMSE
        rmse = torch.sqrt(loss)

        # Log val_loss if in validation stage for ModelCheckpoint
        sync_state = True
        self.log(
            f"{stage}_loss",
            loss,
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
        self.log(
            f"{stage}_r2",
            r2,
            logger=True,
            prog_bar=True,
            sync_dist=sync_state,
            on_step=True,
            on_epoch=True,
        )
        if stage != "train":
            self.log(
                f"{stage}_f1",
                f1,
                logger=True,
                sync_dist=sync_state,
                on_step=True,
                on_epoch=(stage != "train"),
            )
        if stage == "test":
            self.log(
                "test_oa",
                oa,
                logger=True,
                sync_dist=sync_state,
                on_epoch=True,
            )

        return loss

    def training_step(self, batch, batch_idx):
        inputs, targets, masks = batch
        outputs = self(inputs)  # Forward pass #[batch_size, n_classes, height, width]

        return self.compute_loss_and_metrics(outputs, targets, masks, stage="train")

    def validation_step(self, batch, batch_idx):
        inputs, targets, masks = batch
        outputs = self(inputs)  # Forward pass

        return self.compute_loss_and_metrics(outputs, targets, masks, stage="val")

    def on_validation_epoch_end(self):
        # Get the current validation metric (e.g., 'val_r2')
        # Concatenate all predictions and true labels
        sys_r2 = self.val_r2.compute()
        sys_f1 = self.val_f1.compute()
        self.log("ave_val_r2", sys_r2, sync_dist=True)

        print(f"average r2 score at epoch {self.current_epoch}: {sys_r2}")

        # Determine if current epoch has the best validation metric
        is_best = False
        if self.best_val_metric is None or sys_r2 > self.best_val_metric:
            is_best = True
            self.best_val_metric = sys_r2

        if is_best:
            print(f"F1 Score:{sys_f1}")
        # Clear buffers for the next epoch
        self.val_r2.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        inputs, targets, masks = batch
        outputs = self(inputs)  # Forward pass

        return self.compute_loss_and_metrics(outputs, targets, masks, stage="test")

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
