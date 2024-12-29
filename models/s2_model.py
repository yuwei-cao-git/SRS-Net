import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms.v2 as transforms

from .blocks import MF
from .unet import UNet
from .ResUnet import ResUnet

from torchmetrics import MeanSquaredError
from torchmetrics.regression import R2Score
from torchmetrics.functional import r2_score
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassAccuracy


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
        self.aug = self.config["transforms"]
        if self.config["resolution"] == 10:
            self.n_bands = 12
        else:
            self.n_bands = 9

        if self.use_mf:
            # MF Module for seasonal fusion (each season has `n_bands` channels)
            self.mf_module = MF(
                mode="img", channels=self.n_bands, spatial_att=self.spatial_attention
            )
            total_input_channels = (
                64  # MF module outputs 64 channels after processing four seasons
            )
        else:
            total_input_channels = (
                self.n_bands * 4
            )  # If no MF module, concatenating all seasons directly

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
        if self.aug == "random":
            self.transform = transforms.RandomApply(
                torch.nn.ModuleList(
                    [
                        transforms.RandomCrop(size=(128, 128)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToDtype(torch.float32, scale=True),
                        transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                        transforms.RandomRotation(degrees=(0, 180)),
                        transforms.RandomAffine(
                            degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
                        ),
                    ]
                ),
                p=0.3,
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=(128, 128)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                    transforms.RandomRotation(degrees=(0, 180)),
                    transforms.RandomAffine(
                        degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
                    ),
                ]
            )

        # Loss function
        self.criterion = MeanSquaredError()

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

        # Optimizer and scheduler settings
        self.optimizer_type = self.config["optimizer"]
        self.learning_rate = self.config["learning_rate"]
        self.scheduler_type = self.config["scheduler"]

        # Containers for validation predictions and true labels
        self.val_preds = []
        self.true_labels = []
        self.best_test_outputs = None
        self.best_val_metric = None

    def forward(self, inputs):
        # Optionally pass inputs through MF module
        if self.aug is not None:
            inputs = self.transform(inputs)
        if self.use_mf:
            # Apply the MF module first to extract features from input
            fused_features = self.mf_module(inputs)
        else:
            # Concatenate all seasons directly if no MF module
            fused_features = torch.cat(inputs, dim=1)
        logits, _ = self.model(fused_features)
        return logits

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
        # outputs = F.softmax(outputs, dim=1)

        valid_outputs, valid_targets = self.apply_mask(
            outputs, targets, masks, multi_class=True
        )

        # Compute the masked loss
        loss = self.criterion(valid_outputs, valid_targets)
        if stage == "val":
            self.val_preds.append(valid_outputs)
            self.true_labels.append(valid_targets)
        # Round outputs to two decimal place
        valid_outputs = torch.round(valid_outputs, decimals=1)

        # Convert outputs and targets to leading class labels by taking argmax
        pred_labels = torch.argmax(outputs, dim=1)
        true_labels = torch.argmax(targets, dim=1)
        # Apply mask
        valid_preds, valid_true = self.apply_mask(
            pred_labels, true_labels, masks, multi_class=False
        )

        # Renormalize after rounding to ensure outputs sum to 1 #TODO: validate
        # rounded_outputs = rounded_outputs / rounded_outputs.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Calculate R² and F1 score for valid pixels
        if stage == "train":
            r2 = self.train_r2(valid_outputs.view(-1), valid_targets.view(-1))
            f1 = self.train_f1(valid_preds, valid_true)
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
        preds_all = torch.cat(self.val_preds)
        true_labels_all = torch.cat(self.true_labels)
        last_epoch_val_r2 = r2_score(
            torch.round(preds_all.flatten(), decimals=1), true_labels_all.flatten()
        )
        self.log("ave_val_r2", last_epoch_val_r2, sync_dist=True)
        self.log("sys_r2", sys_r2, sync_dist=True)

        print(f"average r2 score at epoch {self.current_epoch}: {last_epoch_val_r2}")

        # Determine if current epoch has the best validation metric
        is_best = False
        if self.best_val_metric is None or last_epoch_val_r2 > self.best_val_metric:
            is_best = True
            self.best_val_metric = last_epoch_val_r2

        if is_best:
            # Store the tensors without converting to NumPy arrays
            self.best_test_outputs = {
                "preds_all": preds_all,
                "true_labels_all": true_labels_all,
            }

        # Clear buffers for the next epoch
        self.val_preds.clear()
        self.true_labels.clear()
        self.val_r2.reset()

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
