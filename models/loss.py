import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.regression import WeightedMeanAbsolutePercentageError


def apply_mask(outputs, targets, mask, multi_class=True, keep_shp=False):
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
    class_dim = 1
    if keep_shp:
        # Set invalid outputs and targets to zero
        if not multi_class:
            expanded_mask = mask
        else:
            expanded_mask = mask.unsqueeze(class_dim).expand_as(outputs)
        outputs = outputs.clone()
        targets = targets.clone()
        outputs[expanded_mask] = 255
        targets[expanded_mask] = 255
        return outputs, targets
    else:
        if multi_class:
            permute_dims = None # Initialize to None
            expected_mask_shape = (outputs.size(0), outputs.size(2), outputs.size(3))
            # Permute to (B, H, W, C) for masking
            permute_dims = (0, 2, 3, 1)
            # Validate mask shape
            if mask.shape != expected_mask_shape:
                raise ValueError(f"Mask shape mismatch. Expected {expected_mask_shape}, got {mask.shape}")
            
            # Permute to put class dimension last: (B, H, W, C)
            outputs_permuted = outputs.permute(*permute_dims).contiguous()
            targets_permuted = targets.permute(*permute_dims).contiguous()
            
            # Apply mask to exclude invalid data points
            valid_outputs = outputs_permuted[~mask]
            valid_targets = targets_permuted[~mask]

            return valid_outputs, valid_targets
        else:
            expanded_mask = mask
            # Assuming mask applies element-wise or needs broadcasting correctly
            valid_outputs = outputs[~expanded_mask]
            valid_targets = targets[~expanded_mask]
            return valid_outputs, valid_targets


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        squared_errors = torch.square(y_pred - y_true)
        loss = torch.mean(squared_errors)
        return loss


# MSE loss
def calc_mse_loss(valid_outputs, valid_targets):
    mse = MSELoss()
    loss = mse(valid_outputs, valid_targets)

    return loss


class PinballLoss:
    def __init__(self, quantile=0.10, reduction="none"):
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction

    def __call__(self, output, target):
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1 - self.quantile) * (abs(error)[bigger_index])

        if self.reduction == "sum":
            loss = loss.sum()
        if self.reduction == "mean":
            loss = loss.mean()

        return loss


def calc_pinball_loss(y_pred, y_true):
    pinball_loss = PinballLoss(quantile=0.10, reduction="mean")
    loss = pinball_loss(y_pred, y_true)

    return loss


# Weighted MSE loss
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        squared_errors = torch.square(y_pred - y_true)
        weighted_squared_errors = squared_errors * self.weights
        loss = torch.mean(weighted_squared_errors)
        return loss


def calc_wmse_loss(valid_outputs, valid_targets, weights):
    weighted_mse = WeightedMSELoss(weights)
    loss = weighted_mse(valid_outputs, valid_targets)

    return loss


# Rooted Weighted loss
def calc_rwmse_loss(valid_outputs, valid_targets, weights):
    weighted_mse = WeightedMSELoss(weights)
    loss = weighted_mse(valid_outputs, valid_targets)

    return torch.sqrt(loss)


# kl loss
def weighted_kl_divergence(y_true, y_pred, weights):
    loss = torch.sum(
        weights * y_true * torch.log((y_true + 1e-8) / (y_pred + 1e-8)), dim=1
    )
    return torch.mean(loss)


# MAE loss
def calc_mae_loss(valid_outputs, valid_targets):
    loss = torch.sum(torch.abs(valid_outputs - valid_targets), dim=1)

    return torch.mean(loss)


# loss for leading species classification
def cal_leading_loss(y_true, y_pred, alpha_leading):
    correct = (y_pred.view(-1) == y_true.view(-1)).float()
    loss_pixel_leads = 1 - correct.mean()  # 1 - accuracy as pseudo-loss
    return loss_pixel_leads * alpha_leading


def mask_output(y_pred, y_true, mask, stage):
    valid_outputs, valid_targets = apply_mask(y_pred, y_true, mask, multi_class=True)
    valid_outputs = torch.round(valid_outputs, decimals=1)

    # Convert outputs and targets to leading class labels by taking argmax
    pred_labels = torch.argmax(y_pred, dim=1)
    true_labels = torch.argmax(y_true, dim=1)

    # Apply mask to leading species labels
    valid_preds, valid_true = apply_mask(
        pred_labels, true_labels, mask, multi_class=False
    )
    targets = {
        f"{stage}_Classification": valid_true,
        f"{stage}_Regression": valid_targets.view(-1),
    }
    preds = {
        f"{stage}_Classification": valid_preds,
        f"{stage}_Regression": valid_outputs.view(-1),
    }

    return preds, targets


def calc_loss(loss_func_name, y_pred, y_true, mask, weights):
    valid_outputs, valid_targets = apply_mask(y_pred, y_true, mask, multi_class=True)
    if loss_func_name == "wmse":
        return calc_wmse_loss(valid_outputs, valid_targets, weights)
    elif loss_func_name == "wrmse":
        return calc_rwmse_loss(valid_outputs, valid_targets, weights)
    elif loss_func_name == "mse":
        return calc_mse_loss(valid_outputs, valid_targets)
    elif loss_func_name == "wkl":
        return weighted_kl_divergence(valid_targets, valid_outputs, weights)
    elif loss_func_name == "mae":
        return calc_mae_loss(valid_outputs, valid_targets)
    elif loss_func_name == "pinball":
        return calc_pinball_loss(valid_outputs, valid_targets)


def calc_masked_loss(loss_func_name, valid_outputs, valid_targets, weights):
    if loss_func_name == "wmse":
        return calc_wmse_loss(valid_outputs, valid_targets, weights)
    elif loss_func_name == "wrmse":
        return calc_rwmse_loss(valid_outputs, valid_targets, weights)
    elif loss_func_name == "mse":
        return calc_mse_loss(valid_outputs, valid_targets)
    elif loss_func_name == "wkl":
        return weighted_kl_divergence(valid_targets, valid_outputs, weights)
    elif loss_func_name == "mae":
        return calc_mae_loss(valid_outputs, valid_targets)
    elif loss_func_name == "pinball":
        return calc_pinball_loss(valid_outputs, valid_targets)


def focal_loss_multiclass(inputs, targets, alpha=0.25, gamma=2, ignore_index=255):
    """
    Multi-class focal loss implementation
    - inputs: raw logits from the model
    - targets: true class labels (as integer indices, not one-hot encoded)
    """

    # Gather the probabilities corresponding to the correct classes
    targets = targets * (targets != ignore_index).long()
    targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1])

    # Convert logits to log probabilities
    log_prob = torch.gather(inputs, 1, targets.unsqueeze(1))
    prob = torch.exp(log_prob)  # Calculate probabilities from log probabilities
    pt = torch.sum(prob * targets_one_hot, dim=-1)

    # Apply focal adjustment
    focal_loss = (
        -alpha * (1 - pt) ** gamma * torch.sum(log_prob * targets_one_hot, dim=-1)
    )

    return focal_loss.mean()
