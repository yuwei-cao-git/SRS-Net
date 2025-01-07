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
    if multi_class:
        expanded_mask = mask.unsqueeze(1).expand_as(
            outputs
        )  # Shape: (batch_size, num_classes, H, W)
        num_classes = outputs.size(1)
    else:
        expanded_mask = mask

    if keep_shp:
        # Set invalid outputs and targets to zero
        outputs = outputs.clone()
        targets = targets.clone()
        outputs[expanded_mask] = 0
        targets[expanded_mask] = 0
        return outputs, targets
    else:
        # Apply mask to exclude invalid data points
        valid_outputs = outputs[~expanded_mask]
        valid_targets = targets[~expanded_mask]
        # Reshape to (-1, num_classes)
        if multi_class:
            valid_outputs = valid_outputs.view(-1, num_classes)
            valid_targets = valid_targets.view(-1, num_classes)
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
    elif loss_func_name == "rwmse":
        return calc_rwmse_loss(valid_outputs, valid_targets, weights)
    elif loss_func_name == "mse":
        return calc_mse_loss(valid_outputs, valid_targets)
    elif loss_func_name == "kl":
        return weighted_kl_divergence(valid_targets, valid_outputs, weights)
    elif loss_func_name == "mae":
        return calc_mae_loss(valid_outputs, valid_targets)
