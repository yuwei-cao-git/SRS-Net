import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        squared_errors = torch.square(y_pred - y_true)
        weighted_squared_errors = squared_errors * self.weights
        loss = torch.mean(weighted_squared_errors)
        return loss


def calc_loss(y_true, y_pred, weights):
    weighted_mse = WeightedMSELoss(weights)
    loss = weighted_mse(y_pred, y_true)

    return loss


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, output, target, nodata_mask):
        """
        Computes per-pixel loss, masking out the nodata regions.
        """
        # output: Shape (batch_size, num_classes, height, width)
        # target: Shape (batch_size, num_classes, height, width)
        # nodata_mask: Shape (batch_size, height, width)

        # Apply the nodata mask to ignore invalid pixels
        valid_mask = ~nodata_mask.unsqueeze(1)  # Shape: (batch_size, 1, height, width)
        valid_mask = valid_mask.expand_as(
            output
        )  # Shape: (batch_size, num_classes, height, width)

        # Compute loss per pixel
        loss = F.mse_loss(
            output * valid_mask.float(), target * valid_mask.float(), reduction="sum"
        )

        # Compute the average loss over valid pixels
        num_valid_pixels = valid_mask.float().sum()
        if num_valid_pixels > 0:
            loss = loss / num_valid_pixels
        else:
            loss = torch.tensor(0.0, requires_grad=True).to(output.device)

        return loss


# create a nn class (just-for-fun choice :-)
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = MaskedMSELoss()

    def forward(self, outputs, targets, mask):
        return torch.sqrt(self.mse(outputs, targets, mask))


def weighted_categorical_crossentropy(y_true, y_pred, weights):
    # y_true and y_pred are tensors of shape (batch_size, num_classes)
    # weights is a tensor of shape (num_classes,)
    loss = -torch.sum(weights * y_true * torch.log(y_pred + 1e-8), dim=1)
    return torch.mean(loss)


def weighted_kl_divergence(y_true, y_pred, weights):
    loss = torch.sum(
        weights * y_true * torch.log((y_true + 1e-8) / (y_pred + 1e-8)), dim=1
    )
    return torch.mean(loss)


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


# contrastive loss
def aggregate_to_superpixels(
    pred_pixel_labels, img_masks, fusion_preds, lambda_contrastive
):
    # Aggregate per-pixel predictions
    valid_pixel_preds, _ = apply_mask(
        pred_pixel_labels,
        pred_pixel_labels,
        img_masks,
        multi_class=False,
        keep_shp=True,
    )
    aggregated_pixel_preds = valid_pixel_preds.mean(dim=[2, 3])  # Shape: (N, C)

    # Compute contrastive loss
    contrastive_loss = F.mse_loss(fusion_preds, aggregated_pixel_preds)

    # Total loss
    lambda_contrastive = lambda_contrastive

    return lambda_contrastive * contrastive_loss
