import torch
import torch.nn as nn


class RelCellTypeMSELoss(nn.Module):
    """
        MSE Loss for self supervision task.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, y_pred, y_true):
        y_true = y_true.view(-1, y_true.shape[-1])
        mask = torch.sum(y_true, dim=-1) > 0
        y_true_red = y_true[mask]
        y_pred_red = y_pred[mask]
        mask = mask.float()
        factor = torch.where(
            torch.sum(mask) == 0.,
            torch.tensor(0.),
            torch.tensor(mask.size(0), dtype=torch.float32) / torch.sum(mask)
        )
        return self.loss_fn(y_true_red, y_pred_red) * factor


class CustomCategoricalCrossEntropyLoss(nn.Module):
    """
        Cross Entropy Loss is scaled up to batch size based on non-observed labels.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, y_pred, y_true):

        # print(y_pred)
        # print(y_true)
        # Define a mask to set the loss for non-observed labels to zero
        mask = torch.sum(y_true, dim=-1) > 0

        y_true_red = y_true[mask]
        y_pred_red = y_pred[mask]
        if y_true_red.size(0) == 0:
            factor = torch.tensor(1.)
        else:
            factor = torch.tensor(y_true.size(0) / y_true_red.size(0), dtype=torch.float32)
        
        # Calculate the cross-entropy loss
        return self.loss_fn(y_true_red, y_pred_red) * factor


class CustomMSELoss(nn.Module):
    """
        MSE Loss is scaled up to batch size based on non-observed labels.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, y_pred, y_true):
        mask = ~torch.isnan(y_true)
        factor = torch.where(
            torch.sum(mask) == 0,
            torch.tensor(0.),
            torch.tensor(mask.numel(), dtype=torch.float32) / torch.sum(mask)
        )
        y_true = torch.where(torch.isnan(y_true), torch.ones_like(y_true), y_true)
        return self.loss_fn(y_true * mask, y_pred * mask) * factor



class CustomCategoricalCrossEntropyNodesLoss(nn.Module):
    """
        Cross entropy for node level supervision.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_true = y_true.view(-1, y_true.size(-1))
        mask = torch.sum(y_true, dim=-1) > 0  # Select nodes that have a target label in one hot encoding.
        y_true_red = y_true[mask]
        y_pred_red = y_pred[mask]
        mask = mask.float()
        eps = 1e-8
        cce = torch.where(  # Scale loss to mean across all nodes in batch.
            torch.sum(mask) == 0,
            torch.tensor(1.),
            torch.tensor(1. / torch.sum(mask), dtype=torch.float32) *
                         torch.mean(- y_true_red * torch.log(y_pred_red + eps) - (1 - y_true_red) * torch.log(1 - y_pred_red + eps))
        )
        return cce
