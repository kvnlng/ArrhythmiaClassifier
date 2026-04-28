import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-Label Classification to handle class imbalance.
    Math: Loss = - (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # We use BCEWithLogitsLoss internally but we tell it to output raw losses per element
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        # Calculate raw BCE Loss per element
        bce_loss = self.bce(logits, targets)

        # We need the predicted probabilities to calculate the modulating factor
        probs = torch.sigmoid(logits)

        # p_t is the probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal Loss modulating factor: (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Alpha weighting for class imbalance
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Final Focal Loss
        focal_loss = alpha_t * modulating_factor * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
