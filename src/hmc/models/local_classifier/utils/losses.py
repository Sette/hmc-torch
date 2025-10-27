import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCELoss(nn.Module):
    def __init__(self):
        super(MaskedBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(
            reduction="none"
        )  # Redução 'none' para manter a forma do tensor

    def forward(self, outputs, targets):
        losses = []
        for output, target in zip(outputs, targets):
            if len(target.shape) > 1:
                mask = target.sum(dim=1) > 0  # Dimensão 1 para targets 2D
            else:
                mask = target.sum() > 0  # Targets 1D ou outros casos

            if mask.any():
                loss = self.bce_loss(output, target)  # Calcula a perda sem redução
                masked_loss = loss[mask]  # Aplica a máscara
                losses.append(masked_loss.mean())  # Calcula a média da perda mascarada

        if len(losses) > 0:
            return torch.stack(losses).mean()  # Retorna um tensor e calcula a média
        else:
            # Retorna uma perda zero se não houver perdas
            return torch.tensor(0.0, requires_grad=True).to(outputs[0].device)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return F_loss.mean()
        elif self.reduction == "sum":
            return F_loss.sum()
        else:
            return F_loss
