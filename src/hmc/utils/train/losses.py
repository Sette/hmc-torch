"""
Utility functions for loss calculation.
"""

import torch
import torch.nn.functional as F
from torch import nn


def calculate_local_loss(output, target, criterion, device="cpu"):
    """
    Calculates the local loss using a specific criterion based on the
        current computation level.

    Args:
        output (torch.Tensor): The output tensor from the model.
        target (torch.Tensor): The ground truth tensor.
        args (Namespace): An object containing the current computation
        level and the criterions.
    Returns:
        torch.Tensor: The calculated loss.
    """

    loss = criterion(output.to(device).double(), target.to(device))
    return loss


def compute_loss(batch, args, step="train"):
    """
    Computes the loss for a given batch.

    Args:
        batch (tuple): A tuple containing the input data and target labels.
        args (Namespace): An object containing the model, dataset, and other
        configuration.
        step (str): The current step (train, val, or test).
    Returns:
        tuple: A tuple containing the local losses, local inputs,
        local outputs, and total loss.
    """
    x = batch[0].float().to(args.device)
    targets = batch[1]

    # Supondo que outputs seja o dict {str(level): logits}
    outputs = args.model(x)

    local_losses = [0.0 for _ in range(args.hmc_dataset.max_depth)]
    local_inputs = {level: torch.tensor([]) for level in args.active_levels}
    local_outputs = {level: torch.tensor([]) for level in args.active_levels}

    total_cls_loss = 0.0

    # 1. Cálculo da Loss de Classificação (BCE) por nível
    for level in args.active_levels:
        if args.level_active[level]:
            loss = calculate_local_loss(
                outputs[str(level)],
                targets[level],
                args.criterion_list[level],
            )
            local_losses[level] += loss.item()
            total_cls_loss += loss

            # Armazenamento para métricas/validação
            if local_outputs[level].shape[0] == 0:
                local_outputs[level] = outputs[
                    str(level)
                ].detach()  # detach para evitar acúmulo de grafo
                local_inputs[level] = targets[level]
            else:
                local_outputs[level] = torch.cat(
                    (local_outputs[level], outputs[str(level)].detach()), dim=0
                )
                local_inputs[level] = torch.cat(
                    (local_inputs[level], targets[level]), dim=0
                )

    # 2. Cálculo da Hierarchical Consistency Loss (HCL)
    # Certifique-se que args.hierarchy_map está no formato
    # {(pai_lvl, pai_idx): [(filho_lvl, filho_idx)]}
    loss_hier = 0.0
    if hasattr(args.hmc_dataset, "hierarchy_map") and args.hmc_dataset.hierarchy_map:
        loss_hier = hierarchical_consistency_loss(
            outputs, args.hmc_dataset.hierarchy_map
        )

    # 3. Loss Total
    # O lambda_hier (ex: 0.1) controla o impacto da consistência
    lambda_hier = getattr(args, "lambda_hier", 0.5)
    total_loss = total_cls_loss + (lambda_hier * loss_hier)

    if step == "train":
        # Zera todos os otimizadores antes do backward
        for opt in args.optimizers:
            opt.zero_grad()

        total_loss.backward()

        # Step em todos os otimizadores ativos
        for level, optimizer in enumerate(args.optimizers):
            if args.level_active[level]:
                optimizer.step()

    return local_losses, local_inputs, local_outputs


class MaskedBCELoss(nn.Module):
    """
    Masked Binary Cross-Entropy Loss
    """

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss(
            reduction="none"
        )  # Redução 'none' para manter a forma do tensor

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Tensores de logits (sem sigmoid aplicada) com shape
                (Batch, Num_Classes)
            targets: Tensores float (0 ou 1) com shape (Batch, Num_Classes)
        Returns:
            torch.Tensor: Masked Binary Cross-Entropy Loss
        """
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
            loss = torch.stack(losses).mean()  # Retorna um tensor e calcula a média
        else:
            # Retorna uma perda zero se não houver perdas
            loss = torch.tensor(0.0, requires_grad=True).to(outputs[0].device)
        return loss


class MultiLabelFocalLoss(nn.Module):
    """
    Multi-Label Focal Loss
    """

    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        """
        Args:
            alpha (float): Fator de balanceamento para a classe positiva
                (0 < alpha < 1).
            Geralmente alpha=0.25 funciona bem para
            reduzir o peso dos negativos (fundo).
            gamma (float): Fator de focagem. Reduz a loss para exemplos fáceis.
            reduction (str): 'mean', 'sum' ou 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: Tensores de logits (sem sigmoid aplicada) com shape
            (Batch, Num_Classes)
        targets: Tensores float (0 ou 1) com shape (Batch, Num_Classes)
        """

        # 1. Calcular BCE com logits (Mais estável numericamente que sigmoid + log)
        # reduction='none' mantém o shape (Batch, Num_Classes)
        #  para podermos aplicar os pesos element-wise
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # 2. Calcular pt (probabilidade da classe verdadeira)
        # Se target=1, pt = p. Se target=0, pt = 1-p.
        # Matematicamente: BCE = -log(pt), logo pt = exp(-BCE)
        pt = torch.exp(-bce_loss)

        # 3. Calcular termo de Focal: (1 - pt)^gamma * BCE
        focal_term = (1 - pt) ** self.gamma * bce_loss

        # 4. Aplicar Alpha ponderado
        # Se alpha for definido, aplica alpha para targets=1 e (1-alpha) para
        # targets=0
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_term
        else:
            loss = focal_term

        # 5. Redução final
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class WeightedMultiLabelFocalLoss(nn.Module):
    """
    Weighted Multi-Label Focal Loss
    """

    def __init__(self, alpha=None, gamma=2, pos_weight=None, reduction="mean"):
        """
        Args:
            alpha (float, optional): Fator de balanceamento global. Se pos_weight for usado,
                alpha pode ser None ou usado para ajuste fino.
            gamma (float): Fator de focagem (padrão 2).
            pos_weight (Tensor, optional): Tensor de pesos para a classe positiva
                (shape: [num_classes]).
                Geralmente calculado como (num_negativos / num_positivos).
            reduction (str): 'mean', 'sum' ou 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensores de logits (sem sigmoid aplicada) com shape (Batch, Num_Classes)
            targets: Tensores float (0 ou 1) com shape (Batch, Num_Classes)
        Returns:
            torch.Tensor: Weighted Multi-Label Focal Loss
        """
        # Garante que inputs e targets tenham o mesmo device e tipo
        if self.pos_weight is not None:
            # Move pos_weight para o mesmo device dos inputs se necessário
            self.pos_weight = self.pos_weight.to(inputs.device)

        # 1. Calcular BCE Loss "crua" (sem redução) para obter a base logarítmica
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # 2. Calcular probabilidade pt (probabilidade da classe verdadeira)
        # pt = exp(-BCE) é numericamente estável
        pt = torch.exp(-bce_loss)

        # 3. Calcular termo Focal: (1 - pt)^gamma * BCE
        focal_loss = (1 - pt) ** self.gamma * bce_loss

        # 4. Aplicação dos Pesos (Alpha ou Pos_Weight)

        if self.pos_weight is not None:
            # Cria uma matriz de pesos onde:
            # Se target == 1: usa pos_weight daquela classe
            # Se target == 0: usa 1.0 (peso padrão para negativos)
            # Isso foca agressivamente em recuperar os positivos raros da Gene Ontology
            weights = targets * self.pos_weight + (1 - targets)
            focal_loss = focal_loss * weights

        elif self.alpha is not None:
            # Fallback para o alpha escalar simples se pos_weight não for fornecido
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = focal_loss * alpha_t

        # 5. Redução
        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()

        return focal_loss


def hierarchical_consistency_loss(logits_dict, class_hierarchy):
    """
    Calculates the hierarchical consistency loss.
    This loss penalizes the model for predicting child classes with higher
        probabilities than their parent classes.
    Args:
        logits_dict (dict): Dictionary mapping each level index to a
            tensor of logits.
        class_hierarchy (dict): Dictionary mapping each parent
            class index to a list of child class indices.
    Returns:
        torch.Tensor: The calculated hierarchical consistency loss.
    """
    h_loss = 0
    probs = {lvl: torch.sigmoid(value) for lvl, value in logits_dict.items()}

    for (p_lvl, p_idx), children in class_hierarchy.items():
        # Probabilidade da classe pai específica
        # probs[p_lvl] tem shape (batch, num_classes_no_nivel)
        prob_pai = probs[str(p_lvl)][:, p_idx]

        for c_lvl, c_idx in children:
            prob_filho = probs[str(c_lvl)][:, c_idx]

            # Penaliza se P(Filho) > P(Pai)
            violation = torch.relu(prob_filho - prob_pai)
            h_loss += torch.mean(violation)

    return h_loss
