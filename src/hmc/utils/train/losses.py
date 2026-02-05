import torch
import torch.nn as nn
import torch.nn.functional as F


def _calculate_local_loss(output, target, criterion, parent_conditioning=None, p_output=None, matrix_r=None):
    loss = criterion(output.double(), target)
    lambda_factor = 0.2

    if parent_conditioning != "none" and p_output is not None:
        device = p_output.device
        matrix_r_tensor = torch.from_numpy(matrix_r).float().to(device)

        parents_projected = torch.mm(p_output, matrix_r_tensor)

        diff = output - parents_projected

        if parent_conditioning == "soft" and p_output is not None:
            # Perda soft: diferença entre a saída atual e a saída do nível anterior
            loss += lambda_factor * torch.mean(diff ** 2)
        elif parent_conditioning == "teacher_forcing" and p_output is not None:
            # Perda de teacher forcing: força a saída atual a se aproximar da saída do nível anterior
            # ReLU: Só penaliza se a diferença for POSITIVA (Filho > Pai)
            # Valores negativos viram 0.
            penalty = torch.relu(diff)

            tf_loss = penalty.mean()
            loss += lambda_factor * tf_loss

    return loss


def calculate_local_loss(output, target, args):
    """
    Calculates the local loss using a specific criterion based on the current computation level.

    Args:
        output (torch.Tensor): The output tensor from the model.
        target (torch.Tensor): The ground truth tensor.
        args (Namespace): An object containing the current computation level and the criterions.
    Returns:
        torch.Tensor: The calculated loss.
    """

    loss = _calculate_local_loss(
        output,
        target,
        args.criterions[args.current_level],
        parent_conditioning=args.parent_conditioning,
    )

    return loss


def calculate_hierarchical_local_loss(output, target, p_output, matrix_r, args):
    """
    Calculates the local loss using a specific criterion based on the current computation level.

    Args:
        output (torch.Tensor): The output tensor from the model.
        target (torch.Tensor): The ground truth tensor.
        p_output (torch.Tensor): The output tensor from the previous level.
        args (Namespace): An object containing the current computation level and the criterions.
    Returns:
        torch.Tensor: The calculated loss.
    """

    loss = _calculate_local_loss(
        output,
        target,
        args.criterions[args.current_level],
        parent_conditioning=args.parent_conditioning,
        p_output=p_output,
        matrix_r=matrix_r,
    )

    return loss


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


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Args:
            alpha (float): Fator de balanceamento para a classe positiva (0 < alpha < 1).
                           Geralmente alpha=0.25 funciona bem para reduzir o peso dos negativos (fundo).
            gamma (float): Fator de focagem. Reduz a loss para exemplos fáceis.
            reduction (str): 'mean', 'sum' ou 'none'.
        """
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: Tensores de logits (sem sigmoid aplicada) com shape (Batch, Num_Classes)
        targets: Tensores float (0 ou 1) com shape (Batch, Num_Classes)
        """

        # 1. Calcular BCE com logits (Mais estável numericamente que sigmoid + log)
        # reduction='none' mantém o shape (Batch, Num_Classes) para podermos aplicar os pesos element-wise
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 2. Calcular pt (probabilidade da classe verdadeira)
        # Se target=1, pt = p. Se target=0, pt = 1-p.
        # Matematicamente: BCE = -log(pt), logo pt = exp(-BCE)
        pt = torch.exp(-bce_loss)

        # 3. Calcular termo de Focal: (1 - pt)^gamma * BCE
        focal_term = (1 - pt) ** self.gamma * bce_loss

        # 4. Aplicar Alpha ponderado
        # Se alpha for definido, aplica alpha para targets=1 e (1-alpha) para targets=0
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_term
        else:
            loss = focal_term

        # 5. Redução final
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedMultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, pos_weight=None, reduction='mean'):
        """
        Args:
            alpha (float, optional): Fator de balanceamento global. Se pos_weight for usado,
                                     alpha pode ser None ou usado para ajuste fino.
            gamma (float): Fator de focagem (padrão 2).
            pos_weight (Tensor, optional): Tensor de pesos para a classe positiva (shape: [num_classes]).
                                           Geralmente calculado como (num_negativos / num_positivos).
            reduction (str): 'mean', 'sum' ou 'none'.
        """
        super(WeightedMultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Garante que inputs e targets tenham o mesmo device e tipo
        if self.pos_weight is not None:
            # Move pos_weight para o mesmo device dos inputs se necessário
            self.pos_weight = self.pos_weight.to(inputs.device)

        # 1. Calcular BCE Loss "crua" (sem redução) para obter a base logarítmica
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

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
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
