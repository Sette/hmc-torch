import torch

def get_constr_out_vectorized_hierarchical(
    outputs: torch.Tensor,
    labels: list[torch.Tensor],
    level: int,
    device: torch.device,
    epsilon: float = 0.1  # fator de suavização (label smoothing)
) -> torch.Tensor:
    if level > 0:
        current_labels = labels[level].float().to(outputs.device)
        
        # Label smoothing hierárquico
        # Redistribui epsilon da massa apenas para classes válidas no nível
        valid_classes_per_sample = current_labels.sum(dim=1, keepdim=True)
        
        # Criar distribuição suavizada
        smooth_labels = current_labels * (1 - epsilon) + \
            (current_labels * epsilon / valid_classes_per_sample)
        
        probs = outputs * smooth_labels
        
        return probs.to(device)
    
    return outputs.to(device)


def get_constr_out_vectorized(
    outputs: torch.Tensor,
    labels: list[torch.Tensor],
    level: int,
    device: torch.device,
    penalty_strength: float = 5.0  # controla a força da penalização
) -> torch.Tensor:
    if level > 0:
        current_labels = labels[level].float().to(outputs.device)
        
        # Penalização exponencial suave
        # Onde label=1: mantém o valor
        # Onde label=0: aplica exp(-penalty_strength) que é próximo a 0 mas não exatamente 0
        penalty_tensor = torch.tensor(penalty_strength, device=outputs.device)
        penalty_mask = current_labels + (1 - current_labels) * torch.exp(-penalty_tensor)
        probs = outputs * penalty_mask
        
        return probs.to(device)
    
    return outputs.to(device)

def get_constr_out_vectorized_hard(
    outputs: torch.Tensor,
    labels: list[
        torch.Tensor
    ],  # Deve ser uma lista de tensors, onde labels[level] é o tensor do nível atual
    level: int,
    device: torch.device,
) -> torch.Tensor:
    """
    outputs: tensor (batch, n_classes_nivel_atual) - As probabilidades/logits do modelo.
    labels: list of tensors - A lista de tensors ground truth por nível.
    level: int - O nível atual.
    device: torch.device - O dispositivo de execução ('cuda' ou 'cpu').

    Retorna um tensor (batch, n_classes_nivel_atual) com os valores de
    outputs mantidos onde labels[level] == 1 e zerados onde labels[level] == 0.
    """
    # Penalização de inconsistência: só se não for o primeiro nível!
    if level > 0:  # tem ancestral!

        # 1. Obter o tensor de rótulos do nível atual (ground truth)
        # O tensor de rótulos já está no formato (batch, n_classes_nivel_atual)
        current_labels = (
            labels[level].float().to(outputs.device)
        )  # Garante que seja float para multiplicação
        # 2. Produto Elemento a Elemento (Vectorização)
        # A multiplicação * elemento a elemento age como uma máscara:
        # Se label é 1, prob * 1 = prob (mantém).
        # Se label é 0, prob * 0 = 0 (zera).
        # Não é necessário usar .detach().cpu().numpy()
        probs = outputs * current_labels

        # Nota: O tensor 'outputs' já deve estar no 'device' e o 'current_labels'
        # é movido para o mesmo dispositivo (outputs.device) para a multiplicação.
        return probs.to(device)

    # Se level == 0 (primeiro nível), retorna outputs inalterado
    return outputs.to(device)


def _calculate_local_loss(output, target, criterion, regularization=None, level=0):

    if regularization == "soft":
        output = get_constr_out_vectorized(
            output,
            target,
            level=level,
            device=output.device,
        )
    loss = criterion(output.double(), target)

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
        regularization=args.model_regularization,
        level=args.current_level,
        )

    return loss
