import torch


def _calculate_local_loss(output, target, criterion):

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

    loss = _calculate_local_loss(output, target, args.criterions[args.current_level])

    return loss
