import torch



def calculate_local_loss(output, target, criterion):

    loss = criterion(output.double(), target)

    return loss
