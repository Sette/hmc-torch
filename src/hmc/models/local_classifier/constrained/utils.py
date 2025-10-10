import torch


def get_constr_out(
    outputs, probs_ancestrais, active_levels, level, device, level_class_indices, r
):
    """
    outputs: tensor (batch, n_classes_nivel_atual)
    probs_ancestrais: tensor (batch, n_classes_nivel_ancestral)
    ...
    """
    # # Penalização de inconsistência: só se não for o primeiro nível!
    if level > 0:  # tem ancestral!
        prev_level = active_levels[level - 1]
        # Índices globais
        idx_ancestrais = torch.as_tensor(
            level_class_indices[prev_level],
            dtype=torch.long,
            device=device,
        )
        idx_filhos = torch.as_tensor(
            level_class_indices[level],
            dtype=torch.long,
            device=device,
        )
        # Matriz de relação entre níveis
        R_sub = r[idx_ancestrais][:, idx_filhos]  # [n_ancestrais, n_descendentes]

        # Probabilidades (ajuste se sua rede retorna logits)

        probs_filhos = outputs

        # Calcula as restrições
        prob_ancestral = torch.matmul(probs_ancestrais, R_sub)  # (batch, n_filhos)
        outputs_constr = torch.min(probs_filhos, prob_ancestral)

        return outputs_constr


def get_constr_out_merge(outputs, active_levels, level, device, level_class_indices, r):
    """
    Applies hierarchical consistency constraints to the network outputs for a specific level.

    Args:
        outputs (torch.Tensor): Output tensor from the network for the current level (batch_size, n_current_level_classes).
        active_levels (list): List of active level indices.
        level (int): Current hierarchical level index.
        device (torch.device): Torch device to perform computations on.
        level_class_indices (dict): Mapping of level indices to their global class indices.
        r (torch.Tensor): Full hierarchical relationship matrix between all classes.

    Returns:
        torch.Tensor: Modified output tensor with hierarchical constraints applied (batch_size, n_current_level_classes).
    """
    # Apply consistency penalty only if this is not the first level
    if level > 0:  # has ancestor level
        prev_level = active_levels[level - 1]
        # Get global indices for ancestor and current (child) level classes
        idx_ancestrais = torch.as_tensor(
            level_class_indices[prev_level],
            dtype=torch.long,
            device=device,
        )
        idx_filhos = torch.as_tensor(
            level_class_indices[level],
            dtype=torch.long,
            device=device,
        )
        # Get the relevant submatrix of hierarchy relations between ancestor and child classes
        R = r.squeeze(0).to(device)
        R_sub = R[idx_ancestrais][:, idx_filhos]  # shape: [n_ancestors, n_children]

        # Convert outputs to double precision
        c_out = outputs.double()  # shape: [batch_size, n_children]
        # Unsqueeze to prepare for broadcasted multiplication:
        # Changes shape from [batch_size, n_children] to [batch_size, 1, n_children]
        c_out = c_out.unsqueeze(1)
        # Expand R_sub to batch dimension for broadcasting: shape: [1, n_ancestors, n_children]
        R_exp = R_sub.unsqueeze(0)
        # Compute the contribution of each ancestor to each child prediction:
        # shape: [batch_size, n_ancestors, n_children]
        prod = c_out * R_exp
        # For each child, select the largest value assigned by any ancestor:
        # Resulting shape: [batch_size, n_children]
        final_out, _ = torch.max(prod, dim=1)
        return final_out


def get_constr_out_old(x, R):
    """
    Given the network output x and a constraint matrix R,
    returns the modified output according to the hierarchical constraints in R.
    """
    # Convert x to double precision
    c_out = x.double()

    # Add a dimension to c_out: from (N, D) to (N, 1, D)
    # N: batch size, D: dimensionality of the output
    c_out = c_out.unsqueeze(1)

    # Expand c_out to match the shape of R:
    # If R is (C, C), c_out becomes (N, C, C)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])

    # Expand R similarly to (N, C, C)
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])

    # Element-wise multiplication of R_batch by c_out.
    # This produces a (N, C, C) tensor.
    # torch.max(...) is taken along dimension=2, resulting in (N, C).
    # This extracts the maximum along the last dimension,
    # effectively applying the hierarchical constraints.
    final_out, _ = torch.max(R_batch * c_out.double(), dim=2)

    return final_out
