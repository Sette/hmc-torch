import torch


def calculate_local_loss(outputs, targets, args, level):

    output = outputs[level]  # Preferencialmente float32
    target = targets[level]

    if args.model_regularization == "mask" and level != 0:
        child_indices = args.class_indices_per_level[level]  # [n_classes_nivel_atual]
        # MCLoss
        # Índices globais dos pais para cada amostra
        parent_target = targets[level - 1]
        parent_indices = args.class_indices_per_level[level - 1]
        parent_index_each_sample = parent_target.argmax(dim=1)
        parent_global_idxs = parent_indices[parent_index_each_sample]
        # Constrói a máscara usando R_global (shape: [1, n, n])
        mask = torch.stack(
            [
                args.R[0, child_indices, parent_global_idxs[b]]
                for b in range(output.shape[0])
            ],
            dim=0,
        )  # [batch, n_classes_nivel_atual]
        masked_output = output + (1 - mask) * (-1e9)
        loss = args.criterions[level](torch.sigmoid(masked_output), target)
    elif args.model_regularization == "soft" and level != 0:
        loss = args.criterions[level](output.double(), target)
        lambda_hier = 0.1
        global_dict = args.hmc_dataset.nodes_idx
        # classes_local_to_global: mapeia idx local para global correto
        local_dict = args.hmc_dataset.local_nodes_idx[level]
        classes_local_to_global = [
            int(global_dict[node.replace("/", ".")])
            for node, i in sorted(local_dict.items(), key=lambda x: x[1])
        ]
        # Constrói vetor de probabilidades globais "espalhado"
        probs_expanded = torch.zeros(
            (output.size(0), args.R.size(0)), device=output.device
        )
        probs_expanded[:, classes_local_to_global] = output
        # Penalidade global
        probs_i = probs_expanded.unsqueeze(2)  # (batch, n_total, 1)
        probs_j = probs_expanded.unsqueeze(1)  # (batch, 1, n_total)
        diff = probs_j - probs_i
        # Máscara hierárquica global (sem diagonal)
        R_mask = args.R - torch.eye(args.R.size(0), device=output.device)
        penalty = torch.clamp(diff * R_mask, min=0).sum() / (
            output.size(0) * R_mask.sum()
        )
        loss = loss + lambda_hier * penalty
    else:
        loss = args.criterions[level](output.double(), target)

    return loss
