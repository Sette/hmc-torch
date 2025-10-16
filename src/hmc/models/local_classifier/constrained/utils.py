import torch


def precalculate_r_sub(r, level_class_indices, level, prev_level, device):
    """Calcula a submatriz de adjacência para a Constraint Layer."""

    # Índices globais de todas as classes do nível anterior
    idx_ancestrais_global = level_class_indices[prev_level]

    # Índices globais de todas as classes do nível atual
    idx_filhos_global = level_class_indices[level]

    # Converte para tensores longos para indexação
    idx_ancestrais_tensor = torch.as_tensor(
        idx_ancestrais_global, dtype=torch.long, device=device
    )
    idx_filhos_tensor = torch.as_tensor(
        idx_filhos_global, dtype=torch.long, device=device
    )

    # Extrai a submatriz de adjacência (N_anterior x N_atual)
    # R[A, D] -> Matriz de adjacência que mapeia ancestral A para descendente D.
    R_sub = r[idx_ancestrais_tensor][:, idx_filhos_tensor].float()

    # Garante que as colunas onde um filho não tem pai (apenas se for DAG) não causem problemas,
    # mas R_sub deve ser uma matriz binária (0 ou 1).
    return R_sub


def apply_hierarchical_constraint_vectorized_corrected(
    outputs: torch.Tensor,
    prev_level_outputs: torch.Tensor,
    R_sub: torch.Tensor,
) -> torch.Tensor:
    """
    Constraint Layer vetorizada para aplicar P[filho] <= min(P[pais]) em um batch.

    outputs: (batch, N_atual) - Predições do modelo para o nível ATUAL.
    prev_level_outputs: (batch, N_anterior) - Predições COERENTES do nível ANTERIOR.
    R_sub: (N_anterior, N_atual) - Matriz de Adjacência (Anterior -> Atual) COMPLETA.
    """

    # 1. Obter o tipo de dado da entrada (geralmente torch.float32)
    dtype = prev_level_outputs.dtype

    # 2. Projeção das Probabilidades Teto (Ceiling Probability)

    # Expande prev_level_outputs: (batch, N_anterior) -> (batch, N_anterior, 1)
    P_ancestral_expanded = prev_level_outputs.unsqueeze(2)

    # Cria um valor muito alto (infinito) com o MESMO DTYPE da entrada.
    # Usar float('inf') garante que o min() ignore esta posição.
    high_value = torch.full_like(P_ancestral_expanded, float("inf"), dtype=dtype)

    # A condição (R_sub > 0) retorna o tensor booleano CORRETO.
    R_sub_bool = R_sub > 0

    # Expande R_sub_bool: (N_anterior, N_atual) -> (1, N_anterior, N_atual) para broadcasting.
    R_sub_bool_expanded = R_sub_bool.unsqueeze(0)

    # P_mascarado: (batch, N_anterior, N_atual)
    # torch.where: Se R_sub_bool=True, usamos P_ancestral. Se False, usamos high_value.
    # Ambos os tensores de valor (P_ancestral_expanded e high_value) têm agora o mesmo dtype.
    P_mascarado = torch.where(R_sub_bool_expanded, P_ancestral_expanded, high_value)

    # 3. Calcule o TETO (minimo entre todos os pais)
    # ceiling_probs: (batch, N_atual)
    ceiling_probs, _ = torch.min(P_mascarado, dim=1)

    # 4. Aplique a Restrição MIN
    # P_coerente[filho] = min(P_original[filho], ceiling_probs)
    coherent_outputs = torch.min(outputs, ceiling_probs)

    return coherent_outputs


def get_constr_out_layer(
    outputs,
    labels,
    level,
    device,
    r,
    nodes_idx,
    local_nodes_reverse_idx,
):
    """
    outputs: tensor (batch, n_classes_nivel_atual)
    ...
    """

    # # Penalização de inconsistência: só se não for o primeiro nível!
    if level > 0:  # tem ancestral!
        prev_level = level - 1
        print("labels reverse dict:", local_nodes_reverse_idx[prev_level])
        print("labels:", labels)
        print("labels shape:", labels[0].shape)

        probs = []
        for example__out, example_in, example_in_prev in zip(
            outputs,
            labels[level],
            labels[prev_level],
        ):
            print("example_out:", example__out)
            print("example_int:", example_in)

            idx_ancestrais = (
                (example_in_prev == 1).nonzero(as_tuple=True)[0].to("cpu").tolist()
            )

            idx_filhos = (example_in == 1).nonzero(as_tuple=True)[0].to("cpu").tolist()

            example_probs = [
                value if idx in idx_filhos else 0
                for idx, value in enumerate(example__out.detach().cpu().numpy())
            ]

            idx_ancestrais_global = [
                nodes_idx.get(local_nodes_reverse_idx[prev_level][idx])
                for idx in idx_ancestrais
            ]

            idx_filhos_global = [
                nodes_idx.get(local_nodes_reverse_idx[level][idx]) for idx in idx_filhos
            ]

            print("idx_ancestrais_global:", idx_ancestrais_global)
            print("idx_filhos_global:", idx_filhos_global)

            idx_ancestrais = torch.as_tensor(
                idx_ancestrais_global,
                dtype=torch.long,
                device=device,
            )

            idx_filhos = torch.as_tensor(
                idx_filhos_global,
                dtype=torch.long,
                device=device,
            )

            # Matriz de relação entre níveis
            R_sub = r[idx_ancestrais][:, idx_filhos]  # [n_ancestrais, n_descendentes]

            outputs = apply_hierarchical_constraint_vectorized_corrected(
                outputs=example__out.unsqueeze(0),
                prev_level_outputs=example_in_prev.unsqueeze(0).float(),
                R_sub=R_sub,
            ).squeeze(0)

            # print("R_sub:", R_sub)

            # Probabilidades (ajuste se sua rede retorna logits)
            print("new example out:", example_probs)
            probs.append(example_probs)

        return probs


def get_constr_out_vectorized(
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


def get_constr_out_old(
    outputs,
    labels,
    level,
    device,
    r,
    nodes_idx,
    local_nodes_reverse_idx,
):
    """
    outputs: tensor (batch, n_cR_sublasses_nivel_atual)
    ...
    """

    # # Penalização de inconsistência: só se não for o primeiro nível!
    if level > 0:  # tem ancestral!
        prev_level = level - 1
        print("labels reverse dict:", local_nodes_reverse_idx[prev_level])
        print("labels:", labels)
        print("labels shape:", labels[0].shape)
        # Índices globais
        # idx_ancestrais = torch.as_tensor(
        #     level_class_indices[prev_level],
        #     dtype=torch.long,
        #     device=device,
        # )
        # idx_filhos = torch.as_tensor(
        #     level_class_indices[level],
        #     dtype=torch.long,
        #     device=device,
        # )
        probs = []
        for example__out, example_in in zip(
            outputs,
            labels[level],
        ):
            print("example_out:", example__out)
            print("example_int:", example_in)

            # idx_ancestrais = (
            #     (example_in_prev == 1).nonzero(as_tuple=True)[0].to("cpu").tolist()
            # )

            idx_filhos = (example_in == 1).nonzero(as_tuple=True)[0].to("cpu").tolist()

            example_probs = [
                value if idx in idx_filhos else 0
                for idx, value in enumerate(example__out.detach().cpu().numpy())
            ]

            # idx_ancestrais_global = [
            #     nodes_idx.get(local_nodes_reverse_idx[prev_level][idx])
            #     for idx in idx_ancestrais
            # ]

            # idx_filhos_global = [
            #     nodes_idx.get(local_nodes_reverse_idx[level][idx]) for idx in idx_filhos
            # ]

            # print("idx_ancestrais_global:", idx_ancestrais_global)
            # print("idx_filhos_global:", idx_filhos_global)

            # idx_ancestrais = torch.as_tensor(
            #     idx_ancestrais_global,
            #     dtype=torch.long,
            #     device=device,
            # )

            # idx_filhos = torch.as_tensor(
            #     idx_filhos_global,
            #     dtype=torch.long,
            #     device=device,
            # )

            # # Matriz de relação entre níveis
            # R_sub = r[idx_ancestrais][:, idx_filhos]  # [n_ancestrais, n_descendentes]

            # print("R_sub:", R_sub)

            # Probabilidades (ajuste se sua rede retorna logits)
            print("new example out:", example_probs)
            probs.append(example_probs)

        return probs
