from hmc.data.arxiv_hierarchy_manager import ArXivHierarchyManager
from hmc.data.arxiv_pytorch_dataset import ArXivPyTorchDataset

# 1. Preparar a Taxonomia
manager = ArXivHierarchyManager()
manager.fit_from_jsonl("arxiv_downloaded_subset.jsonl")

# Extraindo tensores de estrutura para o seu modelo PyTorch:
matriz_adjacencia = torch.from_numpy(manager.a).cuda()
edges_level_1 = torch.from_numpy(manager.edge_index[1]).cuda()

# 2. Inicializar Dataset
dataset = ArXivPyTorchDataset(
    jsonl_path="arxiv_downloaded_subset.jsonl",
    hierarchy_manager=manager,
    tokenizer=tokenizer,
)
