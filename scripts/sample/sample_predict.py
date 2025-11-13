# %%
import json
import os
from datetime import UTC
from datetime import datetime as dt

import torch

from hmc.models import ClassificationModel

base_path = ""
model_path = ""
sample_id = "rock_electronic"

train_path = os.path.join(base_path, sample_id)
torch_path = os.path.join(train_path, "torch")
metadata_path = os.path.join(train_path, "metadata.json")
labels_path = os.path.join(train_path, "labels.json")
testset_path = os.path.join(torch_path, "test")

with open(metadata_path, "r") as f:
    metadata = json.loads(f.read())
    print(metadata)

with open(labels_path, "r") as f:
    labels = json.loads(f.read())


params = {
    "levels_size": labels["levels_size"],
    "sequence_size": metadata["sequence_size"],
    "dropouts": [0.3, 0.7, 0.7, 0.3],
}

model = ClassificationModel(**params)

# Carregar o state_dict
state_dict = torch.load(model_path)

# Imprimir as chaves do state_dict
print("Chaves do state_dict salvo:")
print(state_dict.keys())

# Imprimir as chaves do modelo
print("Chaves do modelo definido:")
print(model.state_dict().keys())

# model_load = torch.load(binary_model)
model.load_state_dict(state_dict)


# %%
# Salvar os argumentos atuais
time_start = dt.now(UTC)
print("[{}] Predict started at {}".format(id, time_start.strftime("%H:%M:%S")))
print(".......................................")

predict = model.predict(testset_path=testset_path, batch_size=64)


time_end = dt.now(UTC)
time_elapsed = time_end - time_start
print(".......................................")

string_format = "[{}] Predict finished at {} / elapsed time {}s"
print(
    string_format.format(
        id, time_end.strftime("%H:%M:%S"), time_elapsed.total_seconds()
    )
)
