import torch
import os
import torch.nn as nn
from lightning import LightningModule

from sklearn.metrics import average_precision_score


def get_constr_out(x, R):
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


class ConstrainedFFNNModel(nn.Module):
    """C-HMCNN(h) model - during training it returns the not-constrained
    output that is then passed to MCLoss"""

    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R):
        super(ConstrainedFFNNModel, self).__init__()

        self.nb_layers = hyperparams["num_layers"]
        self.R = R

        fc = []
        for i in range(self.nb_layers):
            if i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers - 1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)

        self.drop = nn.Dropout(hyperparams["dropout"])

        self.sigmoid = nn.Sigmoid()
        if hyperparams["non_lin"] == "tanh":
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()

    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers - 1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out


class ConstrainedFFNNModelPL(LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hyperparams,
        R,
        to_eval,
        lr,
        weight_decay,
    ):
        super().__init__()
        self.model = ConstrainedFFNNModel(
            input_dim, hidden_dim, output_dim, hyperparams, R
        )
        self.model = self.model.to(self.device)
        self.R = R
        self.to_eval = to_eval.to(self.device)
        self.criterion = nn.BCELoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        self.model.train()

    def on_validation_epoch_start(self):
        """Limpa a lista antes de cada época de teste."""
        self.model.eval()
        self.val_outputs = []

    def on_test_epoch_start(self):
        """Limpa a lista antes de cada época de teste."""
        self.model.eval()
        self.test_outputs = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        output = self.model(x.float())
        constr_output = get_constr_out(output, self.R)

        train_output = y * output.double()
        train_output = get_constr_out(train_output, self.R)
        train_output = (1 - y) * constr_output.double() + y * train_output

        loss = self.criterion(train_output[:, self.to_eval], y[:, self.to_eval])
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)

        constrained_output = self.model(x.float())

        self.val_outputs.append(
            {"constr_output": constrained_output.cpu(), "y": y.cpu()}
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        constrained_output = self.model(x.float())

        self.test_outputs.append(
            {"constr_output": constrained_output.cpu(), "y": y.cpu()}
        )

    def on_test_epoch_end(self):
        """Processa os resultados e salva em `lightning_logs`."""
        if not self.test_outputs:
            return  # Evita erro se não houver dados

        constr_test = torch.cat([x["constr_output"] for x in self.test_outputs], dim=0)
        y_test = torch.cat([x["y"] for x in self.test_outputs], dim=0)

        score = average_precision_score(
            y_test[:, self.to_eval], constr_test.data[:, self.to_eval], average="micro"
        )
        self.log("test_score", score, prog_bar=True, logger=True)

        # 📁 Obtém o diretório do Lightning Logs
        log_dir = (
            self.trainer.logger.log_dir if self.trainer.logger else "lightning_logs"
        )
        results_path = os.path.join(log_dir, "results.csv")

        # 🔥 Cria o diretório se não existir
        os.makedirs(log_dir, exist_ok=True)

        # Salva os resultados em `lightning_logs/results.csv`
        with open(results_path, "a") as f:
            f.write(f"{self.current_epoch},{score}\n")

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return  # Evita erro se não houver dados

        constr_val = torch.cat([x["constr_output"] for x in self.val_outputs], dim=0)
        y_val = torch.cat([x["y"] for x in self.val_outputs], dim=0)

        score = average_precision_score(
            y_val[:, self.to_eval].cpu(),
            constr_val.data[:, self.to_eval].cpu(),
            average="micro",
        )
        self.log("val_score", score, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
