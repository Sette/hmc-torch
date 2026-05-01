"""
This module contains the ConstrainedModel and ConstrainedLightningModel classes.
"""

import os

import torch
from lightning import LightningModule
from sklearn.metrics import average_precision_score
from torch import nn

from hmc.models.global_classifier.constraint.utils import get_constr_out


class ConstrainedModel(nn.Module):
    """C-HMCNN(h) model - during training it returns the not-constrained
    output that is then passed to MCLoss"""

    def __init__(self, **kwargs):
        """
        Initialize the ConstrainedModel.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden dimension.
            output_dim: Output dimension.
            hyperparams: Hyperparameters.
            r_matrix: Constraint matrix.
            baseline_model: Whether to use baseline model.
        """
        super().__init__()

        self.input_dim = kwargs["input_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.output_dim = kwargs["output_dim"]
        self.hyperparams = kwargs["hyperparams"]
        self.r_matrix = kwargs["r_matrix"]

        self.baseline_model = kwargs.get("baseline_model", False)

        self.nb_layers = self.hyperparams["num_layers"]

        fc = []
        for i in range(self.nb_layers):
            if i == 0:
                fc.append(nn.Linear(self.input_dim, self.hidden_dim))
            elif i == self.nb_layers - 1:
                fc.append(nn.Linear(self.hidden_dim, self.output_dim))
            else:
                fc.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc = nn.ModuleList(fc)

        self.drop = nn.Dropout(self.hyperparams["dropout"])

        self.sigmoid = nn.Sigmoid()
        if self.hyperparams["non_lin"] == "tanh":
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()

    def forward(self, x):
        """Forward pass for the ConstrainedModel."""
        for i in range(self.nb_layers):
            if i == self.nb_layers - 1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        if self.baseline_model:
            output = x
        else:
            if self.training:
                output = x
            else:
                output = get_constr_out(x, self.r_matrix)
        return output


class ConstrainedLightningModel(LightningModule):
    """Constrained Lightning Model."""

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize the ConstrainedLightningModel.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden dimension.
            output_dim: Output dimension.
            hyperparams: Hyperparameters.
            r_matrix: Constraint matrix.
            to_eval: Indices to evaluate.
            lr: Learning rate.
            weight_decay: Weight decay.
            baseline_model: Whether to use baseline model.
        """
        super().__init__()
        self.input_dim = kwargs.pop("input_dim")
        self.hidden_dim = kwargs.pop("hidden_dim")
        self.output_dim = kwargs.pop("output_dim")
        self.hyperparams = kwargs.pop("hyperparams")
        self.r_matrix = kwargs.pop("r_matrix")
        self.to_eval = kwargs.pop("to_eval")

        # Usando .get() para valores que têm padrão (default)
        self.lr = kwargs.pop("lr", 1e-3)
        self.weight_decay = kwargs.pop("weight_decay", 0.0)
        self.baseline_model = kwargs.pop("baseline_model", False)

        configs = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "hyperparams": self.hyperparams,
            "r_matrix": self.r_matrix,
            "baseline_model": True,
        }

        self.model = ConstrainedModel(**configs)
        self.model = self.model.to(self.device)
        if not self.baseline_model:
            self.r_matrix = self.r_matrix.to(self.device)
        self.to_eval = self.to_eval.to(self.device)
        self.criterion = nn.BCELoss()
        self.val_outputs = []
        self.test_outputs = []

    def forward(self, *args, **kwargs):
        """
        Forward pass for the ConstrainedLightningModel.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = args[0]
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        """
        Start of training epoch.
        """
        self.model.train()

    def on_validation_epoch_start(self):
        """
        Start of validation epoch.
        """
        self.model.eval()
        self.val_outputs = []

    def on_test_epoch_start(self):
        """
        Start of test epoch.
        """
        self.model.eval()
        self.test_outputs = []

    def training_step(self, *args, **kwargs):
        """
        Training step.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Loss.
        """
        batch = args[0]
        batch_idx = args[1]

        print(f"Training step {batch_idx}")
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        output = self.model(x.float())
        if not self.baseline_model:
            output = get_constr_out(output, self.r_matrix)

            output = y * output.double()
            output = get_constr_out(output, self.r_matrix)
            output = (1 - y) * output.double() + y * output

        loss = self.criterion(output[:, self.to_eval], y[:, self.to_eval])
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, *args, **kwargs):
        """
        Validation step.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Loss.
        """
        batch = args[0]
        x, y = batch
        x = x.to(self.device)

        output = self.model(x.float())

        self.val_outputs.append({"output": output.cpu(), "y": y.cpu()})

    def test_step(self, *args, **kwargs):
        """
        Test step.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        batch = args[0]
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        output = self.model(x.float())

        self.test_outputs.append({"output": output.cpu(), "y": y.cpu()})

    def on_test_epoch_end(self):
        """
        Processa os resultados e salva em `lightning_logs`.
        """
        if not self.test_outputs:
            return  # Evita erro se não houver dados

        output = torch.cat([x["output"] for x in self.test_outputs], dim=0)
        y_test = torch.cat([x["y"] for x in self.test_outputs], dim=0)

        score = average_precision_score(
            y_test[:, self.to_eval], output.data[:, self.to_eval], average="micro"
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
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(f"{self.current_epoch},{score}\n")

    def on_validation_epoch_end(self):
        """
        Processa os resultados e salva em `lightning_logs`.
        """
        if not self.val_outputs:
            return  # Evita erro se não houver dados

        output = torch.cat([x["output"] for x in self.val_outputs], dim=0)
        y_val = torch.cat([x["y"] for x in self.val_outputs], dim=0)

        score = average_precision_score(
            y_val[:, self.to_eval].cpu(),
            output.data[:, self.to_eval].cpu(),
            average="micro",
        )
        self.log("val_score", score, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """
        Configure optimizers.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
