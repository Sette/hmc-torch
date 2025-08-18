import torch.nn as nn


class BaselineFFNNModel(nn.Module):
    """HMCNN(h) model - baseline model without constraints"""

    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R):
        super(BaselineFFNNModel, self).__init__()

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
        return x
