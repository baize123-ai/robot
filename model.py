import torch.nn as nn
import torch

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128,128)):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)
