import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, n_input, n_output):
        super(Policy, self).__init__()

        hidden_n = 50

        self.hidden_layer = nn.Linear(n_input, hidden_n, bias = False)
        self.output_layer = nn.Linear(hidden_n, n_output, bias = False)

    def forward(self, x):
        out = self.hidden_layer(x)
        out = self.output_layer(out)
        return out
