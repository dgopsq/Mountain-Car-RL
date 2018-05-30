import torch
import torch.nn as nn

class CustomNet(nn.Module):
    def __init__(self, n_input, n_output):
        super(CustomNet, self).__init__()

        hidden_n = 20

        self.hidden_layer_1 = nn.Linear(n_input, hidden_n)
        self.output_layer = nn.Linear(hidden_n, n_output)

    def forward(self, x):
        out = self.hidden_layer_1(x)
        out = self.output_layer(out)
        return out
