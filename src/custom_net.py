import torch
import torch.nn as nn

class CustomNet(nn.Module):
    def __init__(self, n_input, n_output):
        super(CustomNet, self).__init__()
        self.hidden_layer_1 = nn.Linear(n_input, n_output * 2)
        self.output_layer = nn.Linear(n_output * 2, n_output)

    def forward(self, x):
        out = self.hidden_layer_1(x)
        out = self.output_layer(out)
        return out
