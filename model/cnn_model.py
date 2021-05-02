import numpy as np
import torch
import torch.nn as nn



class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, device, dropout=0.25):
        super(CNNModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device       

        self.block = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, stride=2),
            nn.ReLU(),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.ReLU(),
        )

        # self.linear = nn.Linear(1024, 1024)

        self.linear = nn.Sequential(
            nn.Linear(1600, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

    def forward(self, x):
        # assert x.max() == 255
        # x = x / 255.0
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, device=self.device, dtype=torch.float32)

        # x = x.transpose(1, 3)
        output = self.block(x)
        output = self.linear(output.flatten(1))

        return torch.relu(output)

