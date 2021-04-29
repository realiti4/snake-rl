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
            nn.Conv2d(3, 128, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, stride=1),
            nn.ReLU(),
        )

        # self.block2 = nn.Sequential(
        #     nn.Conv2d(3, 32, 8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 5, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, stride=2),
        #     nn.ReLU(),
        # )

        # self.linear2 = nn.Sequential(
        #     nn.Linear(2160, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 64),
        #     nn.ReLU(),
        # )

        self.linear = nn.Linear(2048, 512)

    def forward(self, x):
        x = x / 255.0
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, device=self.device, dtype=torch.float32)

        x = x.transpose(1, 3)
        output = self.block(x)
        output = self.linear(output.flatten(1))

        return torch.relu(output)

        print('de')