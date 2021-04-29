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
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        # self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        # self.conv2 = nn.Conv2d(64, 64, 5, stride=2, padding=2)
        # self.conv3 = nn.Conv2d(64, 64, 5, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(64, 32, 3, stride=1, padding=1)

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

        self.linear = nn.Linear(2304, 1024)

    def forward(self, x):
        # assert x.max() == 255
        x = x / 255.0
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, device=self.device, dtype=torch.float32)

        x = x.transpose(1, 3)
        output = self.block(x)
        output = self.linear(output.flatten(1))

        return torch.relu(output)

