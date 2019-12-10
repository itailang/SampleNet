from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetFeatures(nn.Module):
    def __init__(self, bottleneck_size=1024, input_shape="bcn"):
        super().__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        self.input_shape = input_shape

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, bottleneck_size, 1)

    def forward(self, x):
        # x shape should be B x 3 x N
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1)

        if x.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.conv4(y))
        y = F.relu(self.conv5(y))  # Batch x 1024 x NumInPoints

        # Max pooling for global feature vector:
        y = torch.max(y, 2)[0]  # Batch x 1024
        y = y.contiguous()

        return y


class PCRNet(nn.Module):
    def __init__(self, bottleneck_size=1024, input_shape="bcn"):
        super().__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        self.input_shape = input_shape

        self.feat = PointNetFeatures(bottleneck_size, input_shape)

        self.fc1 = nn.Linear(bottleneck_size * 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 7)

    def forward(self, x0, x1):
        # x shape should be B x 3 x N
        y0 = self.feat(x0)
        y1 = self.feat(x1)
        y = torch.cat([y0, y1], dim=1)

        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = F.relu(self.fc4(y))
        y = F.relu(self.fc5(y))
        y = self.fc6(y)  # Batch x 7

        pre_normalized_quat = y[:, 0:4]
        normalized_quat = F.normalize(pre_normalized_quat, dim=1)
        trans = y[:, 4:]
        y = torch.cat([normalized_quat, trans], dim=1)

        # returned y is a vector of 7 twist parameters.
        # pre_normalized_quat is used for loss on norm
        return y, pre_normalized_quat
