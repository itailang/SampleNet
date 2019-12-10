from __future__ import print_function

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN

try:
    from .soft_projection import SoftProjection
    from .chamfer_distance import ChamferDistance
    from . import sputils
except (ModuleNotFoundError, ImportError) as err:
    print(err.__repr__())
    from soft_projection import SoftProjection
    from chamfer_distance import ChamferDistance
    import sputils


class SampleNet(nn.Module):
    def __init__(
        self,
        num_out_points,
        bottleneck_size,
        group_size,
        initial_temperature=1.0,
        is_temperature_trainable=True,
        min_sigma=1e-2,
        input_shape="bcn",
        output_shape="bcn",
        complete_fps=True,
        skip_projection=False,
    ):
        super().__init__()
        self.num_out_points = num_out_points
        self.name = "samplenet"

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, bottleneck_size, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(bottleneck_size)

        self.fc1 = nn.Linear(bottleneck_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3 * num_out_points)

        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(256)

        # projection and matching
        self.project = SoftProjection(
            group_size, initial_temperature, is_temperature_trainable, min_sigma
        )
        self.skip_projection = skip_projection
        self.complete_fps = complete_fps

        # input / output shapes
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if output_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if input_shape != output_shape:
            warnings.warn("SampleNet: input_shape is different to output_shape.")
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor):
        # x shape should be B x 3 x N
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1)

        if x.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        y = F.relu(self.bn5(self.conv5(y)))  # Batch x 128 x NumInPoints

        # Max pooling for global feature vector:
        y = torch.max(y, 2)[0]  # Batch x 128

        y = F.relu(self.bn_fc1(self.fc1(y)))
        y = F.relu(self.bn_fc2(self.fc2(y)))
        y = F.relu(self.bn_fc3(self.fc3(y)))
        y = self.fc4(y)

        y = y.view(-1, 3, self.num_out_points)

        # Simplified points
        simp = y
        match = None
        proj = None

        # Projected points
        if self.training:
            if not self.skip_projection:
                proj = self.project(point_cloud=x, query_cloud=y)
            else:
                proj = simp

        # Matched points
        else:  # Inference
            # Retrieve nearest neighbor indices
            _, idx = KNN(1, transpose_mode=False)(x.contiguous(), y.contiguous())

            """Notice that we detach the tensors and do computations in numpy,
            and then convert back to Tensors.
            This should have no effect as the network is in eval() mode
            and should require no gradients.
            """

            # Convert to numpy arrays in B x N x 3 format. we assume 'bcn' format.
            x = x.permute(0, 2, 1).cpu().detach().numpy()
            y = y.permute(0, 2, 1).cpu().detach().numpy()

            idx = idx.cpu().detach().numpy()
            idx = np.squeeze(idx, axis=1)

            z = sputils.nn_matching(
                x, idx, self.num_out_points, complete_fps=self.complete_fps
            )

            # Matched points are in B x N x 3 format.
            match = torch.tensor(z, dtype=torch.float32).cuda()

        # Change to output shapes
        if self.output_shape == "bnc":
            simp = simp.permute(0, 2, 1)
            if proj is not None:
                proj = proj.permute(0, 2, 1)
        elif self.output_shape == "bcn" and match is not None:
            match = match.permute(0, 2, 1)
            match = match.contiguous()

        # Assert contiguous tensors
        simp = simp.contiguous()
        if proj is not None:
            proj = proj.contiguous()
        if match is not None:
            match = match.contiguous()

        out = proj if self.training else match

        return simp, out

    def sample(self, x):
        simp, proj, match, feat = self.__call__(x)
        return proj

    # Losses:
    # At inference time, there are no sampling losses.
    # When evaluating the model, we'd only want to asses the task loss.

    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
        if self.skip_projection or not self.training:
            return torch.tensor(0).to(ref_pc)
        # ref_pc and samp_pc are B x N x 3 matrices
        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
        return loss

    def get_projection_loss(self):
        sigma = self.project.sigma()
        if self.skip_projection or not self.training:
            return torch.tensor(0).to(sigma)
        return sigma


if __name__ == "__main__":
    point_cloud = np.random.randn(1, 3, 1024)
    point_cloud_pl = torch.tensor(point_cloud, dtype=torch.float32).cuda()
    net = SampleNet(5, 128, group_size=10, initial_temperature=0.1, complete_fps=True)

    net.cuda()
    net.eval()

    for param in net.named_modules():
        print(param)

    simp, proj, match = net.forward(point_cloud_pl)
    simp = simp.detach().cpu().numpy()
    proj = proj.detach().cpu().numpy()
    match = match.detach().cpu().numpy()

    print("*** SIMPLIFIED POINTS ***")
    print(simp)
    print("*** PROJECTED POINTS ***")
    print(proj)
    print("*** MATCHED POINTS ***")
    print(match)

    mse_points = np.sum((proj - match) ** 2, axis=1)
    print("projected points vs. matched points error per point:")
    print(mse_points)
