import torch
import warnings

from pointnet2.utils.pointnet2_utils import gather_operation as gather


class RandomSampler(torch.nn.Module):
    def __init__(self, num_out_points, input_shape="bcn", output_shape="bcn"):
        super().__init__()
        self.num_out_points = num_out_points
        self.name = "random"

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
            warnings.warn("RandomSampler: input_shape is different to output_shape.")
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor):
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1).contiguous()

        B, _, N  = x.shape
        idx = torch.zeros(B, self.num_out_points, dtype=torch.int32, device=x.device)
        for i in range(B):
            rand_perm = torch.randperm(
                N,
                dtype=torch.int32,
                device=x.device
            )
            idx[i] = rand_perm[:self.num_out_points]


        y = gather(x, idx)
        if self.output_shape == "bnc":
            y = y.permute(0, 2, 1).contiguous()

        return y
