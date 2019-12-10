import torch
import numpy as np
import src.quaternion as Q  # works with (w, x, y, z) quaternions
import kornia.geometry.conversions as C  # works with (x, y, z, w) quaternions
import kornia.geometry.linalg as L


def deg_to_rad(deg):
    return np.pi / 180 * deg


def rad_to_deg(rad):
    return 180 / np.pi * rad


class QuaternionTransform:
    def __init__(self, vec: torch.Tensor, inverse: bool = False):
        # inversion: first apply translation
        self._inversion = torch.tensor([inverse])
        # a B x 7 vector of 4 quaternions and 3 translation parameters
        self.vec = vec.view([-1, 7])

    # Dict constructor
    @staticmethod
    def from_dict(d, device):
        return QuaternionTransform(d["vec"].to(device), d["inversion"][0].item())

    # Inverse Constructor
    def inverse(self):
        quat = self.quat()
        trans = self.trans()
        quat = Q.qinv(quat)
        trans = -trans

        vec = torch.cat([quat, trans], dim=1)
        return QuaternionTransform(vec, inverse=(not self.inversion()))

    def as_dict(self):
        return {"inversion": self._inversion, "vec": self.vec}

    def quat(self):
        return self.vec[:, 0:4]

    def trans(self):
        return self.vec[:, 4:]

    def inversion(self):
        # To handle dataloader batching of samples,
        # we take the first item's 'inversion' as the inversion set for the entire batch.
        return self._inversion[0].item()

    @staticmethod
    def wxyz_to_xyzw(q):
        q = q[..., [1, 2, 3, 0]]
        return q

    @staticmethod
    def xyzw_to_wxyz(q):
        q = q[..., [3, 0, 1, 2]]
        return q

    def compute_errors(self, other):
        # Calculate Quaternion Difference Norm Error
        # http://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf chapter 3.3
        # norm_err = torch.mean(
        #     torch.min(
        #         torch.sum((self.quat() - other.quat()) ** 2, dim=1),
        #         torch.sum((self.quat() + other.quat()) ** 2, dim=1),
        #     )
        # )

        q1 = self.quat()
        q2 = other.quat()
        R1 = C.quaternion_to_rotation_matrix(self.wxyz_to_xyzw(q1))
        R2 = C.quaternion_to_rotation_matrix(self.wxyz_to_xyzw(q2))
        R2inv = R2.transpose(1, 2)
        R1_R2inv = torch.bmm(R1, R2inv)

        # Calculate rotation error
        # rot_err = torch.norm(C.rotation_matrix_to_angle_axis(R1_R2inv), dim=1)
        # rot_err = torch.mean(rot_err)

        # Taken from PCN: Point Completion Network
        # https://arxiv.org/pdf/1808.00671.pdf
        rot_err = torch.mean(2 * torch.acos(2 * (torch.sum(q1 * q2, dim=1)) ** 2 - 1))

        # Calculate deviation from Identity
        batch = R1_R2inv.shape[0]
        I = torch.eye(3).unsqueeze(0).expand([batch, -1, -1]).to(R1_R2inv)
        norm_err = torch.sum((R1_R2inv - I) ** 2, dim=(1, 2))
        norm_err = torch.mean(norm_err)

        trans_err = torch.mean(torch.sqrt((self.trans() - other.trans()) ** 2))

        return rot_err, norm_err, trans_err

    def rotate(self, p: torch.Tensor):
        ndim = p.dim()
        if ndim == 2:
            N, _ = p.shape
            assert self.vec.shape[0] == 1
            # repeat transformation vector for each point in shape
            quat = self.quat().expand([N, -1])
            p_rotated = Q.qrot(quat, p)

        elif ndim == 3:
            B, N, _ = p.shape
            quat = self.quat().unsqueeze(1).expand([-1, N, -1]).contiguous()
            p_rotated = Q.qrot(quat, p)

            # R = C.quaternion_to_rotation_matrix(
            #     self.wxyz_to_xyzw(self.quat())
            # )
            # # convert to homogenus transform matrix
            # Rt = torch.nn.functional.pad(R, (0, 1, 0, 1))
            # Rt[:, 3, 3] = 1
            # p_rotated = L.transform_points(Rt, p)

        return p_rotated


def create_random_transform(dtype, max_rotation_deg, max_translation):
    max_rotation = deg_to_rad(max_rotation_deg)
    rot = np.random.uniform(-max_rotation, max_rotation, [1, 3])
    trans = np.random.uniform(-max_translation, max_translation, [1, 3])
    quat = Q.euler_to_quaternion(rot, "xyz")

    vec = np.concatenate([quat, trans], axis=1)
    vec = torch.tensor(vec, dtype=dtype)
    return QuaternionTransform(vec)


class QuaternionFixedDataset(torch.utils.data.Dataset):
    def __init__(self, data, repeat=1, seed=0, apply_noise=False, fixed_noise=False):
        super().__init__()
        self.data = data
        self.include_shapes = data.include_shapes
        self.len_data = len(data)
        self.len_set = len(data) * repeat

        # Fix numpy seed and create fixed transform list
        np.random.seed(seed)
        self.transforms = [
            create_random_transform(torch.float32, 45, 0) for _ in range(self.len_set)
        ]

        self.noise = None
        if fixed_noise:
            self.noise = torch.tensor(
                [0.04 * np.random.randn(1024, 3) for _ in range(self.len_set)],
                dtype=torch.float32,
            )

        self.apply_noise = apply_noise
        self.fixed_noise = fixed_noise

    def __len__(self):
        return self.len_set

    def __getitem__(self, index):
        if self.include_shapes:
            p0, _, shape = self.data[index % self.len_data]
        else:
            p0, _ = self.data[index % self.len_data]
        gt = self.transforms[index]
        p1 = gt.rotate(p0)
        if self.apply_noise:
            if self.fixed_noise:
                noise = self.noise[index].to(p1)
            else:
                noise = torch.tensor(0.04 * np.random.randn(1024, 3)).to(p1)
            p1 = p1 + noise

        igt = gt.as_dict()  # p0 ---> p1

        if self.include_shapes:
            return p0, p1, igt, shape

        return p0, p1, igt


if __name__ == "__main__":
    toy = np.array([[[1.0, 1.0, 1.0], [2, 2, 2]], [[0.0, 1.0, 0.0], [0, 2, 0]]])
    toy = torch.tensor(toy, dtype=torch.float32)
