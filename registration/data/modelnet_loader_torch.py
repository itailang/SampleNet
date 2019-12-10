from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import subprocess
import shlex
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:]
    return data, label


class ModelNetCls(data.Dataset):
    def __init__(
        self,
        num_points,
        transforms,
        train,
        download=True,
        cinfo=None,
        folder="modelnet10_hdf5_2048",
        url="https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
        include_shapes=False,
    ):
        super().__init__()

        self.transforms = transforms

        self.folder = folder
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        self.url = url

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(BASE_DIR, os.path.basename(self.url))
            subprocess.check_call(
                shlex.split("curl {} -o {}".format(self.url, zipfile))
            )

            subprocess.check_call(
                shlex.split("unzip {} -d {}".format(zipfile, BASE_DIR))
            )

            subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        self.train = train
        if self.train:
            self.files = _get_data_files(os.path.join(self.data_dir, "train_files.txt"))
        else:
            self.files = _get_data_files(os.path.join(self.data_dir, "test_files.txt"))

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(BASE_DIR, f))
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)
        if np.ndim(self.labels) == 1:
            self.labels = np.expand_dims(self.labels, axis=1)
        self.set_num_points(num_points)

        if cinfo is not None:
            self.classes, self.class_to_idx = cinfo
        else:
            self.classes, self.class_to_idx = (None, None)

        self.shapes = []
        self.include_shapes = include_shapes
        if self.include_shapes:
            N = len(self.files)
            if self.train:
                T = "train"
            else:
                T = "test"

            for n in range(N):
                jname = os.path.join(self.data_dir, f"ply_data_{T}_{n}_id2file.json")
                with open(jname, "r") as f:
                    shapes = json.load(f)
                    self.shapes += shapes

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        if self.include_shapes:
            shape = self.shapes[idx]
            return current_points, label, shape

        return current_points, label

    def __len__(self):
        return self.points.shape[0]

    def set_num_points(self, pts):
        self.num_points = min(self.points.shape[1], pts)

    def randomize(self):
        pass


if __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            d_utils.PointcloudScale(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
        ]
    )
    dset = ModelNetCls(16, train=True, transforms=transforms)
    print(dset[0][0])
    print(dset[0][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
