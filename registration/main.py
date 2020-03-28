import argparse
import logging
import os
import sys

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from data.modelnet_loader_torch import ModelNetCls
from models import pcrnet
from src import ChamferDistance, FPSSampler, RandomSampler, SampleNet
from src import sputils
from src.pctransforms import OnUnitCube, PointcloudToTensor
from src.qdataset import QuaternionFixedDataset, QuaternionTransform, rad_to_deg

torch.manual_seed(0)

# addpath('../')
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

# dump to GLOBALS dictionary
GLOBALS = None


def append_to_GLOBALS(key, value):
    try:
        GLOBALS[key].append(value)
    except KeyError:
        GLOBALS[key] = []
        GLOBALS[key].append(value)


# fmt: off
def options(argv=None, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='BASENAME', help='output filename (prefix)')  # the result: ${BASENAME}_model_best.pth
    parser.add_argument('--datafolder', required=True, type=str, help='dataset folder')

    # For testing
    parser.add_argument('--test', action='store_true',
                        help='Perform testing routine. Otherwise, the script will train.')

    # Default pointnet behavior is 'fixed'.
    # Loading options:
    #   --transfer-from: load a pretrained PCRNET model.
    #   --resume: load an ongoing training SP-PCRNET model.
    #   --pretrained: load a pretrained SP-PCRNET model (like resume, but reset starting epoch)

    parser.add_argument('--loss-type', default=0, choices=[0, 1], type=int,
                        metavar='TYPE', help='Supervised (0) or Unsupervised (1)')
    parser.add_argument('--sampler', required=True, choices=['fps', 'samplenet', 'random', 'none'], type=str,
                        help='Sampling method.')

    parser.add_argument('--transfer-from', type=str,
                        metavar='PATH', help='path to trained pcrnet')
    parser.add_argument('--train-pcrnet', action='store_true',
                        help='Allow PCRNet training.')
    parser.add_argument('--train-samplenet', action='store_true',
                        help='Allow SampleNet training.')

    parser.add_argument('--num-sampled-clouds', choices=[1, 2], type=int, default=2,
                        help='Number of point clouds to sample (Source / Source + Template)')

    # settings for on training
    parser.add_argument('--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=400, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD', 'RMSProp'],
                        metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args(argv)
    return args
# fmt: on


def main(args, dbg=False):
    global GLOBALS
    if dbg:
        GLOBALS = {}

    trainset, testset = get_datasets(args)
    action = Action(args)
    if args.test:
        test(args, testset, action)
    else:
        train(args, trainset, testset, action)

    return GLOBALS


def test(args, testset, action):
    if not torch.cuda.is_available():
        args.device = "cpu"
    args.device = torch.device(args.device)

    model = action.create_model()

    # action.try_transfer(model, args.pretrained)
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location="cpu"))

    model.to(args.device)
    model.eval()  # Batch norms etc. configured for testing mode.

    # Dataloader
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=args.workers
    )

    action.test_1(model, testloader, args.device, epoch=0)


def train(args, trainset, testset, action):
    if not torch.cuda.is_available():
        args.device = "cpu"
    args.device = torch.device(args.device)

    model = action.create_model()

    # action.try_transfer(model, args.pretrained)
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location="cpu"))
    model.to(args.device)

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])

    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    # Optimizer
    min_loss = float("inf")
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(learnable_params, lr=1e-3)
    elif args.optimizer == "RMSProp":
        optimizer = torch.optim.RMSprop(learnable_params, lr=0.001)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.001, momentum=0.9)

    if checkpoint is not None:
        min_loss = checkpoint["min_loss"]
        optimizer.load_state_dict(checkpoint["optimizer"])

    # training
    LOGGER.debug("train, begin")
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_rotation_error = action.train_1(
            model, trainloader, optimizer, args.device, epoch
        )
        val_loss, val_rotation_error = action.eval_1(
            model, testloader, args.device, epoch
        )

        # scheduler.step()

        is_best = val_loss < min_loss
        min_loss = min(val_loss, min_loss)

        LOGGER.info(
            "epoch, %04d, train_loss=%f, train_rotation_error=%f, val_loss=%f, val_rotation_error=%f",
            epoch + 1,
            train_loss,
            train_rotation_error,
            val_loss,
            val_rotation_error,
        )

        snap = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "min_loss": min_loss,
            "optimizer": optimizer.state_dict(),
        }
        if is_best:
            save_checkpoint(snap, args.outfile, "snap_best")
            save_checkpoint(model.state_dict(), args.outfile, "model_best")

        save_checkpoint(snap, args.outfile, "snap_last")
        save_checkpoint(model.state_dict(), args.outfile, "model_last")

    LOGGER.debug("train, end")


def save_checkpoint(state, filename, suffix):
    torch.save(state, "{}_{}.pth".format(filename, suffix))


class Action:
    def __init__(self, args):
        self.experiment_name = args.pretrained

        self.transfer_from = args.transfer_from

        self.p0_zero_mean = True
        self.p1_zero_mean = True

        self.LOSS_TYPE = args.loss_type

        # SampleNet:
        self.ALPHA = args.alpha  # Sampling loss
        self.LMBDA = args.lmbda  # Projection loss
        self.GAMMA = args.gamma  # Inside sampling loss - linear.
        self.DELTA = args.delta  # Inside sampling loss - point cloud size factor.
        self.NUM_IN_POINTS = args.num_in_points
        self.NUM_OUT_POINTS = args.num_out_points
        self.BOTTLNECK_SIZE = args.bottleneck_size
        self.GROUP_SIZE = args.projection_group_size

        self.SKIP_PROJECTION = args.skip_projection
        self.SAMPLER = args.sampler

        self.TRAIN_SAMPLENET = args.train_samplenet
        self.TRAIN_PCRNET = args.train_pcrnet
        self.NUM_SAMPLED_CLOUDS = args.num_sampled_clouds

    def create_model(self):
        # Create Task network and load pretrained feature weights if requested
        pcrnet_model = pcrnet.PCRNet(input_shape="bnc")

        if self.TRAIN_PCRNET:
            pcrnet_model.requires_grad_(True)
            pcrnet_model.train()
        else:
            pcrnet_model.requires_grad_(False)
            pcrnet_model.eval()

        # Create sampling network
        if self.SAMPLER == "samplenet":
            sampler = SampleNet(
                num_out_points=self.NUM_OUT_POINTS,
                bottleneck_size=self.BOTTLNECK_SIZE,
                group_size=self.GROUP_SIZE,
                initial_temperature=1.0,
                input_shape="bnc",
                output_shape="bnc",
                skip_projection=self.SKIP_PROJECTION,
            )

            if self.TRAIN_SAMPLENET:
                sampler.requires_grad_(True)
                sampler.train()
            else:
                sampler.requires_grad_(False)
                sampler.eval()

        elif self.SAMPLER == "fps":
            sampler = FPSSampler(
                self.NUM_OUT_POINTS, permute=True, input_shape="bnc", output_shape="bnc"
            )

        elif self.SAMPLER == "random":
            sampler = RandomSampler(
                self.NUM_OUT_POINTS, input_shape="bnc", output_shape="bnc"
            )

        else:
            sampler = None

        # Load pcrnet baseline weights
        self.try_transfer(pcrnet_model, self.transfer_from)

        # Attach sampler to pcrnet_model
        pcrnet_model.sampler = sampler

        return pcrnet_model

    @staticmethod
    def try_transfer(model, path):
        if path is not None:
            model.load_state_dict(torch.load(path, map_location="cpu"))
            LOGGER.info(f"Model loaded from {path}")

    def train_1(self, model, trainloader, optimizer, device, epoch):
        vloss = 0.0
        gloss = 0.0

        count = 0
        for i, data in enumerate(tqdm(trainloader)):
            # Sample using one of the samplers:
            if model.sampler is not None and model.sampler.name == "samplenet":
                (
                    sampler_loss,
                    sampled_data,
                    sampler_loss_info,
                ) = self.compute_samplenet_loss(model, data, device)
                simplification_loss = sampler_loss_info["simplification_loss"]
                projection_loss = sampler_loss_info["projection_loss"]
            elif model.sampler is not None and model.sampler.name == "fps":
                sampled_data = self.non_learned_sampling(model, data, device)
                simplification_loss = torch.tensor(0, dtype=torch.float32)
                projection_loss = torch.tensor(0, dtype=torch.float32)
                sampler_loss = torch.tensor(0, dtype=torch.float32)
            else:
                sampled_data = data
                simplification_loss = torch.tensor(0, dtype=torch.float32)
                projection_loss = torch.tensor(0, dtype=torch.float32)
                sampler_loss = torch.tensor(0, dtype=torch.float32)

            pcrnet_loss, pcrnet_loss_info = self.compute_pcrnet_loss(
                model, sampled_data, device, epoch
            )

            chamfer_loss = pcrnet_loss_info["chamfer_loss"]
            rotation_error = pcrnet_loss_info["rot_err"]
            norm_err = pcrnet_loss_info["norm_err"]
            trans_err = pcrnet_loss_info["trans_err"]

            # print(
            #     f"data sample {i:3.0f}: simplification_loss={simplification_loss:.4f}, projection_loss={projection_loss:.4f}, chamfer_loss={chamfer_loss:.4f}, rotation_error={rotation_error:.4f}, norm_err={norm_err:.4f}, trans_err={trans_err:.4f}"
            # )

            # SampleNet loss is already factorized by ALPHA and LMBDA hyper parameters.
            loss = pcrnet_loss + sampler_loss

            optimizer.zero_grad()
            loss.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=10.0)

            optimizer.step()

            vloss1 = loss.item()
            vloss += vloss1
            gloss1 = rotation_error.item()
            gloss += gloss1
            count += 1

        ave_vloss = float(vloss) / count
        ave_gloss = float(gloss) / count
        return ave_vloss, ave_gloss

    def eval_1(self, model, testloader, device, epoch):
        vloss = 0.0
        gloss = 0.0

        # Shift to eval mode for BN / Projection layers
        task_state = model.training
        if model.sampler is not None:
            sampler_state = model.sampler.training
        model.eval()

        count = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                # Sample using one of the samplers:
                if model.sampler is not None and model.sampler.name == "samplenet":
                    (
                        sampler_loss,
                        sampled_data,
                        sampler_loss_info,
                    ) = self.compute_samplenet_loss(model, data, device)
                elif model.sampler is not None and model.sampler.name == "fps":
                    sampled_data = self.non_learned_sampling(model, data, device)
                    sampler_loss = torch.tensor(0, dtype=torch.float32)
                else:
                    sampled_data = data
                    sampler_loss = torch.tensor(0, dtype=torch.float32)

                pcrnet_loss, pcrnet_loss_info = self.compute_pcrnet_loss(
                    model, sampled_data, device, epoch
                )

                rotation_error = pcrnet_loss_info["rot_err"]

                # samplenet loss is already factorized by ALPHA and LMBDA hyper parameters.
                loss = pcrnet_loss + sampler_loss

                vloss1 = loss.item()
                vloss += vloss1
                gloss1 = rotation_error.item()
                gloss += gloss1
                count += 1

        ave_vloss = float(vloss) / count
        ave_gloss = float(gloss) / count

        # Shift back to training (?) mode for task and samppler
        model.train(task_state)
        if model.sampler is not None:
            model.sampler.train(sampler_state)

        return ave_vloss, ave_gloss

    def test_1(self, model, testloader, device, epoch):
        rotation_errors = []
        trans_errs = []
        consistency_errors = []

        with torch.no_grad():
            for i, data_and_shape in enumerate(tqdm(testloader)):

                data = data_and_shape[0:3]
                shape = data_and_shape[3]

                # Sample using one of the samplers:
                if model.sampler is not None and model.sampler.name == "samplenet":
                    _, sampled_data, _ = self.compute_samplenet_loss(
                        model, data, device
                    )
                elif model.sampler is not None and (
                    model.sampler.name in ["fps", "random"]
                ):
                    sampled_data = self.non_learned_sampling(model, data, device)
                else:
                    sampled_data = data

                _, pcrnet_loss_info = self.compute_pcrnet_loss(
                    model, sampled_data, device, epoch
                )

                consistency = self.compute_sampling_consistency(sampled_data, device)
                consistency_errors.append(consistency.item())

                rotation_error = pcrnet_loss_info["rot_err"]
                trans_err = pcrnet_loss_info["trans_err"]

                rotation_errors.append(rotation_error.item())
                trans_errs.append(trans_err.item())

                if GLOBALS is not None:
                    append_to_GLOBALS("data", data)
                    append_to_GLOBALS("rotation_error", rotation_error)
                    append_to_GLOBALS("sampled_data", sampled_data)
                    append_to_GLOBALS(
                        "est_transform", pcrnet_loss_info["est_transform"]
                    )
                    append_to_GLOBALS("shape", shape)

        # Compute Precision curve and AUC.
        rotation_errors = np.array(rotation_errors)
        trans_errs = np.array(trans_errs)
        consistency_errors = np.array(consistency_errors)
        n_samples = len(testloader)
        x = np.arange(0.0, 180.0, 0.5)
        y = np.zeros(len(x))
        for idx, err in enumerate(x):
            precision = np.sum(rotation_errors <= err) / n_samples
            y[idx] = precision

        # plt.figure()
        # plt.plot(x, y)
        # plt.show()
        # plt.savefig("test.png")

        auc = np.sum(y) / len(x)
        print(f"Experiment name: {self.experiment_name}")
        print(f"AUC = {auc}")
        print(f"Mean rotation Error = {np.mean(rotation_errors)}")
        print(f"STD rotation Error = {np.std(rotation_errors)}")
        print(f"Mean consistency Error = {np.mean(consistency_errors)}")
        print(f"STD consistency Error = {np.std(consistency_errors)}")

    def non_learned_sampling(self, model, data, device):
        """Sample p1 point cloud using FPS."""
        p0, p1, igt = data
        p0 = p0.to(device)  # template
        p1 = p1.to(device)  # source

        p1_samp = model.sampler(p1)
        if self.NUM_SAMPLED_CLOUDS == 1:
            sampled_data = (p0, p1_samp, igt)
        elif self.NUM_SAMPLED_CLOUDS == 2:  # Sample template point cloud as well
            p0_samp = model.sampler(p0)
            sampled_data = (p0_samp, p1_samp, igt)

        return sampled_data

    def compute_samplenet_loss(self, model, data, device):
        """Sample point clouds using SampleNet and compute sampling associated losses."""

        p0, p1, igt = data
        p0 = p0.to(device)  # template
        p1 = p1.to(device)  # source

        p1_simplified, p1_projected = model.sampler(p1)

        # Sampling loss
        p1_simplification_loss = model.sampler.get_simplification_loss(
            p1, p1_simplified, self.NUM_OUT_POINTS, self.GAMMA, self.DELTA
        )

        if self.NUM_SAMPLED_CLOUDS == 1:
            simplification_loss = p1_simplification_loss
            sampled_data = (p0, p1_projected, igt)

        elif self.NUM_SAMPLED_CLOUDS == 2:  # Sample template point cloud as well
            p0_simplified, p0_projected = model.sampler(p0)
            p0_simplification_loss = model.sampler.get_simplification_loss(
                p0, p0_simplified, self.NUM_OUT_POINTS, self.GAMMA, self.DELTA
            )
            simplification_loss = 0.5 * (
                p1_simplification_loss + p0_simplification_loss
            )
            sampled_data = (p0_projected, p1_projected, igt)

        # Projection loss
        projection_loss = model.sampler.get_projection_loss()

        samplenet_loss = self.ALPHA * simplification_loss + self.LMBDA * projection_loss

        samplenet_loss_info = {
            "simplification_loss": simplification_loss,
            "projection_loss": projection_loss,
        }

        return samplenet_loss, sampled_data, samplenet_loss_info

    def compute_sampling_consistency(self, sampled_data, device):
        p0s, p1s, igt = sampled_data
        p0s = p0s.to(device)  # template
        p1s = p1s.to(device)  # source

        gt_transform = QuaternionTransform.from_dict(igt, device)
        # p1s_est = gt_transform.rotate(p0s)
        p0s_est = gt_transform.inverse().rotate(p1s)

        # cost_p0_p1, cost_p1_p0 = ChamferDistance()(p1s, p1s_est)
        cost_p0_p1, cost_p1_p0 = ChamferDistance()(p0s, p0s_est)
        cost_p0_p1 = torch.mean(cost_p0_p1)
        cost_p1_p0 = torch.mean(cost_p1_p0)

        consistency = cost_p0_p1 + cost_p1_p0
        return consistency

    def compute_pcrnet_loss(self, model, data, device, epoch):
        p0, p1, igt = data
        p0 = p0.to(device)  # template
        p1 = p1.to(device)  # source
        # igt = igt.to(device) # igt: p0 -> p1

        twist, pre_normalized_quat = model(p0, p1)

        # https://arxiv.org/pdf/1805.06485.pdf QuaterNet quaternient regularization loss
        qnorm_loss = torch.mean((torch.sum(pre_normalized_quat ** 2, dim=1) - 1) ** 2)

        est_transform = QuaternionTransform(twist)
        gt_transform = QuaternionTransform.from_dict(igt, device)

        p1_est = est_transform.rotate(p0)

        cost_p0_p1, cost_p1_p0 = ChamferDistance()(p1, p1_est)
        cost_p0_p1 = torch.mean(cost_p0_p1)
        cost_p1_p0 = torch.mean(cost_p1_p0)

        chamfer_loss = cost_p0_p1 + cost_p1_p0

        rot_err, norm_err, trans_err = est_transform.compute_errors(gt_transform)

        if self.LOSS_TYPE == 0:
            pcrnet_loss = 1.0 * norm_err + 1.0 * chamfer_loss

        elif self.LOSS_TYPE == 1:
            pcrnet_loss = chamfer_loss

        rot_err = rad_to_deg(rot_err)

        pcrnet_loss_info = {
            "chamfer_loss": chamfer_loss,
            "qnorm_loss": qnorm_loss,
            "rot_err": rot_err,
            "norm_err": norm_err,
            "trans_err": trans_err,
            "est_transform": est_transform,
        }

        return pcrnet_loss, pcrnet_loss_info


def get_datasets(args):
    transforms = torchvision.transforms.Compose([PointcloudToTensor(), OnUnitCube()])

    if not args.test:
        traindata = ModelNetCls(
            args.num_in_points,
            transforms=transforms,
            train=True,
            download=False,
            folder=args.datafolder,
        )
        testdata = ModelNetCls(
            args.num_in_points,
            transforms=transforms,
            train=False,
            download=False,
            folder=args.datafolder,
        )

        train_repeats = max(int(5000 / len(traindata)), 1)

        trainset = QuaternionFixedDataset(traindata, repeat=train_repeats, seed=0,)
        testset = QuaternionFixedDataset(testdata, repeat=1, seed=0)
    else:
        testdata = ModelNetCls(
            args.num_in_points,
            transforms=transforms,
            train=False,
            download=False,
            cinfo=None,
            folder=args.datafolder,
            include_shapes=True,
        )
        trainset = None
        testset = QuaternionFixedDataset(testdata, repeat=5, seed=1)

    return trainset, testset


if __name__ == "__main__":
    ARGS = options(parser=sputils.get_parser())

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s:%(name)s, %(asctime)s, %(message)s",
        filename=f"{ARGS.outfile}.log",
    )
    LOGGER.debug("Training (PID=%d), %s", os.getpid(), ARGS)

    _ = main(ARGS)
    LOGGER.debug("done (PID=%d)", os.getpid())
