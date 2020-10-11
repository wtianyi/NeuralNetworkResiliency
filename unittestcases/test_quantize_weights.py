import torch
from torch.utils.data import DataLoader
from networks import *
from utils import *
import unittest
import argparse
import random
import numpy as np

from collections import OrderedDict

from training_functions import quantize_network
from trajectory.utils import extract_param_vec


class TestLoadQuantizeConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_batches = 5
        cls.device = "cuda:0"
        cls.args = argparse.Namespace(
            net_type="mlp",
            dataset="morse",
            training_noise_type="gaussian",
            depth=28,
            widen_factor=10,
            dropout_rate=0.3,
            device=[0],
            training_noise=0.1,
            training_noise_mean=0,
            regularization_type='l2',
            regularization=5e-4,
            lr=1e-2,
            num_epochs=200,
            epochs_lr_decay=60,
            forward_samples=1,
            optim_type='SGD',
            run_name=None,
            cpu=False
        )
        cls.dataset, _, cls.num_classes = get_datasets(cls.args.dataset)
        cls.dataloader = DataLoader(cls.dataset, batch_size=16, shuffle=True, num_workers=0)
        cls.network, _ = get_network(cls.args, cls.num_classes)
        cls.network = cls.network.to(cls.device)
        cls.criterion = nn.CrossEntropyLoss()
        cls.seed = 42

    def test_quant_config(self):
        quant_levels_dict = OrderedDict([
            ("*.0", (16, 16)),
            ("*.2", (None, 8)),
        ])
        apply_quant_profile(quant_levels_dict, self.network)


class TestQuantizeNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_batches = 5
        args = argparse.Namespace(
            net_type="lenet",
            dataset="mnist",
            training_noise_type="gaussian",
            depth=18,
            dropout_rate=0.3,
            device=[0],
            cpu=True
        )
        cls.dataset, _, cls.num_classes = get_datasets("mnist")
        cls.dataloader = torch.utils.data.DataLoader(
            cls.dataset, batch_size=16, shuffle=True, num_workers=0)

    def test_quantize_network(self):
        for nlevels in [2, 3, 4, 5, 6, 7, 8, 10, 16, 32, 64, 96, 128, 256]:
            self.network = LeNet(10)
            # self.network.to('cpu')
            self.network.eval()
            quantize_network(self.network, num_weight_quant_levels=nlevels,
                             calibration_dataloader=self.dataloader)
            for name, child in self.network.named_children():
                print(f"{name}: {type(child)}")
                # print(child.qconfig)
                # try:
                #     weight_levels = torch.tensor([l.unique().nelement() for l in child.weight().dequantize().detach()])
                #     bias_levels = child.bias().detach().unique().nelement()
                #     print(f"weight levels: {weight_levels}, set levels: {nlevels}")
                #     self.assertTrue(torch.all(weight_levels <= nlevels), name + " weight")
                #     print(f"bias levels: {bias_levels}, set levels: {nlevels}")
                #     self.assertLessEqual(bias_levels, nlevels, name + " bias")
                # except Exception as e:
                # print(e)
                if hasattr(child, "weight") and child.weight is not None:
                    weight_levels = torch.tensor(
                        [l.unique().nelement() for l in child.weight.detach()])
                    print(
                        f"weight levels: {weight_levels}, set levels: {nlevels}")
                    self.assertTrue(torch.all(weight_levels <=
                                              nlevels), name + " weight")
                if hasattr(child, "bias") and child.bias is not None:
                    bias_levels = child.bias.detach().unique().nelement()
                    print(f"bias levels: {bias_levels}, set levels: {nlevels}")
                    self.assertLessEqual(bias_levels, nlevels, name + " bias")


if __name__ == '__main__':
    unittest.main()
