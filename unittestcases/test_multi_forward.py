from os import stat
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from networks import *
from utils import *
import unittest
import argparse
import random
import numpy as np
from itertools import product, islice

class TestMultiForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_batches = 5
        cls.device = "cuda:0"
        cls.args = argparse.Namespace(
            net_type="wide_resnet",
            dataset="cifar10",
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

    def test_multiple_forward(self):
        num_passes_list = [3, 5, 10]
        noise_levels = [0, 0.1, 0.2]
        self.network.apply(set_noisy)
        num_batches = 10
        for num, sigma in product(num_passes_list, noise_levels):
            print(f"num={num}, sigma={sigma}")
            self.network.set_sigma_list(sigma)
            for x, y in islice(self.dataloader, num_batches):
                set_random_seed(self.seed)
                grad1 = self._forward_backward_1(x, y, num)
                set_random_seed(self.seed)
                grad2 = self._forward_backward_2(x, y, num)
                self.assertTrue(torch.allclose(grad1, grad2, atol=1e-5), f"grad1={grad1}\ngrad2={grad2}")

    def _forward_backward_1(self, x, y, num):
        x, y = x.to(self.device), y.to(self.device)
        self.network.zero_grad()
        for i in range(num):
            # for n, m in self.network.named_modules():
            #     if isinstance(m, NoisyLayer):
            #         print(f"{n}: {m.param_range_dict}")
            output = self.network(x)
            loss = self.criterion(output, y)
            loss.backward()
        for p in self.network.parameters():
            p.grad.data.mul_(1/num)
        return self._flatten_grad(self.network)

    def _forward_backward_2(self, x, y, num):
        x, y = x.to(self.device), y.to(self.device)
        if num == 1:
            self.network.zero_grad()
            outputs = self.network(x)
            loss = self.criterion(outputs, y)
            loss.backward()
        else:
            grad_list = []
            for j in range(num):
                outputs = self.network(x) 
                loss = self.criterion(outputs, y)
                self.network.zero_grad()
                loss.backward()
                if j == 0:
                    for p in self.network.parameters():
                        if p.grad is not None:
                            grad = p.grad/num
                            grad_list.append(grad)
                else:
                    p_id = 0
                    for p in self.network.parameters():
                        if p.grad is not None:
                            grad_list[p_id] += p.grad/num
                            p_id += 1

            p_id = 0
            for p in self.network.parameters():
                if p.grad is not None:
                    p.grad = grad_list[p_id]
                    p_id += 1
        return self._flatten_grad(self.network)


    @staticmethod
    def _flatten_grad(net: torch.nn.Module) -> torch.Tensor:
        try:
            return torch.cat([p.grad.detach().view(-1).float() for p in net.parameters()], dim=0)
        except Exception as e: # Basically it's because p.grad is None
            print(e)

if __name__ == '__main__':
    unittest.main()