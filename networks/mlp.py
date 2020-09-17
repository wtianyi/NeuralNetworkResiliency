from .noisy_layers import *
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .network_utils import children_of_class
from typing import Iterable
from itertools import cycle
from .quantization import CustomFakeQuantize, get_activation_quant, enable_fake_quant, enable_observer, disable_fake_quant, disable_observer

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layer_sizes, sigma_list=None, mu_list=0, activation_quant_levels=256):
        super(MLP, self).__init__()
        layer_sizes = [input_dim] + hidden_layer_sizes # + [num_classes]

        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(NoisyLinear(in_size, out_size, bias=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(get_activation_quant(activation_quant_levels, enable=False))
        layers.append(NoisyLinear(hidden_layer_sizes[-1], num_classes))
        layers.append(get_activation_quant(activation_quant_levels, enable=False))

        self.model = nn.Sequential(*layers)
        self.set_sigma_list(sigma_list)
        self.set_mu_list(mu_list)

    def forward(self, x):
        return(self.model(x))

    def set_mu_list(self, mu_list: Iterable) -> None:
        noisy_layer_list = list(children_of_class(self, NoisyLayer))

        if mu_list is None:
            mu_list = [0]
        else:
            try:
                mu_list = [float(mu_list)]
            except:
                pass

        for l, m in zip(noisy_layer_list, cycle(mu_list)):
            l.mu = m

    # TODO: rewrite as `set_mu_list`
    def set_sigma_list(self, sigma_list: Iterable) -> None:
        """ Allow None, scalar, 1-length list or
        list with the length of the number of noisy layers
        """
        noisy_layer_list = list(children_of_class(self, NoisyLayer))

        if sigma_list is None:
            sigma_list = [sigma_list]
        else:
            try:
                sigma_list = [float(sigma_list)]
            except:
                pass

        for m, s in zip(noisy_layer_list, cycle(sigma_list)):
            m.sigma = s

    def enable_quantization(self, flag: bool=True) -> None:
        for quant in children_of_class(self, CustomFakeQuantize):
            if flag:
                enable_fake_quant(quant)
                # enable_observer(quant)
            else:
                disable_fake_quant(quant)
                # disable_observer(quant)

    def set_quantization_level(self, quantization_levels: int) -> None:
        for quant in children_of_class(self, CustomFakeQuantize):
            quant.set_qmin_qmax(0, quantization_levels - 1)
