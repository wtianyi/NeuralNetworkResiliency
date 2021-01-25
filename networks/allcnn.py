import torch as th
import torch.nn as nn
from torch.autograd import Variable
import math

from .utils import num_parameters
from .noisy_layers import *


class View(nn.Module):
    def __init__(self, o):
        super(View, self).__init__()
        self.o = o

    def forward(self, x):
        return x.view(-1, self.o)


# TODO: move to noisy layers or utils
# TODO: either accept more strict sigma_list input, or make the function more generic. It's basically not working for non-scalar sigma_list for now
def set_sigma_list(m, sigma_list):
    """ Allow None, scalar, 1-length list or
    list with the length of the number of noisy layers
    """
    noisy_conv_list = list(children_of_class(m, NoisyConv2d))

    if sigma_list is None:
        conv_sigma_list = [sigma_list] * len(noisy_conv_list)
    else:
        try:
            conv_sigma_list = float(sigma_list)
            conv_sigma_list = [conv_sigma_list] * len(noisy_conv_list)
        except:
            assert type(sigma_list) == list
            if len(sigma_list) == 1:
                conv_sigma_list = sigma_list * len(noisy_conv_list)
            else:
                assert len(conv_sigma_list) == len(noisy_conv_list)

    for m, s in zip(noisy_conv_list, conv_sigma_list):
        m.sigma = s

    noisy_fc_list = list(children_of_class(m, NoisyLinear))

    if sigma_list is None:
        fc_sigma_list = [sigma_list] * len(noisy_fc_list)
    else:
        try:
            fc_sigma_list = float(sigma_list)
            fc_sigma_list = [fc_sigma_list] * len(noisy_fc_list)
        except:
            assert type(sigma_list) == list
            if len(sigma_list) == 1:
                fc_sigma_list = sigma_list * len(noisy_fc_list)
            else:
                assert len(fc_sigma_list) == len(noisy_fc_list)

    for m, s in zip(noisy_fc_list, fc_sigma_list):
        m.sigma = s

    noisy_id_list = list(children_of_class(m, NoisyIdentity))

    if sigma_list is None:
        id_sigma_list = [sigma_list] * len(noisy_id_list)
    else:
        try:
            id_sigma_list = float(sigma_list)
            id_sigma_list = [id_sigma_list] * len(noisy_id_list)
        except:
            assert type(sigma_list) == list
            if len(sigma_list) == 1:
                id_sigma_list = sigma_list * len(noisy_id_list)
            else:
                assert len(id_sigma_list) == len(noisy_id_list)

    for m, s in zip(noisy_id_list, id_sigma_list):
        m.sigma = s

    noisy_bn_list = list(children_of_class(m, NoisyBN))

    if sigma_list is None:
        bn_sigma_list = [sigma_list] * len(noisy_bn_list)
    else:
        try:
            bn_sigma_list = float(sigma_list)
            bn_sigma_list = [bn_sigma_list] * len(noisy_bn_list)
        except:
            assert type(sigma_list) == list
            if len(sigma_list) == 1:
                bn_sigma_list = sigma_list * len(noisy_bn_list)
            else:
                assert len(bn_sigma_list) == len(noisy_bn_list)

    for m, s in zip(noisy_bn_list, bn_sigma_list):
        m.sigma = s


class mnistfc(nn.Module):
    def __init__(self, opt):
        super(mnistfc, self).__init__()
        self.name = "mnsitfc"

        c = 1024
        # opt['d'] = 0.5

        self.m = nn.Sequential(
            View(784),
            nn.Dropout(0.2),
            NoisyLinear(784, c),
            NoisyBN(c),
            nn.ReLU(True),
            nn.Dropout(opt["d"]),
            NoisyLinear(c, c),
            NoisyBN(c),
            nn.ReLU(True),
            nn.Dropout(opt["d"]),
            NoisyLinear(c, 10),
        )

        s = "[%s] Num parameters: %d" % (self.name, num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)

    def set_sigma_list(self, sigma_list):
        set_sigma_list(self, sigma_list)


class mnistconv(nn.Module):
    def __init__(self, opt):
        super(mnistconv, self).__init__()
        self.name = "mnistconv"
        # opt['d'] = 0.5

        def convbn(ci, co, ksz, psz, p):
            return nn.Sequential(
                NoisyConv2d(ci, co, ksz),
                NoisyBN(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz, stride=psz),
                nn.Dropout(p),
            )

        self.m = nn.Sequential(
            convbn(1, 20, 5, 3, opt["d"]),
            convbn(20, 50, 5, 2, opt["d"]),
            View(50 * 2 * 2),
            nn.Linear(50 * 2 * 2, 500),
            NoisyBN(500),
            nn.ReLU(True),
            nn.Dropout(opt["d"]),
            NoisyLinear(500, 10),
        )

        s = "[%s] Num parameters: %d" % (self.name, num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)

    def set_sigma_list(self, sigma_list):
        set_sigma_list(self, sigma_list)


class allcnn(nn.Module):
    def __init__(self, opt={"d": 0.5}, c1=96, c2=192):
        super(allcnn, self).__init__()
        self.name = "allcnn"
        # opt['d'] = 0.5

        def convbn(ci, co, ksz, s=1, pz=0):
            return nn.Sequential(
                NoisyConv2d(ci, co, ksz, stride=s, padding=pz),
                NoisyBN(co),
                nn.ReLU(True),
            )

        self.m = nn.Sequential(
            nn.Dropout(0.2),
            convbn(3, c1, 3, 1, 1),
            convbn(c1, c1, 3, 1, 1),
            convbn(c1, c1, 3, 2, 1),
            nn.Dropout(opt["d"]),
            convbn(c1, c2, 3, 1, 1),
            convbn(c2, c2, 3, 1, 1),
            convbn(c2, c2, 3, 2, 1),
            nn.Dropout(opt["d"]),
            convbn(c2, c2, 3, 1, 1),
            convbn(c2, c2, 3, 1, 1),
            convbn(c2, 10, 1, 1),
            nn.AvgPool2d(8),
            View(10),
        )

        s = "[%s] Num parameters: %d" % (self.name, num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)

    def set_sigma_list(self, sigma_list):
        set_sigma_list(self, sigma_list)


# vim: set tw=0 noai ts=4 sw=4 expandtab:
