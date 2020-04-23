# See https://medium.com/@karanbirchahal/aggressive-quantization-how-to-run-mnist-on-a-4-bit-neural-net-using-pytorch-5703f3faa599
from torch.quantization import *
# import torch

# from collections import namedtuple
# import torch
# import torch.nn as nn

# based on 'fbgemm' in qconfig
weight_fake_quant_dummy = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                 quant_min=-128,
                                                 quant_max=127,
                                                 dtype=torch.qint8,
                                                 qscheme=torch.per_channel_symmetric,
                                                 reduce_range=False,
                                                 ch_axis=0)
disable_fake_quant(weight_fake_quant_dummy)


def get_qconfig(nlevel: int):
    return QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                     quant_min=0,
                                                     quant_max=nlevel - 1,
                                                     reduce_range=False),
                   weight=weight_fake_quant_dummy)
