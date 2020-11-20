# Copied from torch/quantization/observer.py
# Modified to allow more than 256 levels (and TODO future quantization schemes)
# See also https://medium.com/@karanbirchahal/aggressive-quantization-how-to-run-mnist-on-a-4-bit-neural-net-using-pytorch-5703f3faa599
from typing import OrderedDict, Set, Type
import torch
from torch.nn.modules.module import _IncompatibleKeys
from torch.quantization.default_mappings import DEFAULT_QCONFIG_PROPAGATE_ALLOWED_LIST # DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.qat as nniqat
import torch.nn.quantized as nnq
from . import quantized_layers as mynnq

import torch.nn.quantized.dynamic as nnqd
import torch.nn.qat as nnqat

from .observers import *
from .noisy_layers import *


class CustomFakeQuantize(FakeQuantize):
    def __init__(self, observer=CustomMovingAverageMinMaxObserver, quant_min=0, quant_max=255, **observer_kwargs):
        super(CustomFakeQuantize, self).__init__(
            observer, quant_min, quant_max, **observer_kwargs)
        self.activation_post_process.set_qmin_qmax(quant_min, quant_max)
        self.observer_cls = observer.with_args(**observer_kwargs)

    def set_qmin_qmax(self, quant_min, quant_max):
        self.quant_min = quant_min
        self.quant_max = quant_max
        # need to preserve the device where the observer is
        device = next(self.activation_post_process.buffers()).device
        # need to re-initialize the observer
        self.activation_post_process = self.observer_cls()
        self.activation_post_process.set_qmin_qmax(quant_min, quant_max)
        self.activation_post_process.to(device)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if prefix + 'scale' in state_dict:
            self.scale = state_dict.pop(prefix + 'scale')
        if prefix + 'zero_point' in state_dict:
            self.zero_point = state_dict.pop(prefix + 'zero_point')
        Module._load_from_state_dict(
            self, state_dict, prefix, local_metadata, False,
            missing_keys, unexpected_keys, error_msgs)

# based on 'fbgemm' in qconfig
# weight_fake_quant_dummy =\
#     CustomFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
#                            quant_min=-128,
#                            quant_max=127,
#                            dtype=torch.qint8,
#                            qscheme=torch.per_channel_symmetric,
#                            reduce_range=False,
#                            ch_axis=0)


def get_activation_quant(nlevel: int, enable: bool = True):
    quant = CustomFakeQuantize.with_args(
        observer=CustomMovingAverageMinMaxObserver,
        dtype=torch.qint32,
        quant_min=0,
        quant_max=nlevel - 1,
        reduce_range=False)()
    if not enable:
        quant.disable_fake_quant()
        # disable_observer(quant) # Do we disable observer or not?
    return quant


def get_qconfig(nlevel_weight: int, nlevel_activation: int) -> QConfig:
    activation_observer = CustomHistogramObserver.with_args(
        quant_min=0, quant_max=nlevel_activation-1, reduce_range=False, dtype=torch.qint32
    )
    if nlevel_weight is None:
        weight_observer = lambda: None
    else:
        weight_observer = CustomMinMaxObserver.with_args(
            quant_min=0, quant_max=nlevel_weight-1,  # dtype=torch.qint8,
            dtype=torch.qint32,
            # qscheme=torch.per_channel_symmetric
            qscheme=torch.per_tensor_affine
        )
    return QConfig(activation=activation_observer, weight=weight_observer)


CUSTOM_MODULE_MAPPING = {
    nn.Linear: mynnq.Linear,
    nn.ReLU: mynnq.ReLU,
    nn.ReLU6: nnq.ReLU6,
    nn.Hardswish: nnq.Hardswish,
    nn.Conv1d: mynnq.Conv1d,
    nn.Conv2d: mynnq.Conv2d,
    nn.Conv3d: mynnq.Conv3d,
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
    nn.LayerNorm: nnq.LayerNorm,
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    # Wrapper Modules:
    nnq.FloatFunctional: nnq.QFunctional,
    # Intrinsic modules:
    nni.ConvReLU1d: nniq.ConvReLU1d,
    nni.ConvReLU2d: nniq.ConvReLU2d,
    nni.ConvReLU3d: nniq.ConvReLU3d,
    nni.LinearReLU: nniq.LinearReLU,
    nni.BNReLU2d: nniq.BNReLU2d,
    nni.BNReLU3d: nniq.BNReLU3d,
    nniqat.ConvReLU2d: nniq.ConvReLU2d,
    nniqat.LinearReLU: nniq.LinearReLU,
    nniqat.ConvBn2d: nnq.Conv2d,
    nniqat.ConvBnReLU2d: nniq.ConvReLU2d,
    # QAT modules:
    nnqat.Linear: nnq.Linear,
    nnqat.Conv2d: nnq.Conv2d,
    # nnqat.Hardswish: nnq.Hardswish,
    # Noisy modules:
    NoisyLinear: mynnq.Linear,
    NoisyConv2d: mynnq.Conv2d,
    # NoisyIdentity: nn.Identity,
    # Custom:
    # CustomFakeQuantize: nn.Identity,
}

CUSTOM_QCONFIG_PROPAGATE_WHITE_LIST = (DEFAULT_QCONFIG_PROPAGATE_ALLOWED_LIST | set(CUSTOM_MODULE_MAPPING.keys()))


def apply_quant_profile(
    quant_levels_dict,
    net: torch.nn.Module,
    swappable_module_list: Set[Type] = CUSTOM_MODULE_MAPPING,
    strict: bool = False
) -> _IncompatibleKeys:
    """
    Apply the quantization profile to the given network model
    """
    from fnmatch import fnmatchcase

    def find_match(key: str, net: torch.nn.Module):
        names = []
        matches = []
        for n, m in net.named_modules():
            if fnmatchcase(n, key):
                names.append(n)
                matches.append(m)
        if len(matches) == 0:
            return None, None
        else:
            return names, matches

    missing_keys = {n for n, _ in get_quantizable_layers(
        net, swappable_module_list)}
    unexpected_keys = []
    error_msgs = []
    for key in quant_levels_dict:
        names, matches = find_match(key, net)
        missing_keys = missing_keys - set(names)
        if matches is None:
            unexpected_keys.append(key)
            continue

        nlevel_weight, nlevel_out = quant_levels_dict[key]
        if nlevel_weight is None and nlevel_out is None:
            continue
        elif nlevel_weight is not None and nlevel_out is None:
            raise ValueError(
                "Weight quantization must be accompanied by output quantization")
        else:
            qconfig = get_qconfig(nlevel_weight, nlevel_out)
        for m in matches:
            m.qconfig = qconfig
    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in quant_levels_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in quant_levels_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading quant_levels_dict for {}:\n\t{}'.format(
            net.__name__, "\n\t".join(error_msgs)))
    return _IncompatibleKeys(missing_keys, unexpected_keys)


def get_quantizable_layers(
    net: torch.nn.Module,
    swappable_module_list: Set[Type] = CUSTOM_MODULE_MAPPING
) -> List[Tuple[str, Type]]:
    """
    Get the list of names of quantizable layers in a network model
    """
    result = []
    for n, m in net.named_modules():
        if type(m) in swappable_module_list:
            result.append((n, type(m)))
    return result
