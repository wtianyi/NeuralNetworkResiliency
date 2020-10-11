import torch
from torch import nn
import torch.nn.functional as F 
import numpy as np
from typing import Callable
from cached_property import cached_property_with_ttl

# TODO: maybe NoisyModule is more generic?
class NoisyLayer(nn.Module):
    noisy = True
    # noise_range = 1
    noise_type_sampler_dict = {
        "gaussian": torch.randn,
        "uniform": lambda *args, **kwargs: 2 * torch.rand(args, **kwargs) - 1
    }
    _fixtest_flag = False # TODO: redundant
    sigma = None
    mu = 0
    pre_sample = False
    record_delta: bool = False

    def __init__(self, *args,
                 use_range: bool=True,
                 pre_sample: bool=False,
                 match_range: bool=True,
                 **kwargs):
        if use_range is True:
            assert pre_sample is not True
        self.use_range = use_range
        self.match_range = match_range
        self.pre_sample = pre_sample
        self._noise_type = "gaussian"
        self.sampler = self.noise_type_sampler_dict[self._noise_type]
        self._noise_range = 1
        self.parameters_to_perturb = ["bias", "weight"]
        self.parameters_to_match = ["bias", "weight"]
        super().__init__(*args, **kwargs)

    # TODO: make it accept custom samplers
    @property
    def noise_type(self):
        return self._noise_type
    @noise_type.setter
    def noise_type(self, n_type: str):
        assert n_type in self.noise_type_sampler_dict, "Unknown noise_type"
        self._noise_type = n_type
        self.sampler = self.noise_type_sampler_dict[n_type]
    
    @property
    def noise_range(self):
        return self._noise_range
    @noise_range.setter
    def noise_range(self, noise_range: float):
        assert 0 <= noise_range and noise_range <= 1, "noise_range must be between [0,1]"
        self._noise_range = noise_range

    def sample_perturbation(self, param: torch.Tensor, num_samples: int, sampler: Callable=torch.randn) -> torch.Tensor:
        # TODO: for linear, if multiple samples, expand along axis 1; for conv2d, if multiple samples, expand along axis 1 as well.
        return sampler(param.size(), device=param.device)

    @property
    def fixtest_flag(self):
        return self._fixtest_flag
    @fixtest_flag.setter
    def fixtest_flag(self, fixtest_flag: bool):
        self._fixtest_flag = fixtest_flag
        if self._fixtest_flag:
            perturbation = self.get_perturbation()
            self.apply_perturbation(**perturbation)

    @cached_property_with_ttl(ttl=30)
    def param_range_dict(self) -> dict:
        return self.calc_param_range_dict()

    def refresh_param_range_dict(self):
        self.param_range_dict = self.calc_param_range_dict()

    def calc_param_range_dict(self) -> dict:
        epsilon = 1e-8
        range_dict = {}
        if not self.use_range:
            for param in self.parameters_to_perturb:
                if hasattr(self, param) and getattr(self, param) is not None:
                    range_dict[param] = 1
        else:
            for param in self.parameters_to_perturb: # bias and weight
                if hasattr(self, param) and getattr(self, param) is not None:
                    setattr(self, param + "_range", cal_range(getattr(self, param), self.noise_range))
                    range_dict[param] = max(getattr(self, param + "_range"), epsilon)
            if self.match_range: # only match bias and weight
                assert len(self.parameters_to_match) == 2, "Can only match ranges of 2 parameters"
                range_names_to_match = [param + "_range" for param in self.parameters_to_match]
                param_name_1, param_name_2 = self.parameters_to_match[0], self.parameters_to_match[1]
                if hasattr(self, range_names_to_match[0]) and hasattr(self, range_names_to_match[1]):
                    range_to_match = [getattr(self, rn) for rn in range_names_to_match]
                    if range_to_match[0] != 0 and range_to_match[1] != 0:
                        range_factor = range_to_match[0] / range_to_match[1]
                        self.merged_weight_range = cal_range(torch.cat((getattr(self, param_name_1).view(-1) / range_factor, getattr(self, param_name_2).view(-1))), self.noise_range)
                        range_dict[param_name_1] = self.merged_weight_range * range_factor # bias
                        range_dict[param_name_2] = self.merged_weight_range # weight
        return range_dict
    
    # @cached_property_with_ttl(ttl=30)
    @property
    def perturbation_stdev_dict(self) -> dict:
        stdev_dict = {}
        for param in self.param_range_dict:
            stdev_dict[param] = self.param_range_dict[param] * self.sigma
        return stdev_dict

    # @cached_property_with_ttl(ttl=30)
    @property
    def perturbation_mean_dict(self) -> dict:
        mean_dict = {}
        for param in self.param_range_dict:
            mean_dict[param] = self.param_range_dict[param] * self.mu
        return mean_dict

    def get_perturbation(self) -> dict:
        """Return the parameter perturbation.
        self.record_delta: If self.record_delta = True, set the relevant registered_buffer with rescaled perturbation
        """
        perturbation_dict = {}
        # pylint: disable=unsubscriptable-object,not-an-iterable
        with torch.no_grad():
            for param in self.param_range_dict:
                n = self.sample_perturbation(getattr(self, param), 1, sampler=self.sampler)
                perturbation_dict[param] = self.param_range_dict[param] * self.sigma * n + self.mu * self.param_range_dict[param]
                if self.record_delta:
                    assert self.noise_type == "gaussian", "delta values are only useful for Gaussian perturbation"
                    buffer_name = "delta_" + param
                    if not hasattr(self, buffer_name):
                        self.register_buffer(buffer_name, torch.empty_like(getattr(self, param)))
                    if self.mu is not 0:
                        # FIXME: what should this be?
                        if self.sigma:
                            # self.__dict__[buffer_name] = n / self.perturbation_stdev_dict[param] - self.mu / self.sigma ** 2
                            self.__dict__[buffer_name] = perturbation_dict[param] - self.mu / self.sigma ** 2
                    else:
                        self.__dict__[buffer_name] = perturbation_dict[param]

        return perturbation_dict

    def apply_perturbation(self, **perturbation_dict):
        """Apply the specified perturbations
        """
        for param in perturbation_dict:
            perturbation = perturbation_dict[param]
            getattr(self, param).data.add_(perturbation)
        return

    def forward(self, input):
        raise NotImplementedError

    def to_original(self) -> None:
        """Modify self.__class__ to pretend as non-noisy conterparts
        """
        raise NotImplementedError

class NoisyConv2d(NoisyLayer, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, mu=0, sigma=1, use_range=True, match_range=True):
        self.mu = mu
        self.sigma = sigma
        if bias:
            self.parameters_to_perturb = ["bias", "weight"]
        else:
            self.parameters_to_perturb = ["weight"]
            match_range = False

        super(NoisyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
              padding, dilation, groups, bias, use_range=use_range, match_range=match_range)

    def forward(self, input):
        if self.noisy and (self.sigma or self.mu) and (not self.fixtest_flag):
            perturbation_dict = self.get_perturbation()
            perturbed_weight = self.weight + perturbation_dict["weight"]
            perturbed_bias = self.bias + perturbation_dict["bias"] if self.bias is not None else None
            return F.conv2d(input, perturbed_weight, perturbed_bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def to_original(self):
        self.__class__ = nn.Conv2d
        self.noisy = False

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
           ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        # if self.sigma:
        s += ', sigma={sigma}'
        s += ', mu={mu}'
        return s.format(**self.__dict__)

# TODO: enable grouped unrolling?
class NoisyConv2dUnrolled(NoisyConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, mu=0, sigma=1, use_range=True, match_range=True):
        super(NoisyConv2dUnrolled, self).__init__(in_channels, out_channels, kernel_size, stride,
                                                padding, dilation, groups, bias, mu, sigma, use_range, match_range)
        self.output_size = None

    @property
    def fixtest_flag(self):
        return self._fixtest_flag
    @fixtest_flag.setter
    def fixtest_flag(self, fixtest_flag: bool):
        self._fixtest_flag = fixtest_flag
        if self._fixtest_flag:
            assert self.output_size is not None, "Need to specify output size (H, W) before setting fixtest"
            assert len(self.output_size) == 2, "self.output_size is expected to have two integers but has value {}".format(self.output_size)
            # prepare the unfold object
            self.fold_params = dict(kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
            self.unfold = nn.Unfold(**self.fold_params)
            # prepare the unrolled kernel
            num_out_locations = self.output_size[0] * self.output_size[1]
            # unrolled_weight dimensions: (out_channels, C * $\prod$ (kernel size), unrolling_instances)
            self.unrolled_weight = self.weight.view(self.weight.size(0),-1,1).expand(-1,-1, num_out_locations)
            self.unrolled_weight = self.unrolled_weight.detach() +\
                torch.randn_like(self.unrolled_weight) * self.sigma * self.perturbation_stdev_dict["weight"] +\
                self.mu * self.perturbation_stdev_dict["weight"]
            if self.bias is not None:
                # unrolled_bias dimensions: (batch, out_channels, unrolling_instances)
                self.unrolled_bias = self.bias.view(1,-1,1).expand(-1, -1, num_out_locations)
                self.unrolled_bias = self.unrolled_bias.detach() +\
                    torch.randn_like(self.unrolled_bias) * self.sigma * self.perturbation_stdev_dict["bias"] +\
                    self.mu * self.perturbation_stdev_dict["bias"]

    def forward(self, input):
        if self.noisy and (self.sigma or self.mu) and (not self.fixtest_flag):
            weight_perturb_stdev = self.perturbation_stdev_dict["weight"]
            output = F.conv2d(input, self.weight + self.mu * weight_perturb_stdev, self.bias, self.stride, self.padding, self.dilation, self.groups)
            if self.bias is not None:
                output += self.perturbation_stdev_dict["bias"] * self.sigma * torch.randn_like(output) + self.mu * self.perturbation_stdev_dict["bias"]
            perturb_stdev = weight_perturb_stdev * self.sigma * torch.sqrt(F.conv2d(input.detach() ** 2, torch.ones_like(self.weight), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups))
            assert output.size() == perturb_stdev.size()
            output += perturb_stdev * torch.randn_like(perturb_stdev)
            self.output_size = output.size()[-2:] # use the latest output to set size
            return output
        elif self.fixtest_flag:
            input_unf = self.unfold(input)
            try:
                # input_unf has dimensions (batch, C * $\prod$ (kernel size), unrolling_instances)
                # unrolled_weight dimensions: (out_channels, C * $\prod$ (kernel size), unrolling_instances)
                # output has dimensions: (batch, out_channels, unrolling_instances)
                output = torch.einsum("bjk,cjk->bck", input_unf, self.unrolled_weight)
            except:
                print(f"input size: {input.size()}")
                print(f"weight size: {self.weight.size()}")
                print(f"unfold input size: {input_unf.size()}")
                print(f"unrolled_weight size: {self.unrolled_weight.size()}")
                output = torch.einsum("bjk,cjk->bck", input_unf, self.unrolled_weight)
            if self.bias is not None:
                output += self.unrolled_bias
            output = output.view(output.size(0), output.size(1), self.output_size[0], self.output_size[1])
            return output
        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            self.output_size = output.size()[-2:] # use the latest output to set size
            return output

class NoisyLinear(NoisyLayer, nn.Linear):
    def __init__(self, in_features, out_features, bias=True, mu=0, sigma=1, use_range=True, match_range=True):
        self.mu = mu
        self.sigma = sigma
        super(NoisyLinear, self).__init__(in_features, out_features, bias, use_range=use_range, match_range=match_range)

    def forward(self, input):
        if self.noisy and (self.sigma or self.mu) and (not self.fixtest_flag):
            perturbation_dict = self.get_perturbation()
            perturbed_weight = self.weight + perturbation_dict["weight"]
            perturbed_bias = self.bias + perturbation_dict["bias"] if self.bias is not None else None
            return F.linear(input, perturbed_weight, perturbed_bias)
        else:
            return F.linear(input, self.weight, self.bias)

    def to_original(self):
        self.__class__ = nn.Linear
        self.noisy = False

    def extra_repr(self):
        s = super().extra_repr()
        s += ', sigma={sigma}'
        s += ', mu={mu}'
        return s.format(**self.__dict__)


# TODO: make this compatible with the get_perturbation & apply_perturbation
class NoisyIdentity(NoisyLayer, nn.Module):
    noisy = True
    def __init__(self, sigma=1):
        self.sigma = sigma
        self.fix_flag = False
        super(NoisyIdentity, self).__init__()

    def forward(self, x):
        if self.noisy and self.sigma and self.training:
            out = torch.ones_like(x) + torch.randn_like(x) * self.sigma
            return out * x

        elif self.noisy and self.sigma and (not self.training) and (not self.fix_flag):
            out = torch.ones_like(x) + torch.randn_like(x) * self.sigma
            self.register_buffer('out_fix', out)
            return out * x

        elif self.noisy and self.sigma and (not self.training) and self.fix_flag:
            if x.size() == self.out_fix.size():
                return self.out_fix * x
            else:
                return self.out_fix[:x.size(0)] * x                
        else:
            return x

    def to_original(self):
        self.__class__ = nn.Identity
        self.noisy = False

    def extra_repr(self):
        s = super().extra_repr()
        s += ', sigma={sigma}'
        return s.format(**self.__dict__)

class NoisyBN(NoisyLayer, nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, mu=0, sigma=1, use_range=True, match_range=True):
        self.mu = mu
        self.sigma = sigma
        self.num_features = num_features        
        # TODO: add noise to running_mean & running_var
        super(NoisyBN, self).__init__(num_features, eps, momentum, affine, track_running_stats, use_range=use_range, match_range=match_range)
        b_eff = torch.zeros(num_features)
        w_eff = torch.zeros(num_features, 1, 1, 1)
        if not hasattr(self, "b_eff"):
            self.register_buffer("b_eff", b_eff)
        else:
            self.b_eff = b_eff
        if not hasattr(self, "w_eff"):
            self.register_buffer("w_eff", w_eff)
        else:
            self.w_eff = w_eff
        self.parameters_to_match   = ["b_eff", "w_eff"]
        self.parameters_to_perturb = ["b_eff", "w_eff"]
    
    @property
    def fixtest_flag(self):
        return self._fixtest_flag
    @fixtest_flag.setter
    def fixtest_flag(self, fixtest_flag: bool):
        self._fixtest_flag = fixtest_flag
        if self._fixtest_flag:
            b_eff = self.bias - (self.running_mean * self.weight) / torch.sqrt(self.running_var + self.eps)
            w_eff = (self.weight / torch.sqrt(self.running_var + self.eps)).view(self.num_features,1,1,1)
            if not hasattr(self, "b_eff"):
                self.register_buffer("b_eff", b_eff)
            else:
                self.b_eff = b_eff
            if not hasattr(self, "w_eff"):
                self.register_buffer("w_eff", w_eff)
            else:
                self.w_eff = w_eff
            self.refresh_param_range_dict()
            self.apply_perturbation(**self.get_perturbation())

    def forward(self, input):
        self._check_input_dim(input)
        #if self.noisy and (self.sigma or self.mu) and (not self.fixtest_flag) and self.training:
        if not self.fixtest_flag: 
            bn_mean = input.mean(axis=(0,2,3))
            bn_var = input.var(axis=(0,2,3), unbiased=False)
            bn_weight, bn_bias = self.weight.detach(), self.bias.detach()
            self.b_eff = self.bias - (bn_mean * self.weight) / torch.sqrt(bn_var + self.eps)
            self.w_eff = (self.weight / torch.sqrt(bn_var + self.eps)).view(self.num_features,1,1,1)
            
            if self.noisy and (self.sigma or self.mu) and self.training:
                self.refresh_param_range_dict()  # refresh aggressively for BN
                perturbation_dict = self.get_perturbation()
                perturbed_w_eff = self.w_eff + perturbation_dict["w_eff"] if self.w_eff is not None else None
                perturbed_b_eff = self.b_eff + perturbation_dict["b_eff"] if self.b_eff is not None else None
                # This call of F.batch_norm is just to update the running_mean and running_var
                # TODO: checkout https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Normalization.cpp#L259 to replicate the update of the stats and save some computing here
                F.batch_norm(input, self.running_mean, self.running_var, bn_weight, bn_bias, self.training, self.momentum, self.eps)
                return F.conv2d(input, perturbed_w_eff, perturbed_b_eff, stride=1, groups=self.num_features)
            else:
                F.batch_norm(input, self.running_mean, self.running_var, bn_weight, bn_bias, self.training, self.momentum, self.eps)                
                return F.conv2d(input, self.w_eff, self.b_eff, stride=1, groups=self.num_features)
        else:
            #return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)
            return F.conv2d(input, self.w_eff, self.b_eff, stride=1, groups=self.num_features)

    def to_original(self):
        self.__class__ = nn.BatchNorm2d
        self.noisy = False

    def extra_repr(self):
        s = super().extra_repr()
        s += ', sigma={sigma}'
        s += ', mu={mu}'
        return s.format(**self.__dict__)

# TODO: enable grouped unrolling?
class NoisyBNUnrolled(NoisyBN):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, mu=0, sigma=1, use_range=True, match_range=True):
        super(NoisyBNUnrolled, self).__init__(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, mu=0, sigma=1, use_range=True, match_range=True)
        self.output_size = None

    @property
    def fixtest_flag(self):
        return self._fixtest_flag
    @fixtest_flag.setter
    def fixtest_flag(self, fixtest_flag: bool):
        self._fixtest_flag = fixtest_flag
        if self._fixtest_flag:
            assert self.output_size is not None, "Need to specify output size (H, W) before setting fixtest"
            assert len(self.output_size) == 2, "self.output_size is expected to have two integers but has value {}".format(self.output_size)

            b_eff = self.bias - (self.running_mean * self.weight) / torch.sqrt(self.running_var + self.eps)
            w_eff = (self.weight / torch.sqrt(self.running_var + self.eps)).view(self.num_features,1,1,1)

            if not hasattr(self, "b_eff"):
                self.register_buffer("b_eff", b_eff)
            else:
                self.b_eff = b_eff
            if not hasattr(self, "w_eff"):
                self.register_buffer("w_eff", w_eff)
            else:
                self.w_eff = w_eff

            perturbation_stdev_dict = self.perturbation_stdev_dict
            w_eff = w_eff.view(1,-1,1,1).expand(-1, -1, self.output_size[0], self.output_size[1])
            w_eff = w_eff + torch.randn_like(w_eff) * self.sigma * perturbation_stdev_dict["w_eff"] + self.mu * perturbation_stdev_dict["w_eff"]
            b_eff = b_eff.view(1,-1,1,1).expand(-1, -1, self.output_size[0], self.output_size[1])
            b_eff = b_eff + torch.randn_like(b_eff) * self.sigma * perturbation_stdev_dict["b_eff"] + self.mu * perturbation_stdev_dict["b_eff"]
            self.w_eff, self.b_eff = w_eff, b_eff

    def forward(self, input):
        if not self.fixtest_flag: 
            bn_mean = input.mean(axis=(0,2,3))
            bn_var = input.var(axis=(0,2,3), unbiased=False)
            bn_weight, bn_bias = self.weight.detach(), self.bias.detach()
            self.b_eff = self.bias - (bn_mean * self.weight) / torch.sqrt(bn_var + self.eps)
            self.w_eff = (self.weight / torch.sqrt(bn_var + self.eps)).view(self.num_features,1,1,1)
            
            if self.noisy and (self.sigma or self.mu) and self.training:
                shifted_w_eff = self.w_eff + self.perturbation_stdev_dict["w_eff"] * self.mu
                shifted_b_eff = self.b_eff + self.perturbation_stdev_dict["b_eff"] * self.mu
                output = F.conv2d(input, shifted_w_eff, shifted_b_eff, stride=1, groups=self.num_features)
                output += self.sigma * self.perturbation_stdev_dict["b_eff"] * torch.randn_like(output)
                output += self.sigma * self.perturbation_stdev_dict["w_eff"] * input * torch.randn_like(output)
                # This call of F.batch_norm is just to update the running_mean and running_var
                # TODO: checkout https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Normalization.cpp#L259 to replicate the update of the stats and save some computing here
                F.batch_norm(input, self.running_mean, self.running_var, bn_weight, bn_bias, self.training, self.momentum, self.eps)
                self.output_size = output.size()[-2:]
                return output
            else:
                F.batch_norm(input, self.running_mean, self.running_var, bn_weight, bn_bias, self.training, self.momentum, self.eps)                
                output = F.conv2d(input, self.w_eff, self.b_eff, stride=1, groups=self.num_features)
                self.output_size = output.size()[-2:]
                return output
        else:
            # print(f"input size: {input.size()}")
            # print(f"w_eff size: {self.w_eff.size()}")
            # print(f"b_eff size: {self.b_eff.size()}")
            output = input * self.w_eff + self.b_eff
            return output

def set_noisy(m, noisy=True):
    if isinstance(m, NoisyLayer):
        m.noisy = noisy
    # if isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear) or isinstance(m, NoisyIdentity) or isinstance(m, NoisyBN):
        # m.noisy = noisy

def set_clean(m):
    if isinstance(m, NoisyLayer):
        m.noisy = False

def set_gaussian_noise(m):
    if isinstance(m, NoisyLayer):
        m.noise_type = 'gaussian'

def set_uniform_noise(m):
    if isinstance(m, NoisyLayer):
        m.noise_type = 'uniform'

def set_noise_type(m, noise_type):
    if noise_type == "gaussian":
        set_gaussian_noise(m)
    elif noise_type == "uniform":
        set_uniform_noise(m)

def set_fixtest(m):
    if isinstance(m, NoisyLayer):
        m.fixtest_flag = True

def set_noisyid_fix(m):
    if isinstance(m, NoisyIdentity):
        m.fix_flag = True

def set_noisyid_unfix(m):
    if isinstance(m, NoisyIdentity):
        m.fix_flag = False    

def set_grad_with_delta(m):
    if isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear) or isinstance(m, NoisyBN):
        if m.sigma is None or m.sigma == 0:
            return
        m.weight.grad = m.delta_weight
        if hasattr(m, "bias"):
            m.bias.grad = m.delta_bias


def cal_range(weight: torch.Tensor, noise_range_ratio: float):
    assert noise_range_ratio <= 1 and noise_range_ratio > 0
    # TODO: with torch.no_grad():
    if noise_range_ratio == 1:
        return (torch.max(weight.detach()) - torch.min(weight.detach()))/2

    top_rank = max(int(weight.nelement() * (1-noise_range_ratio)/2) + 1, 1)
    bottom_rank = min(weight.nelement() - top_rank + 1, weight.nelement())
    tmp = weight.detach().view(-1)
    low_bound, _ = torch.kthvalue(tmp, top_rank)
    high_bound, _ = torch.kthvalue(tmp, bottom_rank)
    range_ = (high_bound - low_bound) / 2

    return range_
