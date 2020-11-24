import torch
import torch.quantization
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from networks import set_gaussian_noise, set_uniform_noise, set_clean, set_noisy, set_fixtest, disable_observer, disable_fake_quant, get_qconfig, CUSTOM_MODULE_MAPPING, CUSTOM_QCONFIG_PROPAGATE_WHITE_LIST, children_of_class, NoisyLayer, CustomFakeQuantize
import pandas as pd
import numpy as np
import argparse
from typing import List, Union, Iterable, Callable
import os, sys
from utils import create_dir, AverageMeter, accuracy, classwise_accuracy
import pandas as pd
from trajectory import TrajectoryLogger, TrajectoryLog
from itertools import product

import wandb

def prepare_network_perturbation(
        net, noise_type: str = 'gaussian', fixtest: bool = False,
        perturbation_level: Union[None, float, Iterable[float]] = None,
        perturbation_mean: Union[None, float, Iterable[float]] = None):
    """Set the perturbation and quantization of the network in-place
    """
    if noise_type == 'gaussian':
        net.apply(set_gaussian_noise)
        if isinstance(net, nn.DataParallel):
            net.module.set_sigma_list(perturbation_level)
            net.module.set_mu_list(perturbation_mean)
        else:
            net.set_sigma_list(perturbation_level)
            net.set_mu_list(perturbation_mean)
    elif noise_type == 'uniform':
        net.apply(set_uniform_noise)
        if isinstance(net, nn.DataParallel):
            net.module.set_sigma_list(1)
        else:
            net.set_sigma_list(1)

    if fixtest:
        net.apply(set_fixtest)


def prepare_network_quantization(
        net, num_quantization_levels: int, calibration_dataloader: torch.utils.data.DataLoader,
        qat: bool = False, num_calibration_batchs: int = 10):  # The last two arguments are redundant for now
    if num_quantization_levels is None:
        return
    # Specify quantization configuration
    net.set_quantization_level(num_quantization_levels)
    net.enable_quantization(False)
    # Calibrate with the test set
    net.eval()
    device = next(net.parameters()).device
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(calibration_dataloader):
            inputs, targets = inputs.to(
                device=device), targets.to(device=device)
            outputs = net(inputs)
    print('Post Training Quantization: Calibration done')
    net.enable_quantization()
    for quant in children_of_class(net, CustomFakeQuantize):
        quant.disable_observer()


def quantize_network(
    net: nn.Module, num_weight_quant_levels: int, num_activation_quant_levels: int,
    calibration_dataloader: torch.utils.data.DataLoader
):
    net.qconfig = get_qconfig(num_weight_quant_levels, num_weight_quant_levels)

    for noisy_layer in children_of_class(net, NoisyLayer):
        noisy_layer.to_original()
    for activation_quant in children_of_class(net, CustomFakeQuantize):
        disable_observer(activation_quant)
        disable_fake_quant(activation_quant)
    torch.quantization.prepare(
        net, inplace=True,
        allow_list=CUSTOM_QCONFIG_PROPAGATE_WHITE_LIST
    )
    # Calibrate with the given set
    net.eval()
    device = next(net.parameters()).device
    with torch.no_grad():
        for inputs, _ in calibration_dataloader:
            inputs = inputs.to(device=device)
            # targets = targets.to(device=device)
            outputs = net(inputs)
    torch.quantization.convert(
        net, inplace=True,
        # modify below to choose whether to use custom quantized layers
        mapping=CUSTOM_MODULE_MAPPING
    )

    # print('Quantization Config:', net.qconfig)
    # if qat:
    #     torch.quantization.prepare_qat(net, inplace=True)
    #     test(net, -1, calibration_dataloader)
    # else:
    #     torch.quantization.prepare(net, inplace=True)
    #     # Calibrate first
    #     print('Post Training Quantization Prepare: Inserting Observers')

    #     # Calibrate with the test set TODO: use the training set to calibrate
    #     test(net, -1, calibration_dataloader)
    #     print('Post Training Quantization: Calibration done')

    #     # Convert to quantized model
    #     torch.quantization.convert(net, inplace=True)
    #     print('Post Training Quantization: Convert done')


class Clipper(dict):  # inherit dict to be serializable
    """A scheduler for grad clipping
    """
    def __init__(self, max_norm: float = None, decay_factor: float = 1, decay_interval: int = None, max_decay_times: int = None):
        assert max_norm is None or max_norm > 0
        assert decay_factor > 0 and decay_factor <= 1
        assert decay_interval is None or decay_interval > 0
        assert max_decay_times is None or max_decay_times > 0
        self.clip_function = clip_grad_norm_
        self.max_norm = max_norm
        self.steps = 0
        self.decay_interval = np.inf if decay_interval is None else decay_interval
        self.decay_factor = decay_factor
        self.max_decay_times = np.inf if max_decay_times is None else max_decay_times
        self.decay_counter = 0
        super(Clipper, self).__init__(
            self,
            clip_function=self.clip_function.__name__,
            max_norm=self.max_norm,
            decay_interval=self.decay_interval,
            decay_factor=self.decay_factor,
            max_decay_times=self.max_decay_times,
        )

    def step(self):
        self.steps += 1
        if self.steps % self.decay_interval == 0 and self.decay_counter < self.max_decay_times:
            self.decay_counter += 1
            self.max_norm *= self.decay_factor

    def clip(self, parameters):
        if self.max_norm is not None:
            self.clip_function(parameters, self.max_norm)

    def __str__(self):
        keys = [
            "clip_function", "max_norm", "decay_interval", "decay_factor", "max_decay_times"
        ]
        string = ", ".join(["{}={}".format(k, getattr(self, k)) for k in keys])
        return "Clipper(" + string + ")"

    @staticmethod
    def from_str(string: str):
        """The parser for the grad_clip"""
        if string is None:
            return Clipper()

        toks = string.split(":")
        try:
            if len(toks) == 1:
                clipper = Clipper(float(toks[0]))
            elif len(toks) == 3:
                clipper = Clipper(float(toks[0]), float(toks[1]), int(toks[2]))
            elif len(toks) == 4:
                clipper = Clipper(float(toks[0]), float(
                    toks[1]), int(toks[2]), int(toks[3]))
            else:
                msg = "Required format: <init_max_norm>[:<decay_factor>:<decay_interval>[:<max_decay_count>]]"
                raise argparse.ArgumentTypeError(msg)
        except Exception as e:
            msg = "Required format: <init_max_norm>[:<decay_factor>:<decay_interval>[:<max_decay_count>]]"
            raise argparse.ArgumentTypeError(msg)
        return clipper

# class_names, trainloader, testloader, device, forward_samples,
#
# print("\n=> Training Epoch #%d, LR=%.4f" % (epoch, optimizer.param_groups[0]["lr"]))
def get_train_test_functions(
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion,
    class_names: List[str],
    device,
):
    epoch = 0
    global_step = 0
    num_classes = len(class_names)

    def train(
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        forward_samples: int,
        clipper: Clipper = None,
        trajectory_logger: TrajectoryLogger = None,
    ):
        nonlocal global_step, epoch
        net.train()
        net.apply(set_noisy)

        train_loss, acc, acc5 = AverageMeter(), AverageMeter(), AverageMeter()
        class_acc = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = (
                inputs.to(device),
                targets.to(device),
            )  # GPU settings

            optimizer.zero_grad()
            for _ in range(forward_samples):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                train_loss.update(loss.item(), inputs.size(0))
                loss.backward()

            for p in net.parameters():
                p.grad.data.mul_(1 / forward_samples)

            optimizer.step()
            if trajectory_logger is not None:
                trajectory_logger.add_param_log(net, global_step)
                trajectory_logger.add_grad_log(net, global_step)
                trajectory_logger.commit()

            optimizer.step()  # Optimizer update
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            class_acc_, class_counts_ = classwise_accuracy(
                outputs, targets, num_classes, 1
            )
            acc.update(prec1, targets.size(0))
            acc5.update(prec5, targets.size(0))
            class_acc.update(class_acc_, class_counts_)

            sys.stdout.write("\r")
            sys.stdout.write(
                "| Iter[{:3d}/{:3d}]\t\tLoss: {:.4f} Acc@1: {:.3%}".format(
                    batch_idx + 1, len(train_loader), train_loss.avg, acc.avg,
                )
            )
            sys.stdout.flush()

            wandb.log(
                {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "train_acc": prec1,
                    "train_top5_acc": prec5,
                    "data_state": getattr(train_loader, "state", None),
                    # "examples": [wandb.Image(inputs[0].detach().cpu().numpy().transpose([1,2,0]), caption="Input Sample")]
                },
                step=global_step,
            )
            global_step += 1

        wandb.log(
            {
                f"prec/{class_name}": p
                for class_name, p in zip(class_names, class_acc.avg)
            },
            step=global_step,
        )
        epoch += 1
        return acc.avg, acc5.avg, train_loss.avg

    def test(net, dataloader):
        net.eval()
        test_loss, acc, acc5 = AverageMeter(), AverageMeter(), AverageMeter()
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = (
                    inputs.to(device),
                    targets.to(device),
                )
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss.update(loss.item(), targets.size(0))
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                acc.update(prec1, targets.size(0))
                acc5.update(prec5, targets.size(0))

        print(
            "\n| Validation Epoch #{:d}\t\t\tLoss: {:.4f} Acc@1: {:.2%}".format(
                epoch, loss.item(), acc.avg
            )
        )

        return acc.avg, acc5.avg

    def test_with_std_mean(
        network_constructor: Callable,
        checkpoint,
        noise_type="gaussian",
        test_mean_list=[None],
        test_std_list=[None],
        test_quantization_levels=[None],
        quantize_weights: bool = False,
        deficit_list = [None],
        sample_num: int = 1,
    ):
        if test_std_list is None:
            test_std_list = [None]
        if test_mean_list is None:
            test_mean_list = [None]
        if test_quantization_levels is None:
            test_quantization_levels = [None]
        results = []
        for stdev, mean, quant_levels, deficit in product(
            test_std_list,
            test_mean_list,
            test_quantization_levels,
            deficit_list
        ):
            if deficit is True:
                test_loader.impair()
            elif deficit is False:
                test_loader.cure()
            def prepare_and_test():
                net = network_constructor()
                net.load_state_dict(checkpoint["state_dict"], strict=False)
                net.to(device)
                net.eval()
                net.apply(set_noisy)
                prepare_network_perturbation(
                    net=net,
                    noise_type=noise_type,
                    fixtest=True,
                    perturbation_level=stdev,
                    perturbation_mean=mean,
                )
                if quantize_weights:
                    quantize_network(
                        net=net,
                        num_weight_quant_levels=quant_levels,
                        num_activation_quant_levels=quant_levels,
                        calibration_dataloader=train_loader,
                    )
                else:
                    prepare_network_quantization(
                        net=net,
                        num_quantization_levels=quant_levels,
                        calibration_dataloader=train_loader,
                        qat=False,
                    )
                test_acc, test_acc_5 = test(net, test_loader)
                return test_acc.cpu().item(), test_acc_5.cpu().item()

            print(
                f"| test noise stdev: {stdev}, test noise mean: {mean},"
                f" test quant levels: {quant_levels}"
            )
            acc_tuple_list = [prepare_and_test() for _ in range(sample_num)]
            test_acc_list, test_acc5_list = zip(*acc_tuple_list)
            results.append(
                {
                    "stdev": stdev,
                    "mean": mean,
                    "quant_levels": quant_levels,
                    "data_state": test_loader.state,
                    "test_acc": test_acc_list,
                    "test_acc5": test_acc5_list,
                }
            )
        df = pd.DataFrame(results)
        df["test_acc_avg"] = df["test_acc"].apply(np.mean)
        df["test_acc5_avg"] = df["test_acc5"].apply(np.mean)
        df = df.fillna(0)
        test_table = wandb.Table(dataframe=df)
        wandb.log({"test_table": test_table}, step=global_step)
        return df

    return train, test_with_std_mean


def save_model(
    net: nn.Module,
    save_point: str,
    file_name: str,
    args,
    metric=1,
    stats_dict: dict = None,
):
    state = {"state_dict": net.state_dict(), "args": args}
    if stats_dict is not None:
        state.update(stats_dict)
    create_dir(save_point)
    if metric == 1:
        save_file = os.path.join(save_point, file_name + "_metric1.pkl")
        torch.save(state, save_file)
        print(f"| Saved Best model to \n{save_file}\nstats = {stats_dict}")
    elif metric == 2:
        save_file = os.path.join(save_point, file_name + "_metric2.pkl")
        torch.save(state, save_file)
        print(f"| Saved Best model to \n{save_file}\nstats = {stats_dict}")

    save_file = os.path.join(save_point, file_name + "_current.pkl")
    torch.save(state, save_file)
    print(f"| Saved Current model to \n {save_file}")
