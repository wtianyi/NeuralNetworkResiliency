from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

# import config as cf

import os, sys, time, datetime
import argparse
import random

from networks import *
from utils import *
import datasets
from training_functions import *
from torch.nn.utils import clip_grad_norm_

import wandb

import numpy as np
import json

all_start_time = datetime.datetime.now()

parser = argparse.ArgumentParser(
    description="HA-SGD Training",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    fromfile_prefix_chars="@",
)
parser.add_argument("--lr", default=0.01, type=float, help="learning_rate")
parser.add_argument("--num_epochs", "-n", default=200, type=int, help="num_epochs")
parser.add_argument(
    "--epochs_lr_decay", "-a", default=60, type=int, help="epochs_for_lr_decay"
)
parser.add_argument("--lr_decay_rate", default=0.2, type=float, help="lr_decay_rate")
parser.add_argument("--batch_size", "-s", default=200, type=int, help="batch size")
parser.add_argument(
    "--net_type",
    default="resnet",
    choices=["resnet", "wide_resnet", "lenet", "mlp"],
    type=str,
    help="model",
)
parser.add_argument("--depth", default=18, type=int, help="depth of model")
parser.add_argument("--widen_factor", default=10, type=int, help="width of model")
parser.add_argument("--dropout_rate", default=0.3, type=float, help="dropout_rate")
parser.add_argument(
    "--dataset",
    default="cifar10",
    type=str,
    choices=["cifar10", "cifar100", "mnist", "morse"],
    help="dataset = [cifar10/cifar100/mnist]",
)
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument(
    "--testOnly", "-t", action="store_true", help="Test mode with the saved model"
)
parser.add_argument(
    "--test_sample_num",
    type=int,
    default=20,
    help="The number of test runs per setting in testOnly mode",
)
parser.add_argument(
    "--load_model",
    type=str,
    default=None,
    help="Specify the model .pkl to load for testing/resuming training",
)
parser.add_argument(
    "--training_noise_type",
    type=str,
    default="gaussian",
    choices=["gaussian", "uniform"],
    help="noise_type = [gaussian/uniform]",
)
parser.add_argument(
    "--testing_noise_type",
    type=str,
    default="gaussian",
    choices=["gaussian", "uniform"],
    help="noise_type = [gaussian/uniform]",
)

parser.add_argument(
    "--training_noise",
    type=float,
    nargs="+",
    default=None,
    help="Set the training noise standard deviation",
)
parser.add_argument(
    "--testing_noise",
    type=float,
    nargs="+",
    default=None,
    help="Set the testing noise standard deviation",
)
parser.add_argument(
    "--training_noise_mean",
    type=float,
    nargs="+",
    default=None,
    help="Set the mean of the training noise",
)
parser.add_argument(
    "--testing_noise_mean",
    type=float,
    nargs="+",
    default=None,
    help="Set the mean of the testing noise in addition to the training mean",
)
parser.add_argument(
    "--testing_noise_mean_random_sign",
    action="store_true",
    help="Set the mean of the testing noise with random sign",
)

parser.add_argument(
    "--forward_samples", default=1, type=int, help="multi samples during forward"
)
parser.add_argument(
    "--tensorboard", action="store_true", help="Turn on the tensorboard monitoring"
)
parser.add_argument(
    "--regularization_type",
    type=str,
    choices=["l2", "l1"],
    default="l2",
    help="Set the type of regularization",
)
parser.add_argument(
    "--regularization",
    type=float,
    default=5e-4,
    help="Set the strength of regularization",
)
parser.add_argument("--seed", help="seed", type=int, default=42)
parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
parser.add_argument(
    "--device", type=int, nargs="+", default=None, help="Set the device(s) to use"
)
parser.add_argument(
    "--optim_type",
    default="SGD",
    type=str,
    choices=["SGD", "EntropySGD", "backpropless"],
    help="Set the type of optimizer",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, help="Set the momentum coefficient"
)
parser.add_argument("--nesterov", action="store_true", help="Use Nesterov momentum")
parser.add_argument(
    "--run_name",
    help="The name of this run (used for tensorboard)",
    type=str,
    default=None,
)

parser.add_argument(
    "--deficit_epochs",
    type=int,
    default=0,
    help="The number of initial epochs of deficit training",
)
# parser.add_argument('--test_with_std', action='store_true', help="fix mean, change std while testing")
# parser.add_argument('--test_with_mean', action='store_true', help="fix std, change mean while testing")

parser.add_argument(
    "--trajectory_dir",
    type=str,
    default=None,
    help="Set the directory for trajectory log",
)
parser.add_argument(
    "--trajectory_interval",
    type=int,
    default=10,
    help="Set the interval of trajectory logging",
)
parser.add_argument(
    "--test_quantization_levels",
    type=int,
    nargs="+",
    default=None,
    help="The levels of quantization during testing",
)
parser.add_argument(
    "--test_quantize_weights",
    action="store_true",
    help="Also quantize the weights with the specified quantization_levels",
)

if __name__ != "__main__":
    sys.exit(1)

args = parser.parse_args()
wandb.init(config=args)

# FIXME: this is assuming that `args.training_noise` always has only one element, and that `args.testing_noise` is a list of scalars
# FIXME: this is assuming that `args.training_noise_mean` always has only one element, and that `args.testing_noise_mean` is a list of scalars

if args.testing_noise is None:
    args.testing_noise = [None]
if args.training_noise is None:
    args.training_noise = [None]
if args.training_noise_mean is None:
    args.training_noise_mean = [None]
if args.testing_noise_mean is None:
    args.testing_noise_mean = [None]

if not args.testOnly:
    args.testing_noise = list(set(args.testing_noise + [args.training_noise[0]]))
    args.testing_noise_mean = list(
        set(args.testing_noise_mean + [args.training_noise_mean[0]])
    )

# test_std_list = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
# #test_mean_list = [0.0]
# #test_mean_pos = [0.0]
# #test_mean_neg = [0.0]

# # test_mean_list = [-0.08, -0.06, -0.04, -0.02, -0.01, -0.004, 0.0, 0.004, 0.01, 0.02, 0.04 , 0.06, 0.08]
# test_mean_list = [-0.004, 0.0, 0.004]
# test_mean_pos = [0.0, 0.004, 0.01, 0.02, 0.04, 0.06, 0.08]
# test_mean_neg = [-0.08, -0.06, -0.04, -0.02, -0.01, -0.004, 0.0]

# Set devices
device = torch.device("cpu")
use_cuda = torch.cuda.is_available() and not args.cpu

# Set random seeds
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
set_random_seed(args.seed, using_cuda=use_cuda)

if use_cuda:
    if args.device:
        device = torch.device("cuda:{:d}".format(args.device[0]))
    else:
        device = torch.device("cuda")
        args.device = range(torch.cuda.device_count())


###################################################
print("\n[Phase 1] : Data Preparation")
# start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, args.num_epochs, args.batch_size, args.optim_type
start_epoch, num_epochs, batch_size, optim_type = (
    0,
    args.num_epochs,
    args.batch_size,
    args.optim_type,
)

trainloader, testloader, num_classes = datasets.get_dataloader(
    args.dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=10,
    pin_memory=True,
)
dataset_meta = datasets.get_meta(args.dataset)
if "label_names" in dataset_meta:
    class_names = dataset_meta["label_names"]
elif "fine_label_names" in dataset_meta:
    class_names = dataset_meta["fine_label_names"]
else:
    class_names = [str(i) for i in range(num_classes)]

#####################################################
print("\n[Phase 2] : Model setup")
net, file_name = get_network(args, num_classes=num_classes)
print("| Building net...")
print(file_name)
net.apply(conv_init)
criterion = nn.CrossEntropyLoss()

if use_cuda:
    if torch.cuda.device_count() > 1 and len(args.device) > 1:
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda(device=device)
    cudnn.benchmark = True


def network_constructor():
    net, file_name = get_network(args, num_classes=num_classes)
    if use_cuda:
        if torch.cuda.device_count() > 1 and len(args.device) > 1:
            net = torch.nn.DataParallel(
                net, device_ids=range(torch.cuda.device_count())
            )
        net.cuda(device=device)
        cudnn.benchmark = True
    return net


train, test_with_std_mean = get_train_test_functions(
    trainloader, testloader, criterion, class_names, device
)

#######################################################
if args.testOnly:
    print("\n Test Only Mode")
    assert os.path.isdir("checkpoint"), "Error: No checkpoint directory found!"
    del net
    if args.load_model:
        checkpoint_file = args.load_model
    else:
        checkpoint_file = (
            "./checkpoint/"
            + args.dataset
            + "/"
            + args.training_noise_type
            + "/"
            + file_name
            + "_metric1.pkl"
        )
        print(
            f"checkpoint_file = {'./checkpoint/'+args.dataset+'/'+args.training_noise_type+'/'+file_name + '_metric1.pkl'}"
        )
    checkpoint = torch.load(checkpoint_file)

    test_acc_df = test_with_std_mean(
        network_constructor,
        checkpoint,
        noise_type=args.testing_noise_type,
        test_mean_list=args.testing_noise_mean,
        test_std_list=args.testing_noise,
        test_quantization_levels=args.test_quantization_levels,
        sample_num=args.test_sample_num,
        quantize_weights=args.test_quantize_weights,
    )

    test_acc_df["start_time"] = all_start_time

    with open(os.path.join("test", file_name + "_metric1.test"), "a") as f:
        f.write("\n")
        json.dump(
            {
                "args": vars(args),
                "test_acc_df": test_acc_df.to_json(),  # load with `pd.read_json()`
            },
            f,
        )

    sys.exit(0)

######################################################
print("\n[Phase 3] : Training model")
print("| Training Epochs = " + str(num_epochs))
print("| Initial Learning Rate = " + str(args.lr))
print("| Optimizer = " + str(optim_type))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.regularization,
    nesterov=args.nesterov,
)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=args.epochs_lr_decay, gamma=args.lr_decay_rate
)

writer = None

best_acc_1 = 0
best_acc_2 = 0
elapsed_time = 0
save_point = os.path.join("checkpoint", args.dataset, args.training_noise_type)

if args.trajectory_dir is not None:
    trajectory_logger = TrajectoryLogger(
        net, args.trajectory_dir, args.trajectory_interval
    )
else:
    trajectory_logger = None

wandb.watch(net, criterion, log="all", log_freq=1000)
trainloader.impair()
for epoch in range(num_epochs):
    print("\n=> Training Epoch #%d, LR=%.4f" % (epoch, optimizer.param_groups[0]["lr"]))
    if epoch >= args.deficit_epochs:
        trainloader.cure()
    start_time = time.time()
    # train
    if args.optim_type == "SGD":
        prepare_network_perturbation(
            net,
            noise_type=args.training_noise_type,
            fixtest=False,
            perturbation_level=args.training_noise,
            perturbation_mean=args.training_noise_mean,
        )

        train_acc, train_acc_5, train_loss = train(
            net,
            optimizer,
            args.forward_samples,
            trajectory_logger=trajectory_logger,
        )
        scheduler.step()
    elif args.optim_type == "EntropySGD":
        pass
    elif args.optim_type == "backpropless":
        pass
    save_model(net, save_point, file_name, args, 0)
    # test
    # net_test, _ = getNetwork(args, num_classes)
    # net_test.to(device)
    checkpoint_file = (
        "./checkpoint/"
        + args.dataset
        + "/"
        + args.training_noise_type
        + "/"
        + file_name
        + "_current.pkl"
    )
    checkpoint = torch.load(checkpoint_file)
    test_acc_df = test_with_std_mean(
        network_constructor,
        checkpoint,
        noise_type=args.testing_noise_type,
        test_mean_list=args.testing_noise_mean,
        test_std_list=args.testing_noise,
        deficit_list = [True, False],
        sample_num=1,
    )

    # TODO: not dealing with training & testing quant_level yet
    training_noise_stdev = (
        args.training_noise[0] if args.training_noise is not None else 0
    )
    training_noise_mean = (
        args.training_noise_mean[0] if args.training_noise_mean is not None else 0
    )
    metric_1 = test_acc_df[
        (
            test_acc_df["mean"] == training_noise_mean
        )  # & (test_acc_df['stdev'] == training_noise_stdev)
    ]["test_acc_avg"]
    assert len(metric_1) > 0, "No metric1 because not testing for the training case"
    best_metric_1 = metric_1.values[0]

    if best_metric_1 > best_acc_1:
        print(best_metric_1)
        save_model(
            net, save_point, file_name, args, 1, {"acc": best_metric_1, "epoch": epoch}
        )
        best_acc_1 = best_metric_1

    # if args.training_noise_mean is not None:
    #     if args.training_noise_mean[0] > 0:
    #         best_metric_2 = sum(test_acc_dict[i] for i in test_mean_pos) / len(test_mean_pos)
    #     elif args.training_noise_mean[0] < 0:
    #         best_metric_2 = sum(test_acc_dict[i] for i in test_mean_neg) / len(test_mean_neg)
    # else:
    #     best_metric_2 = test_acc_dict[0.0]

    # if best_metric_2 > best_acc_2:
    #     print (best_metric_2)
    #     save_model(net, save_point, args, 2, {"acc": best_metric_2, "epoch": epoch})
    #     best_acc_2 = best_metric_2

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print("| Elapsed time : %d:%02d:%02d" % (get_hms(elapsed_time)))
    print("| =====================================================")

print("\n[Phase 4] : Testing model")
print("* Test results : Acc@1 = {:.2%}".format(best_acc_1))
print("* Test results : Acc@1 = {:.2%}".format(best_acc_2))
