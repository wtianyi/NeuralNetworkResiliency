start_epoch = 1

# Only for cifar-10
classes  = {
    'cifar10': ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
}

# Dataset stats
mean = {
    'mnist': ( 0.1307, ),
    'cifar10': ( 0.4914, 0.4822, 0.4465 ),
    'cifar100': ( 0.5071, 0.4867, 0.4408 ) 
}

std = {
    'mnist': ( 1.3081, ),
    'cifar10': ( 0.2023, 0.1994, 0.2010 ),
    'cifar100': ( 0.2675, 0.2565, 0.2761 ) 
}

# test parameters

std = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
mean = [-0.004, 0.0, 0.004]
mean_pos = [0.0, 0.004, 0.01, 0.02, 0.04, 0.06, 0.08]
mean_neg = [-0.08, -0.06, -0.04, -0.02, -0.01, -0.004, 0.0]
