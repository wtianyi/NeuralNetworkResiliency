from torch import nn


def children_of_class(module, target_class):
    """A generator yielding layers of the specified class in a module"""
    if isinstance(module, target_class):
        yield module
    else:
        try:
            for i in module.children():
                yield from children_of_class(i, target_class)
        except:
            pass


def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])


def loop_iterable(iterable):
    """https://stackoverflow.com/a/16638648"""
    while True:
        for elem in iterable:
            yield elem
        # print("Iterable exhausted")
