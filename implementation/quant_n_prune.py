# -*- coding: utf-8 -*-

from torch.nn import Conv2d, Linear
from torch.nn.utils import prune


def modules_to_prune(net):
    modules = []
    for name, module in net.named_modules():
        if isinstance(module, Conv2d) or isinstance(module, Linear):
            modules.append((module, "weight"))
    return modules[:-1]


def prune_net(net, threshold):
    parameters = modules_to_prune(net)
    prune.global_unstructured(
        parameters, pruning_method=prune.L1Unstructured, amount=threshold,
    )
    for module, name in parameters:
        prune.remove(module, name)
    return net
