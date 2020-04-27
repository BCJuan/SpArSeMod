# -*- coding: utf-8 -*-

from torch.nn import Conv2d, Linear, LSTM, GRU
from torch.nn.utils import prune

# TODO: add pruning for LSTM
# TODO: add pruning to bias
def modules_to_prune(net):
    modules = []
    for name, module in net.named_modules():
        if isinstance(module, Conv2d) or isinstance(module, Linear):
            modules.append((module, "weight"))
            modules.append((module, "bias"))
        if isinstance(module, LSTM) or isinstance(module, GRU):
            for i in range(module.num_layers):
                modules.append((module, "weight_ih_l" + str(i)))
                modules.append((module, "weight_hh_l" + str(i)))
                modules.append((module, "bias_ih_l" + str(i)))
                modules.append((module, "bias_hh_l" + str(i)))
                
    return modules[:-1]


def prune_net(net, threshold):
    parameters = modules_to_prune(net)
    prune.global_unstructured(
        parameters, pruning_method=prune.L1Unstructured, amount=threshold,
    )
    for module, name in parameters:
        prune.remove(module, name)
    return net
