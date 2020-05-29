# -*- coding: utf-8 -*-

from torch.nn import Conv2d, Linear, LSTM, GRU, Conv3d
from torch.nn.utils import prune
from torch.quantization import (
    get_default_qconfig,
    prepare,
    convert,
    default_qconfig,
    quantize_dynamic,
)
from torch import qint8

# TODO: add pruning to bias
def modules_to_prune(net):
    modules = []
    for name, module in net.named_modules():
        if isinstance(module, Conv3d):
            modules.append((module, "weight"))
        if isinstance(module, Conv2d):
            modules.append((module, "weight"))
        if isinstance(module, Linear):
            modules.append((module, "weight"))
        if isinstance(module, LSTM) or isinstance(module, GRU):
            for i in range(module.num_layers):
                modules.append((module, "weight_ih_l" + str(i)))
                modules.append((module, "weight_hh_l" + str(i)))

    return modules[: (len(modules) - 1)]


def prune_net(net, threshold):
    parameters = modules_to_prune(net)
    prune.global_unstructured(
        parameters, pruning_method=prune.L1Unstructured, amount=threshold
    )
    for module, name in parameters:
        prune.remove(module, name)
    return net


def quant(net_i, scheme, trainer, quant_params=None):
    if scheme == "post":
        net_i.to("cpu")
        net_i.eval()
        net_i.qconfig = get_default_qconfig("fbgemm")
        net_i.fuse_model()
        prepare(net_i, inplace=True)
        _, net_i = trainer.evaluate(net_i, quant_mode=True)
        convert(net_i, inplace=True)
    elif scheme == "dynamic":
        net_i.to("cpu")
        net_i = quantize_dynamic(net_i, quant_params, dtype=qint8)
    elif scheme == "both":
        net_i.to("cpu")
        net_i.eval()
        net_i = quantize_dynamic(net_i, quant_params, dtype=qint8)
        net_i.qconfig = get_default_qconfig("fbgemm")
        net_i.fuse_model()
        prepare(net_i, inplace=True)
        _, net_i = trainer.evaluate(net_i, quant_mode=True)
        convert(net_i, inplace=True)
    else:
        pass
    return net_i
