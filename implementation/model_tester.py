#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:18:47 2019

@author: kostal
"""


from model import Trainer
from utils_experiment import get_experiment
from ax.modelbridge.factory import get_sobol
from cnn import search_space, Net
from load_data import prepare_cifar10, get_input_shape
from ptflops import get_model_complexity_info
from torchscope import scope

bits = 32
epochs = 1
input_shape = (3, 32, 32)
objectives = 1
std_obj = False
datasets, n_classes = prepare_cifar10()

exp = get_experiment(bits, epochs, objectives, True, datasets, n_classes, search_space(), Net)
sobol = get_sobol(search_space=exp.search_space)
exp.new_trial(
    sobol.gen(
        1, search_space=exp.search_space, optimization_config=exp.optimization_config
    )
)


trainer = Trainer(epochs)
net = Net(exp.trials[0].arm.parameters, classes=10, datasets=datasets)
macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=False,
                                           print_per_layer_stat=True, verbose=True)
scope(net, input_size=(3, 32, 32), batch_size=1, device='cpu')
net
net = trainer.train(net, exp.trials[0].arm.parameters)
print(trainer.evaluate(net))
