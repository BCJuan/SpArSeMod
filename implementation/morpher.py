# -*- coding: utf-8 -*-

from numpy.random import choice, randint, random


class Morpher(object):
    """
    Retrieves configurations of trials and adds morphisms to them.
    """

    def __init__(self):
        self.operations = {
            "num_fc_layers": num_fc_layers,
            "num_conv_blocks": num_conv_blocks,
            "layer_type": layer_type,
            "num_conv_filters": num_conv_filters,
            "kernel_size": kernel_size,
            "downsampling_rate": downsampling_rate,
            "num_fc_weights": num_fc_weights,
            "num_conv_layers": num_conv_layers,
        }

    def retrieve_best_configurations(self, experiment, pareto_arms):
        self.configs = {}
        for arm in pareto_arms:
            params = experiment.arms_by_name[arm].parameters
            self.configs[arm] = params

    def apply_morphs(self, n_morphs=100, max_n_changes=10):
        configs_sub = choice(list(self.configs), size=n_morphs)
        n_changes_x_morph = randint(1, max_n_changes, size=n_morphs)
        new_configs = {}
        for i, (config, changes) in enumerate(zip(configs_sub, n_changes_x_morph)):
            ops = choice(list(self.operations), changes)
            new_configs[(str(i), config)] = self.configs[config]
            for k in ops:
                new_configs[(str(i), config)] = self.operations[k](
                    new_configs[(str(i), config)]
                )
        return new_configs


def num_fc_layers(config):
    """
    Changes in +-1 the number of fc layers in the main branch
    """
    if config["num_fc_layers"] == 0:
        config["num_fc_layers"] = config["num_fc_layers"] + 1
    elif config["num_fc_layers"] == 2:
        config["num_fc_layers"] = config["num_fc_layers"] - 1
    else:
        if random() < 0.5:
            config["num_fc_layers"] = config["num_fc_layers"] - 1
        else:
            config["num_fc_layers"] = config["num_fc_layers"] + 1
    return config


def num_conv_blocks(config):
    """
    Changes num conv blocks +-1
    """
    if config["num_conv_blocks"] == 1:
        config["num_conv_blocks"] = config["num_conv_blocks"] + 1
    else:
        config["num_conv_blocks"] = config["num_conv_blocks"] - 1

    return config


def layer_type(config):
    """
    Changes the layer type of a randomly picked con volution layer
    """
    block = choice(range(1, config["num_conv_blocks"] + 1))
    layer = choice(range(1, config["conv_" + str(block) + "_num_layers"] + 1))
    types = set(config["conv_" + str(block) + "_layer_" + str(layer) + "_type"])
    original_types = set(["Conv2D", "DownsampledConv2D", "SeparableConv2D"])
    possible_types = list(original_types - types)
    new_layer_type = choice(possible_types)
    config["conv_" + str(block) + "_layer_" + str(layer) + "_type"] = str(
        new_layer_type
    )
    return config


def num_conv_filters(config):
    """
    Changes the number of featyure maps in a randomly chosen convolution
    """
    block = choice(range(1, config["num_conv_blocks"] + 1))
    layer = choice(range(1, config["conv_" + str(block) + "_num_layers"] + 1))
    new_n_filters = randint(1, 50)
    config["conv_" + str(block) + "_layer_" + str(layer) + "_filters"] = new_n_filters
    return config


def kernel_size(config):
    """
    Changes the kernel in a randomly chosen convolution
    """
    block = choice(range(1, config["num_conv_blocks"] + 1))
    layer = choice(range(1, config["conv_" + str(block) + "_num_layers"] + 1))
    new_kernel_size = randint(2, 5)
    config["conv_" + str(block) + "_layer_" + str(layer) + "_kernel"] = new_kernel_size
    return config


def downsampling_rate(config):
    """
    Changes the downsampling rate of a randomly chosen convolution
    of type downsampled
    """
    sampling_rates = []
    for i in range(1, config["num_conv_blocks"] + 1):
        for j in range(1, config["conv_" + str(i) + "_num_layers"] + 1):
            if (
                config["conv_" + str(i) + "_layer_" + str(j) + "_type"]
                == "DownsampledConv2D"
            ):
                sampling_rates.append(
                    "conv_" + str(i) + "_layer_" + str(j) + "_downsample"
                )
    new_downsample = random() * 0.5
    if len(sampling_rates) > 0:
        layer_selected = choice(sampling_rates)
        config[layer_selected] = new_downsample
        return config
    else:
        return config


def num_fc_weights(config):
    """
    Changes the number of layers in a randomly chosen convolution block
    """
    if config["num_fc_layers"] != 0:
        layer = choice(range(1, config["num_fc_layers"] + 1))
        n_weights = config["fc_weights_layer_" + str(layer)]
        if n_weights < 6:
            config["fc_weights_layer_" + str(layer)] += 5
        else:
            if random() < 0.5:
                config["fc_weights_layer_" + str(layer)] += 1
            else:
                config["fc_weights_layer_" + str(layer)] -= 1

    return config


def num_conv_layers(config):
    """
    Changes the number of layers in a randomly chosen convolution block
    """
    block = choice(range(1, config["num_conv_blocks"] + 1))
    layers = config["conv_" + str(block) + "_num_layers"]
    if layers == 1:
        config["conv_" + str(block) + "_num_layers"] += 1
    elif layers == 3:
        config["conv_" + str(block) + "_num_layers"] -= 1
    else:
        if random() < 0.5:
            config["conv_" + str(block) + "_num_layers"] += 1
        else:
            config["conv_" + str(block) + "_num_layers"] -= 1
    return config
