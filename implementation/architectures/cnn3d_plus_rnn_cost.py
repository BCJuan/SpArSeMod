from ax import SearchSpace, ParameterType, RangeParameter, ChoiceParameter
from random import random, choice, randint
from torch import nn as nn, rand as rand
from torch.autograd import Variable
from numpy import ceil
from torch.quantization import QuantStub, DeQuantStub, fuse_modules

# TODO: include batch norm and try to make it static
# TODO: if not, build it for dynamic

CONFIGURATION = {
    "max_conv_blocks": 4,
    "max_conv_layer_block": 3,
    "max_fc_layers": 2,
    "max_rnn_layers": 5,
}


def search_space(config=CONFIGURATION):
    """
    Defines the network search space parameters and returns the search
    space object

    Returns
    ------
    Search space object

    """

    params = []
    ##### CONV BLOCKS ######################################################################
    params.append(
        RangeParameter(
            name="num_conv_blocks",
            parameter_type=ParameterType.INT,
            lower=1,
            upper=config["max_conv_blocks"],
        )
    )

    for i in range(1, config["max_conv_blocks"] + 1):
        params.append(
            RangeParameter(
                name="conv_block_" + str(i) + "_num_layers",
                parameter_type=ParameterType.INT,
                lower=1,
                upper=config["max_conv_layer_block"],
            )
        )
        for j in range(1, config["max_conv_layer_block"] + 1):
            params.append(
                RangeParameter(
                    name="block_" + str(i) + "_conv_" + str(j) + "_channels",
                    parameter_type=ParameterType.INT,
                    lower=5,
                    upper=100,
                )
            )
            params.append(
                RangeParameter(
                    name="block_" + str(i) + "_conv_" + str(j) + "_filtersize",
                    parameter_type=ParameterType.INT,
                    lower=2,
                    upper=5,
                )
            )
            params.append(
                RangeParameter(
                    name="block_" + str(i) + "_conv_" + str(j) + "_timefilter",
                    parameter_type=ParameterType.INT,
                    lower=5,
                    upper=25,
                )
            )
        params.append(
            RangeParameter(
                name="drop_" + str(i),
                lower=0.1,
                upper=0.8,
                parameter_type=ParameterType.FLOAT,
            )
        )
        params.append(
            RangeParameter(
                name="down_" + str(i),
                lower=1,
                upper=4,
                parameter_type=ParameterType.INT,
            )
        )
        params.append(
            RangeParameter(
                name="down_time_" + str(i),
                lower=5,
                upper=25,
                parameter_type=ParameterType.INT,
            )
        )
    ### RNN BLOCK ########################################################################
    params.append(
        RangeParameter(
            name="rnn_layers",
            parameter_type=ParameterType.INT,
            lower=1,
            upper=config["max_rnn_layers"],
        )
    )
    params.append(
        RangeParameter(
            name="neurons_layers", parameter_type=ParameterType.INT, lower=8, upper=512
        )
    )
    params.append(
        RangeParameter(
            name="rnn_dropout", parameter_type=ParameterType.FLOAT, lower=0.1, upper=0.8
        )
    )
    params.append(
        RangeParameter(
            name="cell_type", parameter_type=ParameterType.INT, lower=0, upper=1
        )
    )

    ### FC BLOCKS ########################################################################
    params.append(
        RangeParameter(
            name="num_fc_layers",
            lower=0,
            upper=config["max_fc_layers"],
            parameter_type=ParameterType.INT,
        )
    )

    for i in range(1, config["max_fc_layers"] + 1):
        params.append(
            RangeParameter(
                name="fc_weights_layer_" + str(i),
                lower=10,
                upper=200,
                parameter_type=ParameterType.INT,
            )
        )
        params.append(
            RangeParameter(
                name="drop_fc_" + str(i),
                lower=0.1,
                upper=0.5,
                parameter_type=ParameterType.FLOAT,
            )
        )

    ########################################################################
    params.append(
        RangeParameter(
            name="learning_rate",
            lower=0.0001,
            upper=0.01,
            parameter_type=ParameterType.FLOAT,
        )
    )
    params.append(
        RangeParameter(
            name="learning_gamma",
            lower=0.9,
            upper=0.99,
            parameter_type=ParameterType.FLOAT,
        )
    )
    params.append(
        RangeParameter(
            name="learning_step", lower=1, upper=10, parameter_type=ParameterType.INT
        )
    )
    ########################################################################
    params.append(
        RangeParameter(
            name="prune_threshold",
            lower=0.05,
            upper=0.9,
            parameter_type=ParameterType.FLOAT,
        )
    )
    params.append(
        RangeParameter(
            name="batch_size", lower=2, upper=8, parameter_type=ParameterType.INT
        )
    )
    params.append(
        RangeParameter(
            name="max_len", lower=25, upper=750, parameter_type=ParameterType.INT
        )
    )

    search_space = SearchSpace(parameters=params)

    return search_space


def num_fc_layers(config):
    """
    Changes in +-1 the number of fc layers in the main branch
    """
    if config["num_fc_layers"] == 0:
        config["num_fc_layers"] = config["num_fc_layers"] + 1
    elif config["num_fc_layers"] == CONFIGURATION["max_fc_layers"]:
        config["num_fc_layers"] = config["num_fc_layers"] - 1
    else:
        if random() < 0.5:
            config["num_fc_layers"] = config["num_fc_layers"] - 1
        else:
            config["num_fc_layers"] = config["num_fc_layers"] + 1
    return config


def timekernel_size(config):
    """
    Changes the kernel in a randomly chosen convolution
    """
    block = choice(range(1, config["num_conv_blocks"] + 1))
    layer = choice(range(1, config["conv_block_" + str(block) + "_num_layers"] + 1))
    new_kernel_size = randint(5, 25)
    config[
        "block_" + str(block) + "_conv_" + str(layer) + "_timefilter"
    ] = new_kernel_size
    return config


def down_time_rate_change(config):
    """
    Changes the downsampling rate of a randomly chosen convolution
    of type downsampled
    """
    block = choice(range(1, config["num_conv_blocks"] + 1))
    new_down_sample = randint(5, 25)
    config["down_time_" + str(block)] = new_down_sample
    return config


def num_conv_blocks(config):
    """
    Changes num conv blocks +-1
    """
    if config["num_conv_blocks"] == 1:
        config["num_conv_blocks"] = config["num_conv_blocks"] + 1
    elif config["num_conv_blocks"] == CONFIGURATION["max_conv_blocks"]:
        config["num_conv_blocks"] = config["num_conv_blocks"] - 1
    else:
        if random() < 0.5:
            config["num_conv_blocks"] = config["num_conv_blocks"] - 1
        else:
            config["num_conv_blocks"] = config["num_conv_blocks"] + 1

    return config


def num_conv_filters(config):
    """
    Changes the number of featyure maps in a randomly chosen convolution
    """
    block = choice(range(1, config["num_conv_blocks"] + 1))
    layer = choice(range(1, config["conv_block_" + str(block) + "_num_layers"] + 1))
    new_n_filters = randint(1, 100)
    config["block_" + str(block) + "_conv_" + str(layer) + "_channels"] = new_n_filters
    return config


def kernel_size(config):
    """
    Changes the kernel in a randomly chosen convolution
    """
    block = choice(range(1, config["num_conv_blocks"] + 1))
    layer = choice(range(1, config["conv_block_" + str(block) + "_num_layers"] + 1))
    new_kernel_size = randint(2, 5)
    config[
        "block_" + str(block) + "_conv_" + str(layer) + "_filtersize"
    ] = new_kernel_size
    return config


def down_rate_change(config):
    """
    Changes the downsampling rate of a randomly chosen convolution
    of type downsampled
    """
    block = choice(range(1, config["num_conv_blocks"] + 1))
    new_down_sample = randint(1, 4)
    config["down_" + str(block)] = new_down_sample
    return config


def drop_rate_change(config):
    """
    Changes the downsampling rate of a randomly chosen convolution
    of type downsampled
    """
    block = choice(range(1, config["num_conv_blocks"] + 1))
    new_down_sample = random()
    config["drop_" + str(block)] = new_down_sample
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
                config["fc_weights_layer_" + str(layer)] += 5
            else:
                config["fc_weights_layer_" + str(layer)] -= 5

    return config


def num_conv_layers(config):
    """
    Changes the number of layers in a randomly chosen convolution block
    """
    block = choice(range(1, config["num_conv_blocks"] + 1))
    layers = config["conv_block_" + str(block) + "_num_layers"]
    if layers == 1:
        config["conv_block_" + str(block) + "_num_layers"] += 1
    elif layers == 3:
        config["conv_block_" + str(block) + "_num_layers"] -= 1
    else:
        if random() < 0.5:
            config["conv_block_" + str(block) + "_num_layers"] += 1
        else:
            config["conv_block_" + str(block) + "_num_layers"] -= 1
    return config


def change_max_len(config):
    max_len = config["max_len"]
    if max_len > 250:
        config["max_len"] = config["max_len"] - 10
    elif max_len < 50:
        config["max_len"] = config["max_len"] + 10
    else:
        if random() < 0.5:
            config["max_len"] = config["max_len"] + 10
        else:
            config["max_len"] = config["max_len"] - 10
    return config


def change_layers(config):
    n_layers = config["rnn_layers"]
    if n_layers == 5:
        config["rnn_layers"] = config["rnn_layers"] - 1
    elif n_layers == 1:
        config["rnn_layers"] = config["rnn_layers"] + 1
    else:
        if random() < 0.5:
            config["rnn_layers"] = config["rnn_layers"] + 1
        else:
            config["rnn_layers"] = config["rnn_layers"] - 1
    return config


def change_n_neurons(config):
    n_neurons = config["neurons_layers"]
    if n_neurons > 512:
        config["neurons_layers"] = config["neurons_layers"] - 32
    elif n_neurons < 33:
        config["neurons_layers"] = config["neurons_layers"] + 32
    else:
        if random() < 0.5:
            config["neurons_layers"] = config["neurons_layers"] + 32
        else:
            config["neurons_layers"] = config["neurons_layers"] - 32
    return config


def change_dropout(config):
    dropout = config["rnn_dropout"]
    if dropout == 0.5:
        config["rnn_dropout"] = config["rnn_dropout"] - 0.05
    elif dropout < 0.1:
        config["rnn_dropout"] = config["rnn_dropout"] + 0.05
    else:
        if random() < 0.5:
            config["rnn_dropout"] = config["rnn_dropout"] + 0.05
        else:
            config["rnn_dropout"] = config["rnn_dropout"] - 0.05
    return config


operations = {
    "num_fc_layers": num_fc_layers,
    "num_conv_blocks": num_conv_blocks,
    "num_conv_filters": num_conv_filters,
    "kernel_size": kernel_size,
    "down_rate_change": down_rate_change,
    "drop_rate_change": drop_rate_change,
    "timekernel_size": timekernel_size,
    "down_time_rate_change": down_time_rate_change,
    "num_fc_weights": num_fc_weights,
    "num_conv_layers": num_conv_layers,
    "change_max_len": change_max_len,
    "change_layers": change_layers,
    "change_n_neurons": change_n_neurons,
    "change_dropout": change_dropout,
}


class LinearReLU(nn.Sequential):
    def __init__(self, in_neurons, out_neurons):
        super(LinearReLU, self).__init__(
            nn.Linear(in_neurons, out_neurons), nn.ReLU(inplace=False)
        )


class ConvBNReLU3d(nn.Sequential):
    def __init__(
        self, in_planes, out_planes, kernel_size=(20, 2, 2), stride=1, groups=1
    ):
        padding = tuple((k_size - 1) // 2 for k_size in kernel_size)
        super(ConvBNReLU3d, self).__init__(
            nn.Conv3d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=True,
            ),
            nn.BatchNorm3d(out_planes, momentum=0.1),
            nn.ReLU(inplace=False),
        )


class Net(nn.Module):
    """
    Module for the network building process. Consists of the initialization of
    the network, the forward pass, a function to compute the number of
    parameters after the last convolution and a global forward pass.
    """

    def __init__(self, parametrization, classes=10, input_shape=None):
        """
        Initializes the network according to the parametrization passed.

        Args
        ----
        parametrization:
            parameters for the network building
        classes:
            number of classes for the final fully connected layer
        input_shape:
            default shape for the input
        channels:
            number of channels of the input: 1 is grey image, 3 in color

        """
        super(Net, self).__init__()
        self.input_shape = input_shape

        channels = self.input_shape[0]

        self.parametrization = parametrization
        # COnvolution blocks
        conv_blocks = []
        for j in range(1, parametrization.get("num_conv_blocks", 1) + 1):
            conv_blocks.append(self.create_conv_block(j, channels))
        self.conv_blocks = nn.Sequential(*conv_blocks)
        # fully connected blocks
        fc = []
        # Main branch
        self.n_size, self.odd_shape = self._get_conv_output(
            self.parametrization.get("batch_size", 4),
            self.input_shape,
            self._forward_features,
        )
        ##### RNN
        self.layers = parametrization.get("rnn_layers", 1)
        if parametrization.get("cell_type"):
            cell = nn.LSTM
        else:
            cell = nn.GRU
        if self.layers > 1:
            dropout = parametrization.get("rnn_dropout", 0.1)
        else:
            dropout = 0
        self.cell = cell(
            self.odd_shape[-1],
            parametrization.get("neurons_layers", 64),
            batch_first=True,
            num_layers=self.layers,
            dropout=dropout,
        )
        ####
        for i in range(1, parametrization.get("num_fc_layers", 1) + 1):
            fc = self.create_fc_block(fc, i, parametrization.get("neurons_layers", 64))

        # Final Layer
        self.fc = nn.Sequential(*fc)
        classifier = []
        classifier.append(
            nn.Linear(
                parametrization.get(
                    "fc_weights_layer_" + str(parametrization.get("num_fc_layers", 0)),
                    parametrization.get("neurons_layers", 64),
                ),
                classes,
            )
        )
        self.classifier = nn.Sequential(*classifier)
        self.quant1 = QuantStub()
        self.dequant1 = DeQuantStub()
        self.quant2 = QuantStub()
        self.dequant2 = DeQuantStub()

    def create_fc_block(self, fc, i, n_size):
        linear = LinearReLU(
            self.parametrization.get("fc_weights_layer_" + str(i - 1), n_size),
            self.parametrization.get("fc_weights_layer_" + str(i)),
        )
        drop = nn.Dropout(self.parametrization.get("drop_fc_" + str(i)))
        fc.extend([linear, drop])
        return fc

    def create_conv_block(self, j, channels):
        conv = []
        for i in range(
            1, self.parametrization.get("conv_block_" + str(j) + "_num_layers") + 1
        ):

            if i == 1 and j != 1:
                index_l = self.parametrization.get(
                    "conv_block_" + str(j - 1) + "_num_layers"
                )
                index_b = j - 1
            else:
                index_l = i - 1
                index_b = j

            in_channels = self.parametrization.get(
                "block_" + str(index_b) + "_conv_" + str(index_l) + "_channels",
                channels,
            )

            out_channels = self.parametrization.get(
                "block_" + str(j) + "_conv_" + str(i) + "_channels"
            )
            conv.append(
                ConvBNReLU3d(
                    in_planes=in_channels,
                    out_planes=out_channels,
                    kernel_size=(
                        self.parametrization.get(
                            "block_" + str(j) + "_conv_" + str(i) + "_timefilter"
                        ),
                        self.parametrization.get(
                            "block_" + str(j) + "_conv_" + str(i) + "_filtersize"
                        ),
                        self.parametrization.get(
                            "block_" + str(j) + "_conv_" + str(i) + "_filtersize"
                        ),
                    ),
                )
            )
        conv.append(
            nn.AvgPool3d(
                (
                    self.parametrization.get("down_time_" + str(j)),
                    self.parametrization.get("down_" + str(j)),
                    self.parametrization.get("down_" + str(j)),
                )
            )
        )
        conv.append(nn.Dropout(self.parametrization.get("drop_" + str(j))))
        return nn.Sequential(*conv)

    def _get_conv_output(self, bs, shape, feature_function):
        """
        Makes a forward pass to compute the number of parameters for the
        initial fully connected layer

        Args
        ----
        shape:
            is the shape of the input for the conv block

        feature function:
            the function whihc makes the forward pass through the conv block

        Returns
        -------
        the number of fully connected elements of input for the first fc layer
        """
        input = Variable(rand(bs, *shape))
        output_feat = feature_function(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size, output_feat.shape[1:]

    def _forward_features(self, x):
        """
        Computes the forward step for the convolutional blocks

        Args
        ----
        x:
            The input for the convolution blocks
        conv_blocks:
            if true only make a forward through the first conv block only

        Returns
        -------
        The ouput of the network

        """
        for j in range(0, self.parametrization.get("num_conv_blocks", 1)):
            x = self.conv_blocks[j](x)
        return x

    def forward(self, x):
        """
        Global forward pass for both the convolution blocks and the fully
        connected layers.
        """
        out = self.quant1(x)
        out = self._forward_features(out)
        out = out.mean([2, 3])
        out = self.dequant1(out)
        cell_out, self.hidden = self.cell(out)
        if self.parametrization.get("cell_type"):
            out = self.hidden[0][self.layers - 1]
        else:
            out = self.hidden[self.layers - 1]
        out = self.quant2(out)
        if self.parametrization.get("num_fc_layers") > 0:
            out = self.fc(out)
        out = self.classifier(out)
        out = self.dequant2(out)
        return out

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU3d:
                fuse_modules(m, ["0", "1", "2"], inplace=True)
            if type(m) == LinearReLU:
                fuse_modules(m, ["0", "1"], inplace=True)
