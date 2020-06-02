from ax import SearchSpace, ParameterType, RangeParameter, ChoiceParameter
from random import random, choice, randint
from torch import nn as nn, rand as rand
from torch.autograd import Variable
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from numpy import ceil

# TODO: include batch norm and try to make it static
# TODO: if not, build it for dynamic

CONFIGURATION = {"max_conv_blocks": 4, "max_conv_layer_block": 3, "max_fc_layers": 2}


def split_arrange_pad_n_pack_3d(data, max_len):
    """
    Collate that splits the sequences of the cost dataset
    then arranges them in smaller sequences. When arranging them
    in smaller sequences also rearranges the values in them according
    to the Cost configuration (check COst readme regarding how the values
    are ordered). That is way there is this line
    `[i[[7, 6, 5, 4, 3, 2, 1, 0], :] for i in img_sequence]` and the fact that the 2 and 3
    dimensions are now (8, 8). This is due all to the cost data configuration
    To use as collate when dataloading cost and previously made a partial
    and the max len argument is fixed

    Args
    ---
    data: data argument that as collate needs for when called by the dataloader
    max_len: argument to fix the maximum lenght of the subsequences

    Returns
    ------
    pack_padded_data: each subsequence padded and packed for RNN consumption
    new_t_labels: label for each sequence
    """
    t_seqs = [tensor(sequence["signal"], dtype=float32) for sequence in data]
    labels = stack([tensor(label["label"], dtype=tlong) for label in data]).squeeze()
    new_t_seqs, new_t_labels = [], []
    for seq, lab in zip(t_seqs, labels):
        if len(seq) > max_len:
            n_seqs = int(floor(len(seq) // max_len))
            for i in range(n_seqs):
                img_sequence = tensor(
                    seq[(i * max_len) : (i * max_len + max_len), :]
                ).reshape(-1, 8, 8)
                img_sequence = stack(
                    [i[[7, 6, 5, 4, 3, 2, 1, 0], :] for i in img_sequence], axis=0
                )
                img_sequence = unsqueeze(img_sequence, 0)
                new_t_seqs.append(img_sequence)
                new_t_labels.append(lab)
        else:
            len_diff = max_len - len(seq)
            padding = zeros((len_diff, 8, 8))
            seq = tensor(seq).view(-1, 8, 8)
            seq = stack([i[[7, 6, 5, 4, 3, 2, 1, 0], :] for i in seq], axis=0)
            final_seq = cat((seq, padding), 0)
            final_seq = unsqueeze(final_seq, 0)
            new_t_seqs.append(final_seq)
            new_t_labels.append(lab)

    return stack(new_t_seqs), tensor(new_t_labels)


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
                    lower=1,
                    upper=3,
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
                upper=3,
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
            name="batch_size", lower=2, upper=256, parameter_type=ParameterType.INT
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


def layer_type(config):
    """
    Changes the layer type of a randomly picked con volution layer
    """
    block = choice(range(1, config["num_conv_blocks"] + 1))
    layer = choice(range(1, config["conv_block_" + str(block) + "_num_layers"] + 1))
    if config["block_" + str(block) + "_conv_" + str(layer) + "_type"] == 1:
        config["block_" + str(block) + "_conv_" + str(layer) + "_type"] = 0
    else:
        config["block_" + str(block) + "_conv_" + str(layer) + "_type"] = 1
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


def down_rate_change(config):
    """
    Changes the downsampling rate of a randomly chosen convolution
    of type downsampled
    """
    block = choice(range(1, config["num_conv_blocks"] + 1))
    new_down_sample = randint(1, 4)
    config["down_" + str(block)] = new_down_sample
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


operations = {
    "num_fc_layers": num_fc_layers,
    "num_conv_blocks": num_conv_blocks,
    "num_conv_filters": num_conv_filters,
    "kernel_size": kernel_size,
    "timekernel_size": timekernel_size,
    "down_rate_change": down_rate_change,
    "down_time_rate_change": down_time_rate_change,
    "drop_rate_change": drop_rate_change,
    "num_fc_weights": num_fc_weights,
    "num_conv_layers": num_conv_layers,
    "change_max_len": change_max_len,
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

        ####
        for i in range(1, parametrization.get("num_fc_layers", 1) + 1):
            fc = self.create_fc_block(fc, i, self.odd_shape[0])

        # Final Layer
        self.fc = nn.Sequential(*fc)
        classifier = []
        classifier.append(
            nn.Linear(
                parametrization.get(
                    "fc_weights_layer_" + str(parametrization.get("num_fc_layers")),
                    self.odd_shape[0],
                ),
                classes,
            )
        )
        self.classifier = nn.Sequential(*classifier)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

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
        out = self.quant(x)
        out = self._forward_features(out)
        out = out.mean([2, 3, 4])
        if self.parametrization.get("num_fc_layers") > 0:
            out = self.fc(out)
        out = self.classifier(out)
        out = self.dequant(out)
        return out

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU3d:
                fuse_modules(m, ["0", "1", "2"], inplace=True)
            if type(m) == LinearReLU:
                fuse_modules(m, ["0", "1"], inplace=True)
