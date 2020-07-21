from ax.core.parameter import RangeParameter, ParameterType, ChoiceParameter
from ax import SearchSpace
from torch import nn, rand, tensor, stack, float32, zeros, long as tlong, cat
from numpy import floor
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torch.autograd import Variable
from random import choice, random, randint

"""
The only difference between cnn and cnn2d_cost is the inclusion of max len in the search space in the latter case
also its inclusion in the morphing process
"""


def split_arrange_pad_n_pack(data, max_len):
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
                ).view(-1, 8, 8)
                img_sequence = stack(
                    [i[[7, 6, 5, 4, 3, 2, 1, 0], :] for i in img_sequence], axis=0
                )
                new_t_seqs.append(img_sequence)
                new_t_labels.append(lab)
        else:
            len_diff = max_len - len(seq)
            padding = zeros((len_diff, 8, 8))
            seq = tensor(seq).view(-1, 8, 8)
            seq = stack([i[[7, 6, 5, 4, 3, 2, 1, 0], :] for i in seq], axis=0)
            final_seq = cat((seq, padding), 0)
            new_t_seqs.append(final_seq)
            new_t_labels.append(lab)
    return stack(new_t_seqs), tensor(new_t_labels)


class DownsampleConv(nn.Module):
    """
    Convolution operation with two components: downsampling convolution first
    which reduces the number of convolution filters and a standard convolution

    Returns
    -------

    The overall convolution
    """

    def __init__(self, nin, nout, kernel_size, downsample):
        super(DownsampleConv, self).__init__()
        padding = (kernel_size - 1) // 2
        pointwise = nn.Conv2d(nin, round(nin * (1 - downsample)), kernel_size=1, padding=padding)
        seq = [
            nn.Conv2d(round(nin * (1 - downsample)), nout, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_features=nout, momentum=0.1),
            nn.ReLU(inplace=False),
        ]
        self.sole_conv = nn.Sequential(pointwise)
        self.fused_conv = nn.Sequential(*seq)
        self.out_channels = nout

    def forward(self, x):
        out = self.sole_conv(x)
        out = self.fused_conv(out)
        return out


class DepthwiseSeparableConv(nn.Module):
    """
    Convolution operation composed of two different convolutions: the first
    groups the convolution filters in as much groups as entry filters
    and the second is a pointwise convolution with kernel size of 1

    Returns
    ------
    The overall convolution
    """

    def __init__(self, nin, nout, kernel_size):
        super(DepthwiseSeparableConv, self).__init__()
        padding = (kernel_size - 1) // 2
        depthwise = nn.Conv2d(
            nin, nin * kernel_size, kernel_size=kernel_size, padding=padding, groups=nin)
        seq = [
            nn.Conv2d(nin * kernel_size, nout, kernel_size=1, padding=padding),
            nn.BatchNorm2d(num_features=nout, momentum=0.1),
            nn.ReLU(inplace=False),
        ]
        self.out_channels = nout
        self.sole_conv = nn.Sequential(depthwise)
        self.fused_conv = nn.Sequential(*seq)

    def forward(self, x):
        out = self.sole_conv(x)
        out = self.fused_conv(out)
        return out


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False),
        )


class LinearReLU(nn.Sequential):
    def __init__(self, in_neurons, out_neurons):
        super(LinearReLU, self).__init__(
            nn.Linear(in_neurons, out_neurons), nn.ReLU(inplace=False)
        )


# TODO: the net only accepts these parameters right now, the use is in utils_experiment.py, modify so its
# generalizable to ther aprameters, i dont kow kwargs or something
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

        for i in range(1, parametrization.get("num_fc_layers", 1) + 1):
            fc = self.create_fc_block(fc, i, self.odd_shape[0])

        # Final Layer
        self.fc = nn.Sequential(*fc)
        classifier = []
        classifier.append(
            nn.Linear(
                parametrization.get(
                    "fc_weights_layer_" + str(parametrization.get("num_fc_layers", 0)),
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
            1, self.parametrization.get("conv_" + str(j) + "_num_layers", 1) + 1
        ):
            conv_type = self.parametrization.get(
                "conv_" + str(j) + "_layer_" + str(i) + "_type", 0
            )

            if i == 1 and j != 1:
                index_l = self.parametrization.get(
                    "conv_" + str(j - 1) + "_num_layers", 1
                )
                index_b = j - 1
            else:
                index_l = i - 1
                index_b = j

            in_channels = self.parametrization.get(
                "conv_" + str(index_b) + "_layer_" + str(index_l) + "_filters", channels
            )
            if conv_type == 2:
                conv_layer = DepthwiseSeparableConv(
                    in_channels,
                    self.parametrization.get(
                        "conv_" + str(j) + "_layer_" + str(i) + "_filters", 6
                    ),
                    self.parametrization.get(
                        "conv_" + str(j) + "_layer_" + str(i) + "_kernel", 3
                    ),
                )
            elif conv_type == 1:
                conv_layer = DownsampleConv(
                    in_channels,
                    self.parametrization.get(
                        "conv_" + str(j) + "_layer_" + str(i) + "_filters", 6
                    ),
                    self.parametrization.get(
                        "conv_" + str(j) + "_layer_" + str(i) + "_kernel", 3
                    ),
                    self.parametrization.get(
                        "conv_" + str(j) + "_layer_" + str(i) + "_downsample", 0
                    ),
                )

            if conv_type == 0:
                conv.append(
                    ConvBNReLU(
                        in_channels,
                        self.parametrization.get(
                            "conv_" + str(j) + "_layer_" + str(i) + "_filters", 6
                        ),
                        self.parametrization.get(
                            "conv_" + str(j) + "_layer_" + str(i) + "_kernel", 3
                        ),
                    )
                )
            else:
                conv.append(conv_layer)
        if self.parametrization.get("downsample_input_depth_" + str(j + 1)):
            conv.append(
                nn.MaxPool2d(
                    (
                        self.parametrization.get(
                            "input_downsampling_rate_" + str(j + 1)
                        ),
                        self.parametrization.get(
                            "input_downsampling_rate_" + str(j + 1)
                        ),
                    )
                )
            )
        conv.append(nn.Dropout(self.parametrization.get("drop_" + str(j), 0.2)))
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
        out = out.mean([2, 3])
        if self.parametrization.get("num_fc_layers") > 0:
            out = self.fc(out)
        out = self.classifier(out)
        out = self.dequant(out)
        return out

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ["0", "1", "2"], inplace=True)
            if type(m) == LinearReLU:
                fuse_modules(m, ["0", "1"], inplace=True)
            if type(m) == DownsampleConv or type(m) == DepthwiseSeparableConv:
                for idx in range(len(m.fused_conv)):
                    if type(m.fused_conv[idx]) == nn.Conv2d:
                        fuse_modules(
                            m.fused_conv,
                            [str(idx), str(idx + 1), str(idx + 2)],
                            inplace=True,
                        )


def search_space():
    """
    Defines the network search space parameters and returns the search
    space object

    Returns
    ------
    Search space object

    """
    max_number_of_blocks = 2
    max_number_of_layers_per_block = 3
    max_fc_layers = 2
    params = []
    params.append(
        RangeParameter(
            name="num_conv_blocks",
            lower=1,
            upper=max_number_of_blocks,
            parameter_type=ParameterType.INT,
        )
    )
    for i in range(max_number_of_blocks):
        params.append(
            RangeParameter(
                name="downsample_input_depth_" + str(i + 1),
                lower=0,
                upper=1,
                parameter_type=ParameterType.INT,
            )
        )
        params.append(
            RangeParameter(
                name="input_downsampling_rate_" + str(i + 1),
                lower=2,
                upper=4,
                parameter_type=ParameterType.INT,
            )
        )

        params.append(
            RangeParameter(
                name="conv_" + str(i + 1) + "_num_layers",
                lower=1,
                upper=max_number_of_layers_per_block,
                parameter_type=ParameterType.INT,
            )
        )
        params.append(
            RangeParameter(
                name="drop_" + str(i + 1),
                lower=0.1,
                upper=0.9,
                parameter_type=ParameterType.FLOAT,
            )
        )
        for j in range(max_number_of_layers_per_block):
            params.append(
                RangeParameter(
                    name="conv_" + str(i + 1) + "_layer_" + str(j + 1) + "_filters",
                    lower=1,
                    upper=100,
                    parameter_type=ParameterType.INT,
                )
            )

            params.append(
                RangeParameter(
                    name="conv_" + str(i + 1) + "_layer_" + str(j + 1) + "_kernel",
                    lower=2,
                    upper=5,
                    parameter_type=ParameterType.INT,
                )
            )

            params.append(
                RangeParameter(
                    name="conv_" + str(i + 1) + "_layer_" + str(j + 1) + "_type",
                    parameter_type=ParameterType.INT,
                    lower=0,
                    upper=1
                )
            )

            params.append(
                RangeParameter(
                    name="conv_" + str(i + 1) + "_layer_" + str(j + 1) + "_downsample",
                    lower=0,
                    upper=0.5,
                    parameter_type=ParameterType.FLOAT,
                )
            )

    params.append(
        RangeParameter(
            name="num_fc_layers", lower=0, upper=2, parameter_type=ParameterType.INT
        )
    )

    for i in range(max_fc_layers):
        params.append(
            RangeParameter(
                name="fc_weights_layer_" + str(i + 1),
                lower=10,
                upper=200,
                parameter_type=ParameterType.INT,
            )
        )
        params.append(
            RangeParameter(
                name="drop_fc_" + str(i + 1),
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

    possible_types = list([0, 1, 2])
    new_layer_type = choice(possible_types)
    config["conv_" + str(block) + "_layer_" + str(layer) + "_type"] = int(
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


operations = {
    "num_fc_layers": num_fc_layers,
    "num_conv_blocks": num_conv_blocks,
    "layer_type": layer_type,
    "num_conv_filters": num_conv_filters,
    "kernel_size": kernel_size,
    "downsampling_rate": downsampling_rate,
    "num_fc_weights": num_fc_weights,
    "num_conv_layers": num_conv_layers,
    "change_max_len": change_max_len,
}
