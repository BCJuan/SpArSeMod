from ax.core.parameter import RangeParameter, ParameterType, ChoiceParameter
from ax import SearchSpace
from torch import nn, rand
from load_data import get_input_shape
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torch.autograd import Variable


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
        pointwise = nn.Conv2d(nin, round(nin * (1 - downsample)), kernel_size=1)
        seq = [
            nn.Conv2d(round(nin * (1 - downsample)), nout, kernel_size=kernel_size),
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
        depthwise = nn.Conv2d(
            nin, nin * kernel_size, kernel_size=kernel_size, padding=1, groups=nin
        )
        seq = [
            nn.Conv2d(nin * kernel_size, nout, kernel_size=1),
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
                "conv_" + str(j) + "_layer_" + str(i) + "_type", "Conv2D"
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
            if conv_type == "SeparableConv2D":
                conv_layer = DepthwiseSeparableConv(
                    in_channels,
                    self.parametrization.get(
                        "conv_" + str(j) + "_layer_" + str(i) + "_filters", 6
                    ),
                    self.parametrization.get(
                        "conv_" + str(j) + "_layer_" + str(i) + "_kernel", 3
                    ),
                )
            elif conv_type == "DownsampledConv2D":
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

            if conv_type == "Conv2D":
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
    param0 = RangeParameter(
        name="downsample_input_depth_1",
        lower=0,
        upper=1,
        parameter_type=ParameterType.INT,
    )
    param1 = RangeParameter(
        name="input_downsampling_rate_1",
        lower=2,
        upper=4,
        parameter_type=ParameterType.INT,
    )
    #########################################################################
    param2 = RangeParameter(
        name="downsample_input_depth_2",
        lower=0,
        upper=1,
        parameter_type=ParameterType.INT,
    )
    param3 = RangeParameter(
        name="input_downsampling_rate_2",
        lower=2,
        upper=4,
        parameter_type=ParameterType.INT,
    )
    ##########################################################################
    param7 = RangeParameter(
        name="num_conv_blocks", lower=1, upper=2, parameter_type=ParameterType.INT
    )
    #########################################################################
    param8 = RangeParameter(
        name="conv_1_num_layers", lower=1, upper=3, parameter_type=ParameterType.INT
    )
    param9 = RangeParameter(
        name="conv_1_layer_1_filters",
        lower=1,
        upper=100,
        parameter_type=ParameterType.INT,
    )
    param10 = RangeParameter(
        name="conv_1_layer_2_filters",
        lower=1,
        upper=100,
        parameter_type=ParameterType.INT,
    )
    param11 = RangeParameter(
        name="conv_1_layer_3_filters",
        lower=1,
        upper=100,
        parameter_type=ParameterType.INT,
    )
    param12 = RangeParameter(
        name="conv_1_layer_1_kernel", lower=2, upper=5, parameter_type=ParameterType.INT
    )
    param13 = RangeParameter(
        name="conv_1_layer_2_kernel", lower=2, upper=5, parameter_type=ParameterType.INT
    )
    param14 = RangeParameter(
        name="conv_1_layer_3_kernel", lower=2, upper=5, parameter_type=ParameterType.INT
    )
    param15 = ChoiceParameter(
        name="conv_1_layer_1_type",
        parameter_type=ParameterType.STRING,
        values=["Conv2D", "DownsampledConv2D", "SeparableConv2D"],
    )
    param16 = ChoiceParameter(
        name="conv_1_layer_2_type",
        parameter_type=ParameterType.STRING,
        values=["Conv2D", "DownsampledConv2D", "SeparableConv2D"],
    )
    param17 = ChoiceParameter(
        name="conv_1_layer_3_type",
        parameter_type=ParameterType.STRING,
        values=["Conv2D", "DownsampledConv2D", "SeparableConv2D"],
    )
    param18 = RangeParameter(
        name="conv_1_layer_1_downsample",
        lower=0,
        upper=0.5,
        parameter_type=ParameterType.FLOAT,
    )
    param19 = RangeParameter(
        name="conv_1_layer_2_downsample",
        lower=0,
        upper=0.5,
        parameter_type=ParameterType.FLOAT,
    )
    param20 = RangeParameter(
        name="conv_1_layer_3_downsample",
        lower=0,
        upper=0.5,
        parameter_type=ParameterType.FLOAT,
    )
    #########################################################################
    param21 = RangeParameter(
        name="conv_2_num_layers", lower=1, upper=3, parameter_type=ParameterType.INT
    )
    param22 = RangeParameter(
        name="conv_2_layer_1_filters",
        lower=1,
        upper=100,
        parameter_type=ParameterType.INT,
    )
    param23 = RangeParameter(
        name="conv_2_layer_2_filters",
        lower=1,
        upper=100,
        parameter_type=ParameterType.INT,
    )
    param24 = RangeParameter(
        name="conv_2_layer_3_filters",
        lower=1,
        upper=100,
        parameter_type=ParameterType.INT,
    )
    param25 = RangeParameter(
        name="conv_2_layer_1_kernel", lower=2, upper=5, parameter_type=ParameterType.INT
    )
    param26 = RangeParameter(
        name="conv_2_layer_2_kernel", lower=2, upper=5, parameter_type=ParameterType.INT
    )
    param27 = RangeParameter(
        name="conv_2_layer_3_kernel", lower=2, upper=5, parameter_type=ParameterType.INT
    )
    param28 = ChoiceParameter(
        name="conv_2_layer_1_type",
        parameter_type=ParameterType.STRING,
        values=["Conv2D", "DownsampledConv2D", "SeparableConv2D"],
    )
    param29 = ChoiceParameter(
        name="conv_2_layer_2_type",
        parameter_type=ParameterType.STRING,
        values=["Conv2D", "DownsampledConv2D", "SeparableConv2D"],
    )
    param30 = ChoiceParameter(
        name="conv_2_layer_3_type",
        parameter_type=ParameterType.STRING,
        values=["Conv2D", "DownsampledConv2D", "SeparableConv2D"],
    )
    param31 = RangeParameter(
        name="conv_2_layer_1_downsample",
        lower=0,
        upper=0.5,
        parameter_type=ParameterType.FLOAT,
    )
    param32 = RangeParameter(
        name="conv_2_layer_2_downsample",
        lower=0,
        upper=0.5,
        parameter_type=ParameterType.FLOAT,
    )
    param33 = RangeParameter(
        name="conv_2_layer_3_downsample",
        lower=0,
        upper=0.5,
        parameter_type=ParameterType.FLOAT,
    )
    # ##########################################################################
    param47 = RangeParameter(
        name="num_fc_layers", lower=0, upper=2, parameter_type=ParameterType.INT
    )
    param48 = RangeParameter(
        name="fc_weights_layer_1", lower=10, upper=200, parameter_type=ParameterType.INT
    )
    param49 = RangeParameter(
        name="fc_weights_layer_2", lower=10, upper=200, parameter_type=ParameterType.INT
    )
    ########################################################################
    param50 = RangeParameter(
        name="learning_rate",
        lower=0.0001,
        upper=0.01,
        parameter_type=ParameterType.FLOAT,
    )
    param51 = RangeParameter(
        name="learning_gamma", lower=0.9, upper=0.99, parameter_type=ParameterType.FLOAT
    )
    param52 = RangeParameter(
        name="learning_step", lower=1, upper=10000, parameter_type=ParameterType.INT
    )
    ########################################################################
    param53 = RangeParameter(
        name="drop_1", lower=0.1, upper=0.5, parameter_type=ParameterType.FLOAT
    )
    param54 = RangeParameter(
        name="drop_2", lower=0.1, upper=0.5, parameter_type=ParameterType.FLOAT
    )
    param56 = RangeParameter(
        name="drop_fc_1", lower=0.1, upper=0.5, parameter_type=ParameterType.FLOAT
    )
    param57 = RangeParameter(
        name="drop_fc_2", lower=0.1, upper=0.5, parameter_type=ParameterType.FLOAT
    )
    ########################################################################
    param58 = RangeParameter(
        name="prune_threshold",
        lower=0.05,
        upper=0.9,
        parameter_type=ParameterType.FLOAT,
    )
    param59 = RangeParameter(
        name="batch_size", lower=2, upper=8, parameter_type=ParameterType.INT
    )
    search_space = SearchSpace(
        parameters=[
            param0,
            param1,
            param2,
            param3,
            param7,
            param8,
            param9,
            param10,
            param11,
            param12,
            param13,
            param14,
            param15,
            param16,
            param17,
            param18,
            param19,
            param20,
            param21,
            param22,
            param23,
            param24,
            param25,
            param26,
            param27,
            param28,
            param29,
            param30,
            param31,
            param32,
            param33,
            param47,
            param48,
            param49,
            param50,
            param51,
            param52,
            param53,
            param54,
            param56,
            param57,
            param58,
            param59,
        ]
    )

    return search_space
