# Using SpArSeMoD

To use SpArSeMoD you have to build several components:

+ Data Loading
+ Search Space
+ Network builder
+ Configuration file
+ Main call
+ Morphisms

We have seen some of them in the [Getting Started](getting_started.md) guide. In this section we are going to explain with a little bit of detail each of the documents.

Apart from the main optimization procedure to inspect the resulting data, some ax `Experiment` object inspection is needed, as well as some `.csv` wrangling.

## Main Call and Configuration file

SpArSeMoD internals are not directly accessible and it consists mainly in a simple call with some configurational parameters. The call to SpArSeMoD is made as:

```
sparse_instance = Sparse(...)
sparse_instance.run_sparse()
```

The `Sparse` object instantiation has a signature which corresponds to the different configuration parameters:

```
Sparse(
    r1=int(args["R1"]),
    r2=int(args["R2"]),
    r3=int(args["R3"]),
    epochs1=int(args["EPOCHS1"]),
    epochs2=int(args["EPOCHS2"]),
    epochs3=int(args["EPOCHS3"]),
    name=str(args["NAME"]),
    root=args["ROOT"],
    objectives=int(args["OBJECTIVES"]),
    batch_size=int(args["BATCH_SIZE"]),
    morphisms=bool_converter(args["MORPHISMS"]),
    pruning=bool_converter(args["PRUNING"]),
    datasets=datasets,
    classes=n_classes,
    debug=bool_converter(args["DEBUG"]),
    search_space=search_space,
    net=Net,
    flops=int(args["FLOPS"]),
    quant_scheme=str(args["QUANT_SCHEME"]),
    quant_params=quant_params,
    collate_fn=collate_fn,
    splitter=bool_converter(args["SPLITTER"]),
    morpher_ops=operations)
```

Most of those parameters are defined in the configuration file, which we will see next in this section. However, some of them are defined in the main call. From the example in ![cnn_cost_example](../examples/cnn_cost_example):

```
    datasets, n_classes = prepare_cost(folder="../data/data_cost/files", image=True)
    search_space = search_space()
    quant_params = None
    collate_fn = split_arrange_pad_n_pack
```

The first line is the dataset definition, defined in the further section [Data Loading](## Data Loading). It returns a list of the different sets (training, validation and test) and the number of classes. The second line returns an Ax `SearchSpace` object as defined in the section [Search Space](## Search Space). The parameter `quant_params` serves for defining the PyTorch `nn` modules that you want to quantize thorugh dynamic quantization (for example, `quant_params = [nn.Linear, nn.LSTM]`). We will review it in the configuration file description. And finally, the `collate_fn` is a variable for storing the collate fucntion, explained in [the collate section](### Collates).

Those four parameters are the ones defined in the call function and not throguh the configuration file (although you could define all parameters in the main call without using a configuration file).

### Configuration file

The configuration file serves as a parametrization of the SpArSe procedure (see ![theory](theory.md) for more information. Next a detail of the different parameters found in the configuration file is defined:

```
[DEFAULT]
train: BOOL 
    Defines whether we are in train mode or not. Normal usage is train = True. See Test section for a more detailed explanation.

[TRAIN] # train mode configuration
r1: INT
    Number of rounds in the first stage of optimization. With batch size of 1 this shouuld correspond to r1 random configurations
r2: INT
    Number of rounds in the second stage of optimization. Arc or Matern Kernel based Gaussian Process (GP) optimization. r2*batch configurations
r3: INT
    Number of rounds in the third stage. COnfigurations based in morphisms of the pareto frontier.
epochs1: INT
    Training epochs of each configuration at stage 1
epochs2: INT
    Training epochs of each configuration at stage 2
epochs3: INT
    Training epochs of each configuration at stage 3
name: STRING
    Name for the files where SpArSeMoD experiment and results are saved, both csv and json
root: STRING
    Folder where results are saved
objectives: INT
    Number of objectives included. It is incremental:
        1: accuracy
        2: accuracy, model size
        3: accuracy, model size, working memory
        4: accuracy, model size, working memory, latency
batch_size: INT
    Batch size for the Ax GP procedure. That means that if you define 3 as batch size, at each round  three different configurations are tested.
debug: BOOL
    Due to some networks created being too big for some images or feature maps, some times a `RuntimeError` occurs. For this reason, those errors are not directly raised. If you would like to inspect such errors produced, just activate this DEBUG param (will not raise them, but will print them)
flops: INT
    Frequency of your deployment platform. TO compute final latency in the platform.
quant_scheme: STRING
    PyTorch quantization scheme used. There are three possibilities activated trhough three keywords:
        post: post training static quantization
        dynamic: dynamic quantization
        both: applies to some parts post training static quantization and to some dynamic. Must be defined in the network builder.
    Check the PyTorch quantization procedure for more details
morphisms: BOOL
    Carry on with the third stage or not (equivalent to saying r3=0)
pruning: BOOL
    Use pruning during training or not.
splitter: BOOL
    Switch to activate `max_len` parameter if appropriate in the collate fn. Check the section ## COllates in the documentation
```

This is the minimum configuration for being able to use SpArSeMoD. The configuration file, `config.cfg` is simply called and read as:

```
from sparsemod.sparse.utils_data import configuration, bool_converter
args = configuration("TRAIN")
```

and then the parameters used as 

```
...
name=str(args["NAME"]),
root=args["ROOT"],
objectives=int(args["OBJECTIVES"]),
batch_size=int(args["BATCH_SIZE"]),
morphisms=bool_converter(args["MORPHISMS"]),
...
```

#### Test configuration

As you can see in the configuration file, there is a section for defining a "test" configuration. Why does this exist? The problem with neural network configuration and architecture testing is that you might want to do cross splits for testing or test the same configuration several times (not possible right now) and then test the chosen configuration in the test but at the end. 

The idea of this test is the following, you optimize with you validation set and then when you have your best working configuration you test it by doing cross splits of the whole dataset and obtain a final result.

Parameters are very similar to the training ones.
```
[TEST]
pruning: BOOL
    Use pruning during training or not.
splitter: BOOL
    Switch to activate `max_len` parameter if appropriate in the collate fn. Check the section ## COllates in the documentation
quant_scheme: STRING
    PyTorch quantization scheme used. There are three possibilities activated trhough three keywords:
        post: post training static quantization
        dynamic: dynamic quantization
        both: applies to some parts post training static quantization and to some dynamic. Must be defined in the network builder.
    Check the PyTorch quantization procedure for more details
objectives: INT
    Number of objectives included. It is incremental:
        1: accuracy
        2: accuracy, model size
        3: accuracy, model size, working memory
        4: accuracy, model size, working memory, latency
name: STRING
    Name for the files where SpArSeMoD experiment and results are saved, both csv and json
root: STRING
    Folder where results are saved
epochs: INT
    Number of epochs to train the model 
arm: STRING
    Arm number of the best configuration
```


## Data Loading

SpArSeMoD uses PyTorch  `TensorDataset` (although any object built upon `Dataset` should work) and the main loading function must return a list of the training, validation and test sets, altogether with the number of classes in the dataset. For example, let's take a look at the way we load the data from MNIST (the code can be found at `)

```
def prepare_mnist():
    """
    Reads mnist, samples from it a validation dataset in a stratified manner
    and returns train, val and test dataset

    Returns
    ------
    list of datasets
    number of classes
    """
    trainset = MNIST(
        root="./data/data_mnist", train=True, transform=transform_mnist(), download=True
    )
    val_set, tr_set = sample_from_class(trainset, 500)
    ts_set = MNIST(
        root="./data/data_mnist",
        train=False,
        transform=transform_mnist(),
        download=True,
    )
    return [tr_set, val_set, ts_set], len(MNIST.classes)
```

First, we are loading the training set with a simple transform

```
def transform_mnist():
    "Transforms for the mnist dataset"
    return Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
```

Then, with `val_set, tr_set = sample_from_class(trainset, 500)` we are splitting the training data set in a stratified manner into training and validation leaving 500 samples of each class. That is, we end up with a validation set of 500 examples of each class. The important point, however, is the type of object that it returns:

```
return (
        TensorDataset(train_data, train_label),
        TensorDataset(test_data, test_label),
    )

```

And that is what SpArSeMoD uses. Finally, after obtaining the test set, we retunr what is needed for SpArSeMoD:

```
    return [tr_set, val_set, ts_set], len(MNIST.classes)
```

which would be called like: `datasets, classes = prepare_mnist()`

### Collates 

There is the possibility of modifying the input data, for example, for standardizing it, through a `collate_fn`. The idea is that you define your collate before hand and pass it to SpArSeMoD which will use it in its internally defined dataloaders.

Let's take a look at the CoST collate in the example ![cnn_cost_example](../examples/cnn_cost_example):

```
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
```

All in all, works as a normal `collate_fn` since it returns a batch of inputs and labels. In the current version of SpArSeMoD, there is the possibility of adding an especial parameter to the collate: `max_len`. This is parameter is intended to be used when classifying sequences. The idea is that you define in your search space a `max_len` parameter as in the ![cnn_cost_example](../examples/cnn_cost_example) and then when the collate is used it will split the sentence. The usage of this parameter is entirely optional.

The collate function should be passed to SpArSeMoD as an uninstantiaed function, i.e.     `collate_fn = split_arrange_pad_n_pack`

and then in the `Sparse` instantiation

```
from cnn2d_cost import search_space, Net, operations, split_arrange_pad_n_pack
....

sparse_instance = Sparse(
    ... 
    collate_fn=collate_fn,
    ...)
```
 

### Conclusions

So, what does SpArSeMoD need for using data:

+ The data must be in encapsulated in a `Dataset` object (or child of it)
+ The loading function must return a list of the training, validation and test sets and the number of classes
`    return [tr_set, val_set, ts_set], n_classes`
+ The data is called as `datasets, classes = prepare_data()`



## Search Space

SpArSeMoD uses a search space definition from ![Ax](https://ax.dev/) in the ![developed mode](https://ax.dev/tutorials/gpei_hartmann_developer.html). Hence, all parameters use the same API.

To define a parmeter we just pick the type of parameter among different possibilities:

+ `RangeParameter`: parameters with upper and lower bounds and data type fixed.
+ `ChoiceParameter`: Different possibilities like in a set
+ `FixedParameter`

Let's see an example of `RangeParameter`:

```
RangeParameter(
    name="param_name",
    lower=2,
    upper=4,
    parameter_type=ParameterType.INT,
)
```

*An important point is the name that we give to the parameter: we will use it in the network builder*

One of the main problems of Ax and, in general, Gaussian Process based optimizers, is that they do not tolerate _ab initio_ conditional or hierarchical parameters. Hence we must link them ourselves. For example, we may use  parameter for the number of convolutional layers, another for the number of layers at the first of them and so on. 

This idea is used in examples such as ![Cost example](../examples/cnn_cost_example/cnn2d_cost.py). Let's see how in the case of convolutions:

```
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
```

As we can see we have a variable `max_number_of_blocks = 2`, which defines the first loop.  Then we have subvariables for next loops `max_number_of_layers_per_block = 3`. We use a first loop to define variables at this hierarchical level, such as the number of layers in that convolutional block

```
for i in range(max_number_of_blocks):
...
    params.append(
        RangeParameter(
            name="conv_" + str(i + 1) + "_num_layers",
            lower=1,
            upper=max_number_of_layers_per_block,
            parameter_type=ParameterType.INT,
        )
    )
...
```

and then, inside this loop, we define other sub loops such as the number of filters for each of the layers at each of the convolution bolocks.

```
for i in range(max_number_of_blocks):
...
    for j in range(max_number_of_layers_per_block):
    ...
        params.append(
        RangeParameter(
            name="conv_" + str(i + 1) + "_layer_" + str(j + 1) + "_filters",
            lower=1,
            upper=100,
            parameter_type=ParameterType.INT,
            )
        )
...
```

Finally we transform the list of parameters into a `SearchSpace` object: `search_space = SearchSpace(parameters=params)` which will be used by SpArSeMoD.

Important to note is that we can also include constraints between parameters, such as a sum constraint or magnitude. See `ParameterConstraint` at ![Ax](https://ax.dev/api/core.html?highlight=parameter#module-ax.core.parameter).

This is a proposal for defining the search space with Ax variables. However, we can use whatever way we desire iff we use an Ax search space type parameters and object.

To use it, import it in the Sparse main call and add it to `Sparse`:

```
from cnn import search_space, Net, operations
....
search_:space = search_space()

sparse_instance = Sparse(
    ... 
    search_space=search_space,
    ...)
```
 

### Conclusion

We need to:

+ Create a `SearchSpace` object with Ax parameters
+ The names used will be used by the network builder


## Network Builder

The next element that has to be built is the network builder. It will use the parameters defined in the search space. Generally, it is like a PyTorch Module but being able to accept a configuration dictionary to build the specific network.

Let's see the class signature:

```
class Net(nn.Module):
    def __init__(self, parametrization, classes=10, input_shape=None):
    ...
````

as see it needs to have three parameters: the dictionary containing the specific configuration, the number of classes, and the input shape.

More requirements are the following. It obviously needs to have a `forward` method, and if quantization is in use, you need to have a `fuse modules` method.

Regarding how it workd, it basically is a generalized network builder. For example to build the convolutional part of a network it first deines a general loop for each block:

```
...
conv_blocks = []
for j in range(1, parametrization.get("num_conv_blocks", 1) + 1):
    conv_blocks.append(self.create_conv_block(j, channels))
self.conv_blocks = nn.Sequential(*conv_blocks)
...
```
where `parametrization` contains the dictionary of the configuration, and `self.create_conv_block` is a functon for building each block. In it we can see a generalized convolution builder:

```
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

            if conv_type == search_:space = search_space()hannels,
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
        conv.append(nn.Droposearch_:space = search_space()

In the `forward` method, if quantization is used, you should include the `QuantStub` and `DeQuantStub` operators as specified in PyTorch quantization guidelines. For example, for only post training static quantization, we could do

```

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
```
In general, regarding quantization, you can follow the pytorch quantization guidelines since the quantization procedures have been made following those procedures.

Once you have defined your network builder, you have to import it in your `main` function and pass it to `Sparse`.

```
from cnn import search_space, Net, operations
....

sparse_instance = Sparse(
    ... 
    net=Net,
    ...)
```
 
## Morphisms

Morphisms correspond to the third stage of the procedure. Morphisms basically consist in hard coded functions that modify the configurtion specifically. As those operations are architecture dependent they have to be defined by the user. They are linked to the parameters defined in the Search Space and then they should use the same names.

 Let's see an example:

```
def kernel_size(config):
    """
    Changes the kernel in a randomly chosen convolution
    """
    block = choice(range(1, config["num_conv_blocks"] + 1))
    layer = choice(range(1, config["conv_" + str(block) + "_num_layers"] + 1))
    new_kernel_size = randint(2, 5)
    config["conv_" + str(block) + "_layer_" + str(layer) + "_kernel"] = new_kernel_size
    return config
```

Here we see the only two things that are mandatory for a morphism in SpArSeMoD, the argument must only be a dictionary containng the configurations and the retunrn must be that modified dictionary. In this latter case, we can see that the morphism detailed changes the kernel size of a random convolution layer of the network.

Once you have defined your morphims, grouped them in a dictionary, as for example:

```
operations = {
    "num_fc_layers": num_fc_layers,
    "num_conv_blocks": num_conv_blocks,
    "layer_type": layer_type,
    "num_conv_filters": num_conv_filters,
    "kernel_size": kernel_size,
    "downsampling_rate": downsampling_rate,
    "num_fc_weights": num_fc_weights,
    "num_conv_layers": num_conv_layers,
}
```
Then import them in the main function and add them to Sparse


```
from cnn import search_space, Net, operations
....

sparse_instance = Sparse(
    ... 
    morpher_ops=operations,
    ...)
```

## Inspecting results

Sparse produces two files for the results, one `.csv` where the performance results of each cofigurationn are save along the arm name, and a `.json` where the experiment is saved.

We will use the `.csv` to recover the values of the performance and the `.json` to recover the parameter of the best configuration.

### .csv results

If you take a look at a `.csv`, you will see that the results do not resemble real values corresponding to model size, ram or ltency. That is because results are standarized internally by SpArSeMoD to be able to put all objectives in more or less the same range ([0,1]). The only performance value that can be read directly is accuracy. The others must be transformed.

An example of all the procedure can be found at ![cnn_cost_example](../examples/cnn_cost_example/visualizing_results.ipynb) in the form of a jupyter notebook tutorial.


### Best configuration details

To be able to see a configuration specific parameters the following procedure is required. 

1. Reload the experiment as if you were to run SpArSe again

```
    sparse_exp = SparseExperiment(
        name=str(args["NAME"]),
        root=args["ROOT"],
        objectives=int(args["OBJECTIVES"]),
        pruning=bool_converter(args["PRUNING"]),
        epochs=args["epochs1"],
        datasets=datasets,
        classes=n_classes,
        search_space=sspace,
        net=Net,
        flops=int(args["FLOPS"]),
        quant_scheme=str(args["QUANT_SCHEME"]),
        quant_params=quant_params,
        collate_fn=collate_fn,
        splitter=bool_converter(args["SPLITTER"]),
        models_path =path.join(args["ROOT"], "models")
    )
```

2. Obtain the experiment and the data by calling the function `load_experiment`

```
    exp, data = sparse_exp.create_load_experiment()
```

3. Select the arm tht you want to inspect

```
exp.arms_by_name['779_0']
```