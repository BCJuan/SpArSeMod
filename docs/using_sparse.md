# Using SpArSeMoD

To use SpArSeMoD you have to build several components:

+ Data Loading
+ Search Space
+ Network builder
+ Configuration file
+ Main call
+ Morphisms

We have seen some of them in the [Getting Started](getting_started.md) guide. In this section we are going to explain with a little bit of detail each of the documents.

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

### Conlcusions

So, what does SpArSeMoD need for using data:

+ The data must be in encapsulated in a `Dataset` object (or child of it)
+ The loading function must return a list of the training, validation and test sets and the number of classes
`    return [tr_set, val_set, ts_set], n_classes`
+ The data is called as `datasets, classes = prepare_data()`

******************DATA loaders********************

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

### Conclusion

We need to:

+ Create a `SearchSpace` object with Ax parameters
+ The names used will be used by the network builder


## Network Builder