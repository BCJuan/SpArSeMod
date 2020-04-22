

## Usage

1. Create a Space Search function which returns a SearchSpace object, as in `cnn.py`
2. Create a network builder, as in `cnn.py`, which needs to accept the following parameters
    + Parametrization: that is the dictionary coming from the search space
    + The number of classes
    + A datset Object, as the object `TensorDataset`
3. Your data must be arranged in a list of `TensorDataset` objects as depicted for MNIST, CIFAR10, and CIFAR2 in `load_data.py`.
4. Depending on how your model behaves, that is, how it works with data (for example, sequences in RNNs), you can write a `collate_fn` and pass it to the search program which will use it with the dataloaders.

### Parameters

+ Quantization: 'post' or 'dynamic'
    + 'post': you should follow the procedure for static quantization in  the pytorch tutorial
    + 'dynamic': you should establish the modules that yolu want to quantize as in pytorch tutorials, or as established in main. If not specified, for example, when using `post`.
