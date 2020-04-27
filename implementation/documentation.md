

## Usage

1. Create a Space Search function which returns a SearchSpace object, as in `cnn.py`.
    The following elements should always appear on the search space
    + Learning rate
    + Learning step (adjust for your dataset)
    + Learning gamma 
    + Prune parameter 
    + Batch size

2. Create a network builder, as in `cnn.py`, which needs to accept the following parameters
    + Parametrization: that is the dictionary coming from the search space
    + The number of classes
    + Input shape

3. Your data must be arranged in a list of `TensorDataset` or `Dataset` objects as depicted for MNIST, CIFAR10, and CIFAR2 in `load_data.py`.
4. Depending on how your model behaves, that is, how it works with data (for example, sequences in RNNs), you can write a `collate_fn` and pass it to the search program which will use it with the dataloaders. In the case you write a collate for rnns it should end with a packed sequence object.
5. Beware that the data shape passed to the net comes from the dataloader but without batch dimension. In the case

6. IF using sequences and making explicit a chop, the chop variable should be called `max_len` as in rnn. If not chopping, make a fixed parameter in search space
7. You have to define the operations in the morpher as in `cnn.py` or `rnn.py`


### Parameters

+ Quantization: 'post' or 'dynamic'
    + 'post': you should follow the procedure for static quantization in  the pytorch tutorial
    + 'dynamic': you should establish the modules that yolu want to quantize as in pytorch tutorials, or as established in main. If not specified, for example, when using `post`.
+ Splitter: use it for telling sparse that the collate needs a splitter parameters, such as max sequence length for the inputs, and before dataloading it should make
a partial.
