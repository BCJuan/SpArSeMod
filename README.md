# EXPERIMENTAL/RESEARCH CODE (WIP)

# Description

Adaptation and extension of ![SpArSe](https://www.cs.princeton.edu/~rpa/pubs/fedorov2019sparse.pdf). The present code delivers, given a search space, a network builder and morphims operators, network optimized according to performance error, working memory, model size and latency.

That is, this software delivers well performing networks suitable for resource constrained devices in terms of memory size and latency.

Build with Pytorch and Ax.


# Usage

The folder `sparsemod/examples/` contains several separate functioning examples. Each folder is a different example and contains all the examples to run a sessions of SparseMod.

To adapt to your specific problem, try to imitate one of those examples. The components needed are:
1. Create a Space Search function which returns a SearchSpace object, as in `sparsemod/examples/cnn_cifar_example/cnn.py`.
2. Create a network builder, as in `sparsemod/examples/cnn_cifar_example/cnn.py`, which needs to accept the following parameters
3. Your data must be arranged in a list of `TensorDataset` or `Dataset` objects as depicted for MNIST, CIFAR10, and CIFAR2 in `sparsemod/examples/load_data.py`.
4. Complete `config.cfg`, with all the parameters (check configuration section for this)
5. Build a file (example is in current `main.py`) where you call the function `Sparse` from `sparse.sparse` and pass all the config parameters.
6. Wait until the process completes. You will have to check the results to see which has been the best network.

### Optional Usage

1. Depending on how your model behaves, that is, how it works with data (for example, sequences in RNNs), you can write a `collate_fn` and pass it to the search program which will use it with the dataloaders. As detailed, for example, in 
2. IF using sequences and making explicit a chop, the chop variable should be called `max_len` as in rnn. If not chopping, make a fixed parameter in search space. Check functions in `sparsemod/examples/cnn2d_cost.py` -> `split_arrange_pad_n_pack`


## Configuration

The parameters in the `config.ini`file obey the following:
```
[DEFAULT]
train: Wheter to train or to test

[TRAIN]
r1:
    (INT) rounds in sobol regime
r2:
    (INT) rounds in GP based optimization
r3: 
    (INT) rounds in morphism with point choice through GP based evaluation
epochs1:
    (INT) Number of epochs in round 1
epochs2:
    (INT) Number of epochs in round 2
epochs3:
    (INT) Number of epochs in round 3
name:
    (STRING) name for the files where results are saved
root:
    (STRING) folder where to save the results
objectives:
    (INT) number of objectives to optimize. It is sequential and accumulative:
    + 1: Accuracy
    + 2: Accuacy and Model Size
    + 3: Accuracy, MOdel Size and Working Memory
    + 4: Accuracy, MOdel Size, Working Memory and Latency
batch_size:
    (INT) Batch size for the GP. 
debug:
    (BOOL) Whether to print error messages or not. `RuntimeError` and `IndexError` are not raised but jumped over as exceptions. Check code explanation section.
flops:
    (INT) Number of operations per second that your device is able to perform. Frequency could be used as approximation.
quant_scheme:
    (STRING) the procedure according to ![pytorch quantization](https://pytorch.org/docs/stable/quantization.html) by which the nets are quantized
    Options:
    + 'post': quantizes following static post training 
    + 'dynamic': only activations, for RNNs
    +' both': when applying post to some parts and dynamic to others
morphisms:
    (BOOL) Whether to use morphisms or not
pruning:
    (BOOL) Whether to prune the networks or not
splitter:
    (BOOL) If you are using a splitter of sequences then you should activate this function


[TEST] -> configuration for testing the network if the procedure has been carried only with training and validation
pruning: 
    same as in training
splitter:
    same as in training
quant_scheme:
    same as in training
objectives:
    same as in training
name:
    same as in training
root:
    (STRING) path where the model to load is found
epochs:
    (INT) Total epochs to train the model before testing
arm:
    (STRING) the name of the configuration to use for testing
```

## Results

1. The models will be placed in the folder specified in the configuration as `root`, and there, a models folder will be created, where all the models will be stored. Only pareto models are saved.
2. Results for the evaluations will be placed in the specified folder in the configuration. It consists of a `csv` where results are saved as dataframe, a `json` for the experiment and a txt for the time taken for the whole experiment.


# Installation

1. Clone this repository
2. Install conda environment: `conda create env -f sparse.yml`
    1. Alternatively, install packages directly from pip
3. run `pip install .` 
4. In example you can find a full working example to begin with.

# Points to solve/improve

The next points represent current WIP points considered as weak or defective points of the framework. 

1. Check batch size for GP works
2. Add notebook for selecting best network and inspecting results
3. When loading the data it should be loaded only up to an index, since we dont want to reload morphed solutions results


Necessary improvements for the framework:


1. Add raytune distribution
2. Add tolerance checker as stopper
3. Solving the serialization problem when saving the experiment. The JSON files weighs so much because everything inside the metrics classes is saved inside

# Code Explanation

1. Why `RuntimeError` or `IndexError`are not raised in net evaluation.
    - The main reason is that with tiny images such as CIFAR2 or MNIST, sometimes the netowrk is to deep and is then not functional (cannot operate on the image). However, upsizing the image is not an option since this would increment the size of the images used and hence memory requirements. Hence, when such a network is created the errors are produced and jumped over.