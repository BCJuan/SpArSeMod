<p align="center">
    <br>
    <img src="https://github.com/BCJuan/SpArSeMod/blob/reorganize/images/logo.png" width="750"/>
    <br>
<p>

<p align="center">
    <a href="https://github.com/BCJuan/SpArSeMod/blob/reorganize/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/badge/license-GNU%20GPL%203.0-blue">
    </a>
        <a href="https://github.com/BCJuan/SpArSeMod/releases/tag/">
        <img alt="GitHub release" src="https://img.shields.io/badge/release-v0.1.2-blue">
    </a>
</p>
<h3 align="center">
<p>Neural Network Development for Microcontrollers
</h3>

# Description

Adaptation and extension of ![SpArSe](https://www.cs.princeton.edu/~rpa/pubs/fedorov2019sparse.pdf). The present code delivers neural networks optimized according to performance error, working memory (~RAM), model size (~Flash) and latency.

The configuration space (or search space) of the network, that is, the components that build it up are elligible by the user: you can build up networks with CNNs, RNNs or whatever structure you like. However, you will have to build the network builder: the function that will specify the concrete network from a search space point. 

Build with [PyTorch](https://pytorch.org/), [GPyTorch](https://gpytorch.ai/)  and [Ax](https://ax.dev/).

The result obtained after running a SpArSeMoD process is the group of best networks (Pareto frontier networks):

<p align="center">
    <br>
    <img src="https://github.com/BCJuan/SpArSeMod/blob/reorganize/images/result_sample.png" width="500"/>
    <br>
<p>


## Table of contents
* [Description](#description)
* [Installation](#installation)
* [Usage](#sample-use-and-examples)
* [Improvements](#improvements)
* [Support](#support)
* [License](#license)

## Installation

1. Clone this repository
2. Install conda environment: `conda create env -f sparse.yml`
    1. Alternatively, install packages directly from pip
3. run `pip install .` 

## Sample use and examples

The folder `examples/` contains several separate functioning examples. Each folder is a different example and contains all the examples to run a sessions of SparseMod.

To adapt to your specific problem, try to imitate one of those examples. The components needed are:
1. Create a Space Search function which returns a `SearchSpace` object, as in `/examples/cnn_cifar_example/cnn.py`.
2. Create a network builder, as in `examples/cnn_cifar_example/cnn.py`, which needs to accept the following parameters: `parametrization, classes=10, input_shape=None`
3. Build you dataset and data pipeline as in `examples/cnn_cifar_example/load_data.py`.
4. Build the main call function as in `examples/cnn_cifar_example/main.py`.
4. Complete `config.cfg`, with all the necessary parameters .
5. Run `python main.py`
6. Wait until the process completes. You will have to check the results to see which has been the best network.

The process described is for a CNN based network and the [CIFAR 10 Binary](http://manikvarma.org/code/LDKL/download.html).


### Results

1. The models will be placed in the folder specified in the configuration as `root`, and there, a `models` folder will be created, where all the models will be stored. Only pareto models are saved.
2. Results for the evaluations will be placed in the specified folder in the configuration. It consists of a `csv` where results are saved as dataframe, a `json` for the experiment and a txt for the time taken for the whole experiment.
3. In the file `examples/result_inspection.ipynb`, an example of analysing the results is exemplified.
4. In case yo want to recover the exact structure of your chosen netwrok, you will ahve to rerun SpArse to reload the experiment and inspect the network. An example of such process is found in `examples/cnn_cost_example/read_experiment.py`

## Improvements

The next points represent current WIP points considered as weak or defective points of the framework. 

1. Check batch size for GP works
2. Add notebook for selecting best network and inspecting results
3. When loading the data it should be loaded only up to an index, since we dont want to reload morphed solutions results


Necessary improvements for the framework:


1. Add raytune distribution
2. Add tolerance checker as stopper
3. Solving the serialization problem when saving the experiment. The JSON files weighs so much because everything inside the metrics classes is saved inside

## Support

If you are having issues, please let us know.
We have a mailing list located at: juan.borrego@uab.cat


## License

The project is licensed under the GNU GLP 3.0 License

## Acknowledgement

This code has been developed by a joint collaboration of 
 <div class="row">
  <div class="column">
    <img alt="GitHub" src="https://github.com/BCJuan/SpArSeMod/blob/reorganize/images/uab.jpeg"  style="width:100%" href="https://www.uab.cat/web/directory/search/entities-1345675609174.html?param1=1345674960027">

  </div>
  <div class="column">
    <a href="https://github.com/BCJuan/SpArSeMod/releases/tag/">
        <img alt="GitHub release" src="https://github.com/BCJuan/SpArSeMod/blob/reorganize/images/kostal.png" style="width:100%">
    </a>    
  </div>
</div> 
