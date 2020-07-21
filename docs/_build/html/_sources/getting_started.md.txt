# Getting Started

In this section, the example `cnn_cost_example` from [Examples](https://github.com/BCJuan/SpArSeMod/tree/reorganize/examples/cnn_cost_example) is explained
to see how SpArSeMod works. 

This is just a tutorial showing how to use SpArSeMod, for the specifics, background theory and related,
reading the paper [SpArSe: Sparse Architecture Search for CNNs on Resource-Constrained Microcontrollers](https://arxiv.org/abs/1905.12107) is
highly recommended.

If you want to skip the paper and read a summary of most important facts, you can go to [./]

## Following cnn_cost_example

In every run we are going to need 4 new components written: search space, network builder, main call and configuration. The use of morphisms is 
optional but recommended. 

Clone the repository and follow installation instructions. You will need [anaconda](https://docs.conda.io/en/latest/miniconda.html) or install yourself
the dependencies.

### Getting the data

The tutorial is based on a modified version of CIFAR10, the [CIFAR10 Binary](http://manikvarma.org/code/LDKL/download.html), where instead of 2 classes we have 2. First run the file
`examples/download_data.py` it will download CoST (Corpus of Social Touch) and create the necessary directories. Then, download the CIFAR Binary from this [link](https://docs.google.com/uc?export=download&id=0B5E8qFcWFPQOYXg3X29uMDdINU0) and extract the files `cifar10binary.test`and `cifar10binary.train` into the `examples/data/data_cifar2` folder.

