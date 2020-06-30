"""
Classes and functions to be able to load all the datasets used:

COST, CIFAR10, CIFAR2 and MNIST

"""

from os import listdir, path, mkdir, rmdir
from itertools import product
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import TensorDataset, Dataset
from torch import tensor, int64, stack, Tensor, FloatTensor
from numpy import (
    asarray,
    loadtxt,
    zeros_like,
    where,
    vstack,
    unique,
    save,
    load,
    arange,
    random,
    mean as npmean,
    std as npstd,
)

from pandas import read_csv
from sklearn.preprocessing import StandardScaler

def read_cifar2(folder="./data/data_cifar2", width=20, height=20):
    """
    reads, loads and mounts cifar2 dataset

    Args
    ----
    folder: where the data for cifar 2 is placed
    width: how to arrange sequences from cifar 2
    height: ""

    Returns
    ----
    train_dataset: to be further split into val and train
    test_dataset

    """
    for fil in listdir(folder):
        data = loadtxt(path.join(folder, fil), skiprows=1)
        if fil.split(".")[1] == "train":
            train_label = data[:, 0]
            zeros_label = zeros_like(train_label)
            train_label = where(train_label > 0, train_label, zeros_label)
            train_data = data[:, 1:].reshape(-1, width, height)
        else:
            test_label = data[:, 0]
            zeros_label = zeros_like(test_label)
            test_label = where(test_label > 0, test_label, zeros_label)
            test_data = data[:, 1:].reshape(-1, width, height)
    return (
        TensorDataset(FloatTensor(train_data).unsqueeze(1), tensor(train_label)),
        TensorDataset(FloatTensor(test_data).unsqueeze(1), tensor(test_label)),
    )


# https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
def sample_from_class(data_set, k):
    """
    function to sample data and their labels from a dataset in pytorch in
    a stratified manner

    Args
    ----
    data_set
    k: the number of samples that will be accuimulated in the new slit

    Returns
    -----
    train_dataset
    val_dataset

    """
    class_counts = {}
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for data, label in data_set:
        class_i = label.item() if isinstance(label, Tensor) else label
        class_counts[class_i] = class_counts.get(class_i, 0) + 1
        if class_counts[class_i] <= k:
            train_data.append(data)
            train_label.append(label)
        else:
            test_data.append(data)
            test_label.append(label)

    train_data = stack(train_data)
    train_label = tensor(train_label, dtype=int64)
    test_data = stack(test_data)
    test_label = tensor(test_label, dtype=int64)

    return (
        TensorDataset(train_data, train_label),
        TensorDataset(test_data, test_label),
    )


def transform_mnist():
    "Transforms for the mnist dataset"
    return Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


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


def transform_cifar10():
    "transforms for the cifar 10"
    return Compose(
        [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )


def prepare_cifar10():
    """
    Reads cifar10, samples from it a validation dataset in a stratified manner
    and returns train, val and test dataset

    Returns
    ------
    list of datasets
    number of classes
    """
    trainset = CIFAR10(
        root="./data/data_cifar10",
        train=True,
        transform=transform_cifar10(),
        download=True,
    )
    val_set, tr_set = sample_from_class(trainset, 500)
    ts_set = CIFAR10(
        root="./data/data_cifar10",
        train=False,
        transform=transform_cifar10(),
        download=True,
    )
    return [tr_set, val_set, ts_set], 10


def prepare_cifar2(folder="./data/data_cifar2", n_val=2500):
    """
    Reads cifar2, samples from it a validation dataset in a stratified manner
    and returns train, val and test dataset

    Returns
    ------
    list of datasets
    number of classes
    """
    tr_dt, ts_dt = read_cifar2(folder=folder)
    val_set, tr_set = sample_from_class(tr_dt, n_val)
    return [tr_set, val_set, ts_dt], 2
