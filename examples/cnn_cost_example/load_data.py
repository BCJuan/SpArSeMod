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


def read_n_save_cost(folder="./data/data_cost", subfolder="files"):
    """
    Function used to split the original cost dataset in different sequences
    according to subject gest and variant.
    the files are finally saved as npy

    Args
    ----
    folder: where the cost file is located
    subfolder: where to save the files
    """
    name_subfolder = path.join(folder, subfolder)

    if not path.exists(name_subfolder):
        mkdir(name_subfolder)
    else:
        rmdir(name_subfolder)
        mkdir(name_subfolder)

    data_frame = read_csv(path.join(folder, "CoST.csv"))
    value_cols = [column for column in data_frame.columns if "ch" in column]

    total_product = product(
        unique(data_frame[" gesture"]),
        unique(data_frame["subject"]),
        unique(data_frame[" variant"]),
    )
    for gest, subj, vals in total_product:
        simple_df = data_frame[
            (data_frame[" gesture"] == gest)
            & (data_frame["subject"] == subj)
            & (data_frame[" variant"] == vals)
        ].reset_index()
        if not simple_df.empty:
            new_df = []
            counter = 0
            for i, (_, row) in enumerate(simple_df.iterrows()):
                new_df.append(row[value_cols].values)
                if (i == len(simple_df) - 1) or (simple_df.loc[i + 1, " frame"] == 1):
                    name = (
                        str(gest)
                        + "_"
                        + str(subj)
                        + "_"
                        + str(vals)
                        + "_"
                        + str(counter)
                    )
                    save(path.join(name_subfolder, name), vstack([new_df]))
                    new_df = []
                    counter += 1


def build_cost(folder="./data/data_cost/files/"):
    """
    Reads all the npy files produced by read_n_save_cost
    and returns X sequences and y label per sequence
    altogether with metadata corresponding to subject variant
    and so

    Args
    ----
    folder: where the files are located

    Returns
    -------
    X: sequences
    y: label per sequence
    extra: data regarding subject, variant, and so (check cost)

    """
    X = []
    y = []
    extra = {}
    for i, arxiv in enumerate(listdir(folder)):
        X.append(load(path.join(folder, arxiv)))
        meta = arxiv.split("_")
        y.append(int(meta[0]) - 1)
        extra[str(i)] = dict({"subject": int(meta[1]), "variant": int(meta[2])})
    return X, y, extra


def divide_cost(x_data, y_data, extra, test_subjects=None):
    """
    Divides cost in training, validation and test
    randomly divides into 20 subjects training, 5 validation
    and 6 test
    unless test_subjects is specified. Then only this subject is
    passed as test, there is no validation, and all the other go to
    training

    Args
    ----
    x_data: sequences
    y_data: label per sequence
    extra: metadata corresponding to subject and so
    test_subjects: (OPtional) the idx of test subjects

    Returns
    -------
    x_train,
    y_train,
    extra_train,
    x_val,
    y_val,
    extra_val,
    x_test,
    y_test,
    extra_test,
    """
    x_train, x_val, x_test = [], [], []
    y_train, y_val, y_test = [], [], []
    extra_train, extra_val, extra_test = {}, {}, {}
    subjects = arange(1, 32)
    if test_subjects:
        train_subjects = [i for i in subjects if i not in test_subjects]
        val_subjects = test_subjects
    else:
        train_subjects = random.choice(subjects, size=25, replace=False)
        val_subjects = random.choice(train_subjects, size=5, replace=False)
        test_subjects = [i for i in range(1, 32) if i not in train_subjects]
        train_subjects = [i for i in train_subjects if i not in val_subjects]
    for i, (seq, label, meta) in enumerate(zip(x_data, y_data, extra)):
        if extra[meta]["subject"] in train_subjects:
            x_train.append(seq)
            y_train.append(label)
            extra_train[len(x_train)] = extra[meta]
        if extra[meta]["subject"] in val_subjects:
            x_val.append(seq)
            y_val.append(label)
            extra_val[len(x_train)] = extra[meta]
        if extra[meta]["subject"] in test_subjects:
            x_test.append(seq)
            y_test.append(label)
            extra_test[len(x_train)] = extra[meta]
    return (
        x_train,
        y_train,
        extra_train,
        x_val,
        y_val,
        extra_val,
        x_test,
        y_test,
        extra_test,
    )


def scale_X(x_train, x_test, image=False):
    """
    Function to scale the sequences from cost
    Depending if the image arg is activated they are scaled
    as sequences or images

    Args
    ----
    x_train:
    x_test
    image: boolean, to standardize as an image or as a sequence

    Returns
    -----
    x_train_esc
    x_test_esc
    scaler: can be a Scaler object from sklearn or a list of [mean, std]
            depending on the arg image
    """
    if image:
        mean = npmean(vstack(x_train).reshape(-1, 8, 8), axis=0)
        std = npstd(vstack(x_train).reshape(-1, 8, 8), axis=0)
        x_train_esc = [(asarray(i).reshape(-1, 8, 8) - mean) / std for i in x_train]
        x_test_esc = [(asarray(i).reshape(-1, 8, 8) - mean) / std for i in x_test]
        scaler = [mean, std]
    else:
        scaler = StandardScaler()
        scaler.fit(vstack(x_train))
        x_train_esc = [scaler.transform(i).tolist() for i in x_train]
        x_test_esc = [scaler.transform(i).tolist() for i in x_test]
    return x_train_esc, x_test_esc, scaler


class CostDataset(Dataset):
    """
    Class to build the cost dataset
    """

    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        sample = {"signal": self.x_data[idx], "label": self.y_data[idx]}
        if self.transform:
            sample = self.transform(sample)

        return sample


def prepare_cost(folder="./data/data_cost/files/", test_subjects=None, image=False):
    """
    Makes the whole process of loading the cost dataset, dividing, standardizing it
    and build it as  as separate datasets

    Args
    ----
    folder: where the files from read_n_save_cost are placed to be read
    test_subjects: (optional) if there are test subjects for the dataset to be slit by
    image: wheter we arrange data as an image or a sequence

    Returns
    -------
    t_set
    v_set
    ts_set
    number of classes
    """

    # turns y labels into numbers; # generate dataset according to y labels stratification
    # scaling, o ne hot encoding
    x_data, y_data, extra = build_cost(folder=folder)
    x_train, y_train, _, x_val, y_val, _, x_test, y_test, _ = divide_cost(
        x_data, y_data, extra, test_subjects=test_subjects
    )
    x_train, x_test, scaler = scale_X(x_train, x_test, image=image)
    if image:
        x_val = [
            (asarray(seq).reshape(-1, 8, 8) - scaler[0]) / scaler[1] for seq in x_val
        ]
    else:
        x_val = [scaler.transform(seq) for seq in x_val]
    t_set = CostDataset(x_train, y_train)
    v_set = CostDataset(x_val, y_val)
    ts_set = CostDataset(x_test, y_test)
    return [t_set, v_set, ts_set], 14





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
