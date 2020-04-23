from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import TensorDataset, DataLoader
from torch import unsqueeze, cat, tensor, int64, stack, Tensor, unsqueeze, FloatTensor
from numpy import asarray, loadtxt, zeros_like, where, vstack, unique, save, load, arange, random
from os import listdir, path, mkdir, rmdir
from pandas import read_csv
from itertools import product
from collections import Counter
from tqdm import tqdm

def get_input_shape(datasets):
    shape = datasets[0][0][0].shape
    if len(shape) > 2:
        return shape
    else:
        return (1, *shape)

def read_n_save_cost(folder="data_cost", subfolder="files"):
    name_subfolder = path.join(folder, subfolder)

    if not path.exists(name_subfolder):
        mkdir(name_subfolder)
    else:
        rmdir(name_subfolder)
        mkdir(name_subfolder)
    
    df = read_csv(path.join(folder, "CoST.csv"))
    value_cols = [column for column in df.columns if 'ch' in column]
    gestures = unique(df[' gesture'])
    subjects = unique(df['subject'])
    variants = unique(df[' variant'])
    data = {}
    total_product = product(gestures, subjects, variants)
    for g, s, v in total_product:
        print(g, s, v)
        simple_df = df[(df[' gesture'] == g) & (df['subject'] == s) & (df[' variant'] == v)].reset_index()
        if not simple_df.empty:
            new_df = []
            counter = 0
            for i, (index, row) in enumerate(simple_df.iterrows()):
                new_df.append(row[value_cols].values)
                if (i == len(simple_df) - 1) or (simple_df.loc[i + 1, ' frame'] == 1) :
                    name = str(g) + "_" + str(s) + "_" + str(v) + "_" + str(counter)
                    save(path.join(name_subfolder, name), vstack([new_df]))
                    new_df = []
                    counter +=1

def build_cost(folder="./data_cost/files/"):
    X = []
    y = []
    extra = {}
    for i, arxiv in enumerate(listdir(folder)):
        X.append(load(path.join(folder, arxiv)))
        meta = arxiv.split("_")
        y.append(int(meta[0]) - 1)
        extra[str(i)] = dict({'subject': int(meta[1]), 'variant': int(meta[2])})
    return X, y, extra


def divide_cost(X, y, extra):
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []
    extra_train, extra_val, extra_test = {}, {}, {}
    subjects = arange(1, 32)
    train_subjects = random.choice(subjects, size=25, replace=False)
    val_subjects = random.choice(train_subjects, size=5, replace=False)
    train_subjects = [i for i in train_subjects if i not in val_subjects]
    for i, (seq, label, meta) in enumerate(zip(X, y, extra)):
        if extra[meta]['subject'] in train_subjects:
            X_train.append(seq)
            y_train.append(label)
            extra_train[len(X_train)] = extra[meta]
        elif extra[meta]['subject'] in val_subjects:
            X_val.append(seq)
            y_val.append(label)
            extra_val[len(X_train)] = extra[meta]
        else:
            X_test.append(seq)
            y_test.append(label)
            extra_test[len(X_train)] = extra[meta]
    return X_train, y_train, extra_train, X_val, y_val, extra_val, X_test, y_test, extra_test


def scale_X(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(vstack(X_train))
    X_train_esc = [scaler.transform(i).tolist() for i in X_train]
    X_test_esc = [scaler.transform(i).tolist() for i in X_test]
    return X_train_esc, X_test_esc, scaler


def create_cost_dataset(folder="./data_cost/files/"):
    # turns y labels into numbers; # generate dataset according to y labels stratification
    # scaling, o ne hot encoding
    X, y, extra = build_cost()
    X_t, y_t, e_t, X_v, y_v, e_v, X_ts, y_ts, e_ts = divide_cost(X, y, extra)
    X_t, X_ts, scaler = scale_X(X_t, X_ts)
    X_val = [scaler.transform(seq) for seq in X_val]
    return X_t, y_t, X_v, y_v, X_ts, y_ts

def read_cifar2(folder="data_cifar2", width=20, height=20):
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
def sampleFromClass(ds, k):
    class_counts = {}
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for data, label in ds:
        c = label.item() if type(label) == Tensor else label
        class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts[c] <= k:
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
    return Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


def prepare_mnist():
    trainset = MNIST(
        root="./data_mnist", train=True, transform=transform_mnist(), download=True
    )
    val_set, tr_set = sampleFromClass(trainset, 500)
    ts_set = MNIST(
        root="./data_mnist", train=False, transform=transform_mnist(), download=True
    )
    return [tr_set, val_set, ts_set], len(MNIST.classes)


def transform_cifar10():
    return Compose(
        [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )


def prepare_cifar10():
    trainset = CIFAR10(
        root="./data_cifar10", train=True, transform=transform_cifar10(), download=True
    )
    val_set, tr_set = sampleFromClass(trainset, 500)
    ts_set = CIFAR10(
        root="./data_cifar10", train=False, transform=transform_cifar10(), download=True
    )
    return [tr_set, val_set, ts_set], 10


def prepare_cifar2():
    tr_dt, ts_dt = read_cifar2()
    val_set, tr_set = sampleFromClass(tr_dt, 2500)
    return [tr_set, val_set, ts_dt], 2
