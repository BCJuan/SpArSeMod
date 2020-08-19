from os import mkdir, path
from time import time
from warnings import filterwarnings

import numpy as np
from torch import manual_seed

from cnn import Net, operations, search_space
from load_data import prepare_cifar2
from sparsemod.utils_data import bool_converter, configuration
from sparsemod.utils_experiment import SparseExperiment


def load_experiment(data_folder="../data/data_cost/files"):

    manual_seed(42)
    np.random.seed(42)

    filterwarnings(action="ignore", category=DeprecationWarning, module=r".*")
    filterwarnings(action="ignore", module=r"torch.quantization")
    filterwarnings(action="ignore", category=UserWarning)

    datasets, n_classes = prepare_cifar2(folder=data_folder)
    sspace = search_space()
    quant_params = None
    collate_fn = None

    args = configuration("TRAIN")
    if not path.exists(args["ROOT"]):
        mkdir(args["ROOT"])
    time_init = time()
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
        models_path=path.join(args["ROOT"], "models"),
    )
    exp, data = sparse_exp.create_load_experiment()
    return exp, data
