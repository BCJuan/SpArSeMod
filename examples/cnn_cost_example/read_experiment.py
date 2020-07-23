from warnings import filterwarnings
from os import path, mkdir
from time import time
import numpy as np
from torch import nn as nn, manual_seed
from sparsemod.utils_data import configuration, bool_converter
from sparsemod.utils_experiment import SparseExperiment

from cnn2d_cost import search_space, Net, operations, split_arrange_pad_n_pack
from load_data import  prepare_cost

def load_experiment(data_folder="../data/data_cost/files"):

    manual_seed(42)
    np.random.seed(42)

    filterwarnings(action="ignore", category=DeprecationWarning, module=r".*")
    filterwarnings(action="ignore", module=r"torch.quantization")
    filterwarnings(action="ignore", category=UserWarning)

    datasets, n_classes = prepare_cost(folder=data_folder, image=True)
    sspace = search_space()   
    quant_params = None
    collate_fn = split_arrange_pad_n_pack


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
        models_path =path.join(args["ROOT"], "models")
    )
    exp, data = sparse_exp.create_load_experiment()
    return exp, data 

        