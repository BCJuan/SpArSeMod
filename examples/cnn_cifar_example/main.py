from warnings import filterwarnings
from os import path, mkdir
from time import time
import numpy as np
from torch import nn as nn
from sparsemod.utils_data import configuration, bool_converter, str_to_list
from sparsemod.sparse import Sparse
from load_data import prepare_cifar10, prepare_mnist, prepare_cifar2
from cnn import search_space, Net, operations
from trainer import SimpleTrainer

if __name__ == "__main__":

    filterwarnings(action="ignore", category=DeprecationWarning, module=r".*")
    filterwarnings(action="ignore", module=r"torch.quantization")
    filterwarnings(action="ignore", category=UserWarning)

    datasets, n_classes = prepare_cifar2(folder="../data/data_cifar2")
    search_space = search_space()
    quant_params = None
    collate_fn = None

    if bool_converter(configuration("DEFAULT")["TRAIN"]):
        args = configuration("TRAIN")

        time_init = time()
        sparse_instance = Sparse(
            r1=int(args["R1"]),
            r2=int(args["R2"]),
            r3=int(args["R3"]),
            epochs1=int(args["EPOCHS1"]),
            epochs2=int(args["EPOCHS2"]),
            epochs3=int(args["EPOCHS3"]),
            name=str(args["NAME"]),
            root=args["ROOT"],
            objectives=str_to_list(args["OBJECTIVES"]),
            batch_size=int(args["BATCH_SIZE"]),
            morphisms=bool_converter(args["MORPHISMS"]),
            pruning=bool_converter(args["PRUNING"]),
            datasets=datasets,
            classes=n_classes,
            debug=bool_converter(args["DEBUG"]),
            search_space=search_space,
            net=Net,
            flops=int(args["FLOPS"]),
            quant_scheme=str(args["QUANT_SCHEME"]),
            quant_params=quant_params,
            collate_fn=collate_fn,
            splitter=bool_converter(args["SPLITTER"]),
            morpher_ops=operations,
            arc=bool_converter(args["ARC"]),
            cuda=str(args['CUDA_N']),
            trainer = SimpleTrainer
        )
        sparse_instance.run_sparse()
        time_end = time()
        diff_time = time_end - time_init
        print(diff_time)
        np.savetxt(path.join(args["ROOT"], "time.txt"), np.asarray([diff_time]))

