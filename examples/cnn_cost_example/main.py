from warnings import filterwarnings
from os import path, mkdir
from time import time
import numpy as np
from torch import nn as nn, manual_seed
from sparsemod.utils_data import configuration, bool_converter, str_to_list
from sparsemod.sparse import Sparse
from load_data import prepare_cost
from test import ModelTester
from cnn2d_cost import search_space, Net, operations, split_arrange_pad_n_pack


if __name__ == "__main__":

    manual_seed(42)
    np.random.seed(42)

    filterwarnings(action="ignore", category=DeprecationWarning, module=r".*")
    filterwarnings(action="ignore", module=r"torch.quantization")
    filterwarnings(action="ignore", category=UserWarning)

    datasets, n_classes = prepare_cost(folder="../data/data_cost/files", image=True)
    search_space = search_space()
    quant_params = None
    collate_fn = split_arrange_pad_n_pack

    if bool_converter(configuration("DEFAULT")["TRAIN"]):
        args = configuration("TRAIN")
        if not path.exists(args["ROOT"]):
            mkdir(args["ROOT"])
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
            cuda=str(args['CUDA_N'])
        )
        sparse_instance.run_sparse()
        time_end = time()
        diff_time = time_end - time_init
        print(diff_time)
        np.savetxt(path.join(args["ROOT"], "time.txt"), np.asarray([diff_time]))
    else:
        args = configuration("TEST")
        ModelTester(
            root=str(args["ROOT"]),
            name=str(args["NAME"]),
            arm=str(args["ARM"]),
            n_obj=str_to_list(args["OBJECTIVES"]),
            epochs=int(args["EPOCHS"]),
            pruning=bool_converter(args["PRUNING"]),
            quant_scheme=str(args["QUANT_SCHEME"]),
            quant_params=quant_params,
            collate_fn=collate_fn,
            splitter=bool_converter(args["SPLITTER"]),
            net=Net,
        ).leave_one_out()
