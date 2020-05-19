from warnings import filterwarnings
from os import path, mkdir
from sparse.utils_data import configuration, bool_converter # str_to_list
from sparse.sparse import sparse
from load_data import (
    prepare_cifar10, prepare_mnist, prepare_cifar2, prepare_cost, split_pad_n_pack,
    split_arrange_pad_n_pack, insample_pad_n_pack, split_arrange_pad_n_pack_3d,
    insample_arrange_pad_n_pack_3d
)
from torch import nn as nn
from test import ModelTester

# from architectures.rnn import search_space, Net, operations
# from architectures.cnn2d_cost import search_space, Net, operations
from architectures.cnn2d_plus_rnn_cost import search_space, Net, operations
# from architectures.cnn3d_plus_rnn_cost import search_space, Net, operations
# from architectures.cnn3d import search_space, Net, operations

if __name__ == "__main__":

    filterwarnings(action="ignore", category=DeprecationWarning, module=r".*")
    filterwarnings(action="ignore", module=r"torch.quantization")
    filterwarnings(action="ignore", category=UserWarning)

    datasets, n_classes = prepare_cost(image=True)
    search_space = search_space()
    quant_params = {nn.LSTM}
    # quant_params = None
    # quant_params = {nn.LSTM, nn.Linear, nn.GRU}
    # quant_params = {nn.LSTM, nn.Linear, nn.GRU, nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d, nn.MaxPool3d, nn.ReLU}
    # collate_fn = split_arrange_pad_n_pack
    collate_fn = split_arrange_pad_n_pack
    
    if bool_converter(configuration("DEFAULT")['TRAIN']):
        args = configuration("TRAIN")
        if not path.exists(args["ROOT"]):
            mkdir(args["ROOT"])
        sparse(
            r1=int(args["R1"]),
            r2=int(args["R2"]),
            r3=int(args["R3"]),
            bits=int(args["BITS"]),
            epochs1=int(args["EPOCHS1"]),
            epochs2=int(args["EPOCHS2"]),
            epochs3=int(args["EPOCHS3"]),
            name=str(args["NAME"]),
            root=args["ROOT"],
            objectives=int(args["OBJECTIVES"]),
            batch_size=int(args["BATCH_SIZE"]),
            desired_n_param=int(args["DESIRED_N_PARAM"]),
            desired_acc=float(args["DESIRED_ACC"]),
            desired_ram=int(args["DESIRED_RAM"]),
            morphisms=bool_converter(args["MORPHISMS"]),
            begin_sobol=bool_converter(args["BEGIN_SOBOL"]),
            scalarizations=bool_converter(args["SCALARIZATIONS"]),
            pruning=bool_converter(args["PRUNING"]),
            datasets=datasets,
            classes=n_classes,
            debug=bool_converter(args["DEBUG"]),
            search_space=search_space,
            net=Net,
            flops=int(args["FLOPS"]),
            desired_latency=int(args["DESIRED_LATENCY"]),
            quant_scheme=str(args['QUANT_SCHEME']),
            quant_params=quant_params,
            collate_fn=collate_fn,
            splitter=bool_converter(args['SPLITTER']),
            morpher_ops = operations
        )
    else:
        args = configuration("TEST")
        ModelTester(root=str(args["ROOT"]),
                    name=str(args["NAME"]),
                    arm=int(args['ARM']),
                    n_obj=int(args["OBJECTIVES"]),
                    epochs=int(args["EPOCHS"]),
                    pruning=bool_converter(args["PRUNING"]),
                    quant_scheme=str(args['QUANT_SCHEME']),
                    quant_params=quant_params,
                    collate_fn=collate_fn,
                    splitter=bool_converter(args['SPLITTER']),
                    net=Net
                    ).leave_one_out()
