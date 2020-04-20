from warnings import filterwarnings
from os import path, mkdir
from utils_data import configuration, bool_converter # str_to_list
from sparse import sparse
from load_data import prepare_cifar10, prepare_mnist, prepare_cifar2

from cnn import search_space, Net

if __name__ == "__main__":

    filterwarnings(action="ignore", category=DeprecationWarning, module=r".*")
    filterwarnings(action="ignore", module=r"torch.quantization")
    args = configuration("DEFAULT")

    if not path.exists(args["ROOT"]):
        mkdir(args["ROOT"])

    datasets, n_classes = prepare_cifar2()
    search_space = search_space()

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
    )
