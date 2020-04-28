from json import loads
from pickle import load, dump
from configparser import ConfigParser
from torch.nn.utils.rnn import PackedSequence

def get_shape_from_dataloader(dataloader, params):
    input, label = next(iter(dataloader))
    if isinstance(input, PackedSequence):
        shape = (params.get('max_len'), input.data.shape[-1])
    else:
        shape =  tuple(input.shape[1:])
    return shape

def str_to_list(conf):
    return loads(conf)


def save_to_pickle(dic, name="./results/configurations.pkl"):
    with open(name, "wb") as file:
        dump(dic, file)


def load_pickle(name="./results/configurations.pkl"):
    with open(name, "rb") as file:
        return load(file)


def configuration(config):
    pars = ConfigParser()
    with open('config.cfg') as f:
        pars.read_file(f)

    return pars[config]


def bool_converter(input):
    if input == "False":
        return False
    elif input == "True":
        return True
    else:
        raise ("Boolean config parameters must be written as True or False")
