"""
Utils related to proper use of data, such as arguments or
shape computations. THis is a varied collection of utilities
"""
from json import loads
from pickle import load, dump
from configparser import ConfigParser
from torch.nn.utils.rnn import PackedSequence


def get_shape_from_dataloader(dataloader, params):
    """ Gets shape from dataloader. Either from image or sequence"""
    inp, _ = next(iter(dataloader))
    if isinstance(inp, PackedSequence):
        shape = (params.get("max_len"), inp.data.shape[-1])
    else:
        shape = tuple(inp.shape[1:])
    return shape


def str_to_list(conf):
    """Loads a string to a list"""
    return loads(conf)


def save_to_pickle(dic, name="./results/configurations.pkl"):
    """Pickle save function"""
    with open(name, "wb") as file:
        dump(dic, file)


def load_pickle(name="./results/configurations.pkl"):
    """Pickle load function"""
    with open(name, "rb") as file:
        return load(file)


def configuration(config):
    """Loads and parses configuration files"""
    pars = ConfigParser()
    with open("config.cfg") as conf_file:
        pars.read_file(conf_file)

    return pars[config]


def bool_converter(inp):
    """ COnverts a string in configuration file to a bool"""
    if inp == "False":
        return False
    elif inp == "True":
        return True
    else:
        raise "Boolean config parameters must be written as True or False"
