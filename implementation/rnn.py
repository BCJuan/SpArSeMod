from ax import SearchSpace, ParameterType, RangeParameter, ChoiceParameter
from torch.nn import LSTM, Linear, Sequential, Module, GRU
from random import random, randint

def search_space():
    """
    Defines the network search space parameters and returns the search
    space object

    Returns
    ------
    Search space object

    """
    params = []
    ##### CONV BLOCKS ######################################################################
    params.append(RangeParameter(name="rnn_layers", parameter_type=ParameterType.INT,
                            lower=1, upper=5))
    params.append(RangeParameter(name="neurons_layers", parameter_type=ParameterType.INT,
                        lower=8, upper=512))
    params.append(RangeParameter(name="rnn_dropout", parameter_type=ParameterType.FLOAT, lower=0.1, upper=0.5))
    params.append(ChoiceParameter(name="cell_type", parameter_type=ParameterType.STRING, values=['lstm', 'gru']))
    #######################################################################################

    ### FC BLOCKS ########################################################################
    params.append(RangeParameter(name="fc_layers", lower = 0, upper=1, parameter_type=ParameterType.INT))
    params.append(RangeParameter(name="neurons_fc_layer_1", lower=10 , upper=128, parameter_type=ParameterType.INT))
    
    #### SPLIT SEUENCES  ##################################################
    params.append(RangeParameter(name='max_len', lower = 50, upper=250, parameter_type=ParameterType.INT))
    
    ###### MANDATORY PARAMETERS ############################################
    params.append(RangeParameter(
        name="learning_rate",
        lower=0.0001,
        upper=0.01,
        parameter_type=ParameterType.FLOAT,
    ))
    params.append(RangeParameter(
        name="learning_gamma", lower=0.9, upper=0.99, parameter_type=ParameterType.FLOAT
    ))
    params.append(RangeParameter(
        name="learning_step", lower=1, upper=10000, parameter_type=ParameterType.INT
    ))
    params.append(RangeParameter(
        name="prune_threshold",
        lower=0.05,
        upper=0.9,
        parameter_type=ParameterType.FLOAT,
    ))
    params.append(RangeParameter(
        name="batch_size", lower=2, upper=8, parameter_type=ParameterType.INT
    ))
    ########################################################################

    search_space = SearchSpace(
        parameters=params
    )

    return search_space


class Net(Module):

    def __init__(self, params, classes, input_shape):
        super(Net, self).__init__()
        # input shpae coming from dataloader, hence we only have to take features, ie last dim
        self.input_dim = input_shape[-1]
        self.params = params
        self.layers = params.get("rnn_layers", 1)
        if params.get('cell_type') == 'lstm':
            cell = LSTM
        elif params.get('cell_type') == 'gru':
            cell = GRU
        
        self.cell = cell(self.input_dim, params.get("neurons_layers", 64), batch_first=True,
                        num_layers=self.layers,
                        dropout=params.get("rnn_dropout", 0.1))
        fc = []
        for i in range(params.get("fc_layers")):
            fc.append(Linear(params.get("neurons_layers", 64), params.get("neurons_fc_layer_" + str(i + 1))))
        
        fc.append(Linear( params.get("neurons_fc_layer_" + str(params.get("fc_layers" )), params.get("neurons_layers", 64)), classes))
        self.fc = Sequential(*fc)


    def forward(self, sequence):
       
        cell_out, self.hidden = self.cell(sequence)
        if self.params.get('cell_type') == 'lstm':
            return self.fc(self.hidden[0][self.layers - 1])     
        else:
            return self.fc(self.hidden[self.layers - 1]) 


def change_layers(config):
    n_layers = config['rnn_layers']
    if n_layers == 5:
        config['rnn_layers'] = config['rnn_layers'] - 1
    elif n_layers == 1:
        config['rnn_layers'] = config['rnn_layers'] + 1
    else:
        if random() < 0.5:
            config['rnn_layers'] = config['rnn_layers'] + 1
        else:
            config['rnn_layers'] = config['rnn_layers'] - 1
    return config
    

def change_n_neurons(config):
    n_neurons = config['neurons_layers']
    if n_neurons > 512:
        config['neurons_layers'] = config['neurons_layers'] - 32
    elif n_neurons < 33:
        config['neurons_layers'] = config['neurons_layers'] + 32
    else:
        if random() < 0.5:
            config['neurons_layers'] = config['neurons_layers'] + 32
        else:
            config['neurons_layers'] = config['neurons_layers'] - 32
    return config


def change_dropout(config):
    dropout = config['rnn_dropout']
    if dropout == 0.5:
        config['rnn_dropout'] = config['rnn_dropout'] - 0.05
    elif dropout < 0.1:
        config['rnn_dropout'] = config['rnn_dropout'] + 0.05
    else:
        if random() < 0.5:
            config['rnn_dropout'] = config['rnn_dropout'] + 0.05
        else:
            config['rnn_dropout'] = config['rnn_dropout'] - 0.05
    return config


def change_fc(config):
    fc = config['fc_layers']
    if fc == 1:
        config['fc_layers'] = config['fc_layers'] - 1
    elif fc == 0:
        config['fc_layers'] = config['fc_layers'] + 1
    return config


def change_fc_neurons(config):
    n_neurons = config['neurons_fc_layer_1']
    if n_neurons > 128:
        config['neurons_fc_layer_1'] = config['neurons_fc_layer_1'] - 5
    elif n_neurons < 10:
        config['neurons_fc_layer_1'] = config['neurons_fc_layer_1'] + 5
    else:
        if random() < 0.5:
            config['neurons_fc_layer_1'] = config['neurons_fc_layer_1'] + 5
        else:
            config['neurons_fc_layer_1'] = config['neurons_fc_layer_1'] - 5
    return config


def change_max_len(config):
    max_len = config['max_len']
    if max_len > 250:
        config['max_len'] = config['max_len'] - 10
    elif max_len < 50:
        config['max_len'] = config['max_len'] + 10
    else:
        if random() < 0.5:
            config['max_len'] = config['max_len'] + 10
        else:
            config['max_len'] = config['max_len'] - 10
    return config


operations =  {
    "change_layers": change_layers,
    "change_n_neurons": change_n_neurons,
    "change_dropout": change_dropout,
    "change_fc": change_fc,
    "change_fc_neurons": change_fc_neurons,
    "change_max_len": change_max_len
}