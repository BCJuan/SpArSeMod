from ax import SearchSpace, ParameterType, RangeParameter
from torch.nn import LSTM, Linear, Sequential, Module


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

    #######################################################################################

    ### FC BLOCKS ########################################################################
    params.append(RangeParameter(name="fc_layers", lower = 0, upper=1, parameter_type=ParameterType.INT))
    for i in range(1, 2):
        params.append(RangeParameter(name="neurons_fc_layer_" + str(i), lower=10 , upper=128, parameter_type=ParameterType.INT))
    
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
        self.layers = params.get("rnn_layers", 1)

        self.cell = LSTM(self.input_dim, params.get("neurons_layers", 64), batch_first=True,
                        num_layers=self.layers,
                        dropout=params.get("rnn_dropout", 0.1))
        fc = []
        for i in range(params.get("fc_layers")):
            fc.append(Linear(params.get("neurons_layers", 64), params.get("neurons_fc_layer_" + str(i + 1))))
        
        fc.append(Linear( params.get("neurons_fc_layer_" + str(params.get("fc_layers" )), params.get("neurons_layers", 64)), classes))
        self.fc = Sequential(*fc)
    def forward(self, sequence):
        cell_out, self.hidden = self.cell(sequence)
        y = self.fc(self.hidden[0][self.layers - 1])     
        return y