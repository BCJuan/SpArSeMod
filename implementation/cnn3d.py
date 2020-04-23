from ax import SearchSpace, ParameterType, RangeParameter

# TODO: include batch norm and try to make it static 
# TODO: if not, build it for dynamic
def search_space_static():
    """
    Defines the network search space parameters and returns the search
    space object

    Returns
    ------
    Search space object

    """
    params = []
    ##### CONV BLOCKS ######################################################################
    params.append(RangeParameter(name="conv3d_blocks", parameter_type=ParameterType.INT,
                            lower=1, upper=3))

    for i in range(1, 3):
        for j in range(1, 3):
            params.append(RangeParameter(name="block_" + str(i) + "_conv_" + str(j) + "_channels", parameter_type=ParameterType.INT,
                                    lower=5, upper=100))
            params.append(RangeParameter(name="block_" + str(i) + "_conv_" + str(j) + "_filtersize", parameter_type=ParameterType.INT,
                                    lower=3, upper=10))
        params.append(RangeParameter(
        name="drop_" + str(i), lower=0.1, upper=0.5, parameter_type=ParameterType.FLOAT
        ))
    #######################################################################################

    ### FC BLOCKS ########################################################################
    params.append(RangeParameter(name="fc_blocks"))
    
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

class Net(nn.Module):

    def  __init__(self, param, classes, datasets):
        super().__init__()
