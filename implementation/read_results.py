from sparse.utils_experiment import load_data, pass_data_to_exp
from os import path

def arm_parameters(name, arm, root, n_obj=4):
    exp = load_data(path.join(root, name), n_obj)
    data = pass_data_to_exp(path.join(root, name + ".csv"))
    exp.attach_data(data)
    print(exp.trials[arm].arm.parameters)