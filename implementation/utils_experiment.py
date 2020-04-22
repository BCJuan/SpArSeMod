# -*- coding: utf-8 -*-


from ax import Metric, Data, Runner, Experiment, OptimizationConfig
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.core.objective import ScalarizedObjective, Objective
from pandas import DataFrame, read_csv
import pickle
from os import path
from model import Trainer
from numpy import array, prod, max as maxim, log
from torch import randn, save, load
from torch.quantization import get_default_qconfig, prepare, convert, default_qconfig
import copy
from load_data import get_input_shape
# TODO: use original repo
from flops_counter_experimental import get_model_complexity_info


class AccuracyMetric(Metric):
    """
    Class for the accuracy metric
    """

    # TODO: stringt to call specific dataset, look at the trainer class

    def __init__(self, epochs, name, pruning, datasets, classes, net):
        super().__init__(name, lower_is_better=True)
        self.epochs = epochs
        self.trainer = Trainer(pruning=pruning, datasets=datasets)
        self.reload = False
        self.old_net = None
        self.pruning = pruning
        self.datasets = datasets
        self.classes = classes
        self.net = net

    def fetch_trial_data(self, trial):
        """
        Function to retrieve the trials data for this metric
        """
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            self.parametrization = arm.parameters
            result = self.train_evaluate(arm_name)
            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "mean": result,
                    "sem": 0.0,
                    "trial_index": trial.index,
                }
            )
        return Data(df=DataFrame.from_records(records))

    def train_evaluate(self, name):
        """
        Trains the network and evaluates its performance on the test set
        """

        # TODO: classes from outside
        self.trainer.load_dataloaders(self.parametrization.get("batch_size", 4))
        net_i = self.net(
            self.parametrization, classes=self.classes, datasets=self.datasets
        )
        net_i = self.trainer.train(
            net_i,
            self.parametrization,
            name,
            self.epochs,
            self.reload,
            self.old_net,
        )
        net_i.to("cpu")
        net_i.eval()
        net_i.fuse_model()
        net_i.qconfig = get_default_qconfig("fbgemm")
        prepare(net_i, inplace=True)
        _, net_i = self.trainer.evaluate(net_i, quant_mode=True)
        convert(net_i, inplace=True)
        result, net_i = self.trainer.evaluate(net_i, quant_mode=False)
        save(net_i.state_dict(), "./models/" + str(name) + "_qq" + ".pth")
        return 1 - result


class WeightMetric(Metric):
    """
    Class for the weight metric
    """

    def __init__(self, name, datasets, classes, net):
        super().__init__(name, lower_is_better=True)
        # TODO: maximum limit is nowadays hardcoded as 10**8, change to
        # variable
        self.top = log(10 ** 8)
        self.datasets = datasets
        self.classes = classes
        self.net = net

    def fetch_trial_data(self, trial):
        """
        Function to retrieve the trials data for this metric
        """
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            self.parametrization = arm.parameters
            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "mean": self.net_weighting(),
                    "sem": 0.0,
                    "trial_index": trial.index,
                }
            )
        return Data(df=DataFrame.from_records(records))

    def net_weighting(self):
        """
        Builds the network and evaluates how many parameters does it have
        """
        # TODO: classes from outside
        net_i = self.net(
            self.parametrization, classes=self.classes, datasets=self.datasets
        )
        n_params = int(sum((p != 0).sum() for p in net_i.parameters()))
        weight = log(n_params) / self.top
        return weight


# TODO: compute feature map outside Net and make it generalizable
class FeatureMapMetric(Metric):
    """
    Class for the weight metric
    """

    def __init__(self, bits, name, datasets, classes, net):
        super().__init__(name, lower_is_better=True)
        self.bits = bits
        # TODO: maximum limit is nowadays hardcoded as 10**8, change to
        # variable
        self.top = log(10 ** 8)
        self.datasets = datasets
        self.classes = classes
        self.net = net

    def fetch_trial_data(self, trial):
        """
        Function to retrieve the trials data for this metric
        """
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            self.parametrization = arm.parameters
            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "mean": self.net_weighting(),
                    "sem": 0.0,
                    "trial_index": trial.index,
                    # TODO: add time spent in each trial
                }
            )
        return Data(df=DataFrame.from_records(records))

    def net_weighting(self):
        """
        Builds the network and evaluates how many parameters does it have
        """
        # TODO: classes from outside
        net_i = self.net(
            self.parametrization, classes=self.classes, datasets=self.datasets
        )
        net_i.eval()
        # TODO: substitute inputs by a random tensor with batch 1
        # and delete the bb (batshcsize) division in maximum
        # TODO: obtain feature map as an external function as in weight
        macs, params, maxram = get_model_complexity_info(net_i, tuple(get_input_shape(self.datasets)), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
        # TODO: standarize_onjective
        return log(maxram) / self.top


class LatencyMetric(Metric):

    def __init__(self, name, datasets, classes, net, flops_capacity):
        super().__init__(name, lower_is_better=True)
        self.datasets = datasets
        self.classes = classes
        self.net = net
        self.flops_capacity = flops_capacity
        self.top = log(10**4)

    def fetch_trial_data(self, trial):
        """
        Function to retrieve the trials data for this metric
        """
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            self.parametrization = arm.parameters
            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "mean": self.latency_measure(),
                    "sem": 0.0,
                    "trial_index": trial.index,
                    # TODO: add time spent in each trial
                }
            )
        return Data(df=DataFrame.from_records(records))

    def latency_measure(self):
        """
        Returns in miliseconds
        """
        net_i = self.net(
            self.parametrization, classes=self.classes, datasets=self.datasets
        )
        macs, params, maxram = get_model_complexity_info(net_i, tuple(get_input_shape(self.datasets)), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
        miliseconds = macs*1000/self.flops_capacity
        return log(miliseconds)/self.top


class MyRunner(Runner):
    """
    Runner class for fetching and deploying evaluations
    """

    def run(self, trial):
        return {"name": str(trial.index)}


def get_experiment(
    bits, epochs, objectives, pruning, datasets, classes, search_space, net, flops
):
    """
    Main experiment function: establishes the experiment and defines
    the configuration with the scalarization with the appropriate metrics

    Args
    ----
    parameters:
        the configuration for training the network
    device:
        device for running the experiments

    Returns
    ------
    The experiment object
    """
    metric_list = [
        AccuracyMetric(
            epochs,
            name="accuracy",
            pruning=pruning,
            datasets=datasets,
            classes=classes,
            net=net,
        ),
        WeightMetric(name="weight", datasets=datasets, classes=classes, net=net),
        FeatureMapMetric(bits, name="ram", datasets=datasets, classes=classes, net=net),
        LatencyMetric(name='latency', datasets=datasets, classes=classes, net=net, flops_capacity=flops)
    ]
    experiment = Experiment(
        name="experiment_building_blocks", search_space=search_space,
    )

    if objectives > 1:
        weights = [1 / objectives for i in range(objectives)]
        objective = ScalarizedObjective(
            metrics=metric_list[:objectives], weights=weights, minimize=True,
        )
    else:
        objective = Objective(metric=metric_list[:objectives][0], minimize=True,)

    optimization_config = OptimizationConfig(objective=objective,)

    experiment.optimization_config = optimization_config
    experiment.runner = MyRunner()
    return experiment


def group_attach_and_save(data, new_data, exp, name, root, objectives):
    data, new_data = update_data(data, new_data)
    exp.attach_data(new_data)
    save_data(exp, data, name, root, objectives)
    return data, new_data, exp


def update_data(data, new_data):
    data = Data.from_multiple_data(data=[data, new_data]) if new_data else data
    return data, new_data


def save_data(exp, data, name=None, root=None, n_obj=None):
    """
    FUnction for saving data, experiment and runner
    """
    data.df.to_csv(path.join(root, name + ".csv"))
    metrics = [AccuracyMetric, WeightMetric, FeatureMapMetric, LatencyMetric]
    for i in range(n_obj):
        register_metric(metrics[i])
    register_runner(MyRunner)
    save(exp, path.join(root, name + ".json"))


def load_data(name, n_obj=None):
    metrics = [AccuracyMetric, WeightMetric, FeatureMapMetric, LatencyMetric]
    for i in range(n_obj):
        register_metric(metrics[i])
    register_runner(MyRunner)
    name = name + ".json"
    return load(name)


def pass_data_to_exp(csv):
    df = read_csv(csv, index_col=0)
    return Data(df=df)


def create_load_experiment(
    root, name, objectives, bits, epochs1, pruning, datasets, classes, search_space, net, flops
):
    """
    Creates or loads an experiment where all the data and trials 
    are going to be stored.

    It also controls the metrics and running the trials

    Args
    ----
    """
    if path.exists(path.join(root, name + ".json")):
        exp = load_data(path.join(root, name), objectives)
        data = pass_data_to_exp(path.join(root, name + ".csv"))
        exp.attach_data(data)
    else:
        exp = get_experiment(
            bits, epochs1, objectives, pruning, datasets, classes, search_space, net, flops
        )
        data = Data()
    return exp, data
