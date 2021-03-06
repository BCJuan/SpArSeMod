# -*- coding: utf-8 -*-
"""
Experiment Class for sparse
Metrics classes
Runner class
Helper load and save functions
"""
# pylint: disable=E1101, W0221, W0201
from os import path
import copy
from functools import partial
from ax import Metric, Data, Runner, Experiment, OptimizationConfig
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.core.objective import MultiObjective, Objective
from pandas import DataFrame, read_csv
from numpy import log
from torch import save, load
from operator import itemgetter
from .utils_data import get_shape_from_dataloader
from .quant_n_prune import quant
from .model import SimpleTrainer

# TODO: use original repo
from .flops_counter_experimental import get_model_complexity_info


class SparseExperiment:
    """
    Class for the Sparse Experiment
    """

    def __init__(self, epochs, **kwargs):
        allowed_keys = {
            "root",
            "name",
            "objectives",
            "epochs",
            "pruning",
            "datasets",
            "classes",
            "search_space",
            "net",
            "flops",
            "quant_scheme",
            "quant_params",
            "collate_fn",
            "splitter",
            "models_path",
            "cuda",
            "trainer"
        }
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
        self.epochs = epochs

    def create_load_experiment(self):
        """ Creates the experiment or loads it from the json file"""
        if path.exists(path.join(self.root, self.name + ".json")):
            exp = load_data(path.join(self.root, self.name), self.objectives)
            data = pass_data_to_exp(path.join(self.root, self.name + ".csv"))
            exp.attach_data(data)
        else:
            exp = self.get_experiment()
            data = Data()
        return exp, data

    def get_experiment(self):
        """ Creates the experiment defining the metrics and the configuration"""
        metric_list = [
            AccuracyMetric(
                self.epochs,
                name="error",
                pruning=self.pruning,
                datasets=self.datasets,
                classes=self.classes,
                net=self.net,
                quant_scheme=self.quant_scheme,
                quant_params=self.quant_params,
                collate_fn=self.collate_fn,
                splitter=self.splitter,
                models_path=self.models_path,
                cuda=self.cuda,
                trainer=self.trainer
            ),
            WeightMetric(
                name="weight",
                datasets=self.datasets,
                classes=self.classes,
                net=self.net,
                collate_fn=self.collate_fn,
                splitter=self.splitter,
                trainer=self.trainer
            ),
            FeatureMapMetric(
                name="ram",
                datasets=self.datasets,
                classes=self.classes,
                net=self.net,
                collate_fn=self.collate_fn,
                splitter=self.splitter,
                trainer=self.trainer
            ),
            LatencyMetric(
                name="latency",
                datasets=self.datasets,
                classes=self.classes,
                net=self.net,
                flops_capacity=self.flops,
                collate_fn=self.collate_fn,
                splitter=self.splitter,
                trainer=self.trainer
            ),
        ]
        experiment = Experiment(
            name="experiment_building_blocks", search_space=self.search_space
        )
        metrics = list(itemgetter(*self.objectives)(metric_list))
        if len(self.objectives) > 1:
            objective = MultiObjective(
                metrics=metrics, minimize=True
            )
        else:
            objective = Objective(
                metric=metrics[0], minimize=True
            )

        optimization_config = OptimizationConfig(objective=objective)
        experiment.optimization_config = optimization_config
        experiment.runner = MyRunner()
        return experiment


class AccuracyMetric(Metric):
    """
    Class for the accuracy metric
    """

    # TODO: stringt to call specific dataset, look at the trainer class

    def __init__(
        self,
        epochs,
        name,
        pruning,
        datasets,
        classes,
        net,
        quant_scheme,
        quant_params=None,
        collate_fn=None,
        splitter=False,
        models_path=None,
        cuda="cuda:0",
        trainer=None
    ):
        super().__init__(name, lower_is_better=True)
        self.epochs = epochs

        if trainer:
            self.trainer = trainer(pruning=pruning, datasets=datasets, models_path=models_path, cuda=cuda)
        else:
            self.trainer = SimpleTrainer(
                pruning=pruning, datasets=datasets, models_path=models_path, cuda=cuda
            )
        self.reload = False
        self.old_net = None
        self.pruning = pruning
        self.datasets = datasets
        self.classes = classes
        self.net = net
        self.quant_scheme = quant_scheme
        self.quant_params = quant_params
        self.collate_fn = collate_fn
        self.splitter = splitter
        self.models_path = models_path
        self.cuda = cuda

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
        collate_fn = copy.copy(self.collate_fn)
        if self.splitter:
            collate_fn = partial(
                collate_fn, max_len=self.parametrization.get("max_len")
            )
        self.trainer.load_dataloaders(
            self.parametrization.get("batch_size", 4), collate_fn=collate_fn
        )
        input_shape = get_shape_from_dataloader(
            self.trainer.dataloader["train"], self.parametrization
        )
        net_i = self.net(
            self.parametrization, classes=self.classes, input_shape=input_shape
        )
        net_i = self.trainer.train(
            net_i, self.parametrization, name, self.epochs, self.reload, self.old_net
        )
        net_i = quant(net_i, self.quant_scheme, self.trainer, self.quant_params)
        result, net_i = self.trainer.evaluate(net_i, quant_mode=False)
        save(
            net_i.state_dict(), path.join(self.models_path, str(name) + "_qq" + ".pth")
        )
        return 1 - result


class WeightMetric(Metric):
    """
    Class for the weight metric
    """

    def __init__(self, name, datasets, classes, net, collate_fn, splitter, trainer=None):
        super().__init__(name, lower_is_better=True)
        # TODO: maximum limit is nowadays hardcoded as 10**8, change to
        # variable
        self.top = log(10 ** 8)
        self.classes = classes
        self.net = net
        if trainer:
            self.trainer = trainer(pruning=True, datasets=datasets)
        else:
            self.trainer = SimpleTrainer(pruning=True, datasets=datasets)
        self.collate_fn = collate_fn
        self.splitter = splitter

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
        collate_fn = copy.copy(self.collate_fn)
        if self.splitter:
            collate_fn = partial(
                collate_fn, max_len=self.parametrization.get("max_len")
            )
        self.trainer.load_dataloaders(
            self.parametrization.get("batch_size", 4), collate_fn=collate_fn
        )
        input_shape = get_shape_from_dataloader(
            self.trainer.dataloader["train"], self.parametrization
        )
        net_i = self.net(
            self.parametrization, classes=self.classes, input_shape=input_shape
        )
        n_params = int(sum((p != 0).sum() for p in net_i.parameters()))
        weight = log(n_params) / self.top
        return weight


# TODO: compute feature map outside Net and make it generalizable
class FeatureMapMetric(Metric):
    """
    Class for the weight metric
    """

    def __init__(self, name, datasets, classes, net, collate_fn, splitter, trainer=None):
        super().__init__(name, lower_is_better=True)
        # TODO: maximum limit is nowadays hardcoded as 10**8, change to
        # variable
        self.top = log(10 ** 8)
        self.classes = classes
        self.net = net
        if trainer:
            self.trainer = trainer(pruning=True, datasets=datasets)
        else:
            self.trainer = SimpleTrainer(pruning=True, datasets=datasets)
        self.collate_fn = collate_fn
        self.splitter = splitter

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
        collate_fn = copy.copy(self.collate_fn)
        if self.splitter:
            collate_fn = partial(
                collate_fn, max_len=self.parametrization.get("max_len")
            )
        self.trainer.load_dataloaders(
            self.parametrization.get("batch_size", 4), collate_fn=collate_fn
        )
        input_shape = get_shape_from_dataloader(
            self.trainer.dataloader["train"], self.parametrization
        )
        net_i = self.net(
            self.parametrization, classes=self.classes, input_shape=input_shape
        )
        net_i.eval()

        _, _, maxram = get_model_complexity_info(
            net_i,
            input_shape,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        # TODO: standarize_onjective
        return log(maxram) / self.top


class LatencyMetric(Metric):
    """ IMplements latency according to the number of operations in the network"""

    def __init__(
        self, name, datasets, classes, net, flops_capacity, collate_fn, splitter,
        trainer=None
    ):
        super().__init__(name, lower_is_better=True)
        self.classes = classes
        self.net = net
        self.flops_capacity = flops_capacity
        self.top = log(10 ** 4)
        if trainer:
            self.trainer = trainer(pruning=True, datasets=datasets)
        else:
            self.trainer = SimpleTrainer(pruning=True, datasets=datasets)
        self.collate_fn = collate_fn
        self.splitter = splitter

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
        collate_fn = copy.copy(self.collate_fn)
        if self.splitter:
            collate_fn = partial(
                collate_fn, max_len=self.parametrization.get("max_len")
            )
        self.trainer.load_dataloaders(
            self.parametrization.get("batch_size", 4), collate_fn=collate_fn
        )
        input_shape = get_shape_from_dataloader(
            self.trainer.dataloader["train"], self.parametrization
        )
        net_i = self.net(
            self.parametrization, classes=self.classes, input_shape=input_shape
        )
        # input shape can be an image CxHxW or a sequence LxF
        macs, _, _ = get_model_complexity_info(
            net_i,
            input_shape,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        miliseconds = macs * 1000 / self.flops_capacity
        # TODO: is necessary this add to avoid negatives?
        # https://math.stackexchange.com/questions/1111041/showing-y%E2%89%88x-for-small-x-if-y-logx1
        return log(miliseconds + 1) / self.top


class MyRunner(Runner):
    """
    Runner class for fetching and deploying evaluations
    """

    def run(self, trial):
        return {"name": str(trial.index)}


def load_data(name, n_obj=None):
    """ Loads the data from the experiment file json"""
    metrics = [AccuracyMetric, WeightMetric, FeatureMapMetric, LatencyMetric]
    metrics_register = list(itemgetter(*n_obj)(metrics))
    for i in metrics_register:
        register_metric(i)
    register_runner(MyRunner)
    name = name + ".json"
    return load(name)


def pass_data_to_exp(csv):
    """Loads the values from each of the evaluations to be further
    passed to a experiment"""
    dataframe = read_csv(csv, index_col=0)
    return Data(df=dataframe)
