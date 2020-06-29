from .utils_experiment import load_data, pass_data_to_exp
from os import path
from ..examples.load_data import prepare_cost
from torch import load
from .model import Trainer
from .utils_experiment import AccuracyMetric
from .sparse import reload_net
from numpy import zeros, save, mean as npmean, asarray
from .utils_data import get_shape_from_dataloader
import copy
from functools import partial
from tqdm import tqdm


class ModelTester(object):
    def __init__(
        self,
        root,
        name,
        arm,
        n_obj,
        epochs,
        pruning,
        quant_scheme,
        quant_params,
        collate_fn,
        splitter,
        net,
    ):

        self.root = root
        self.name = name
        self.arm = arm
        self.n_obj = n_obj
        self.epochs = epochs
        self.pruning = pruning
        self.quant_scheme = quant_scheme
        self.quant_params = quant_params
        self.collate_fn = collate_fn
        self.splitter = splitter
        self.net = net
        self.models_path = path.join(self.root, "models")

    def arm_parameters(self):
        exp = load_data(path.join(self.root, self.name), self.n_obj)
        data = pass_data_to_exp(path.join(self.root, self.name + ".csv"))
        exp.attach_data(data)
        return exp.trials[self.arm].arm.parameters

    def reload_net(self, arm_params, classes, input_shape, net):
        net = net(arm_params, classes, input_shape)
        # TODO: problem with batch and arm names
        model_filename = path.join(self.models_path, str(self.arm) + "_0.pth")
        net.load_state_dict(load(model_filename))
        return net

    def leave_one_out(self):
        n_subjects = 32
        results = []
        for i in tqdm(range(1, n_subjects)):
            datasets, n_classes = prepare_cost(
                test_subjects=[i], folder="../data/data_cost/files/"
            )
            params = self.arm_parameters()
            net = copy.copy(self.net)
            AccMetric = AccuracyMetric(
                self.epochs,
                "Accuracy Test",
                self.pruning,
                datasets,
                n_classes,
                self.net,
                self.quant_scheme,
                self.quant_params,
                self.collate_fn,
                self.splitter,
            )
            AccMetric.parametrization = self.arm_parameters()
            AccMetric.reload = True
            collate_fn = copy.copy(self.collate_fn)
            if self.splitter:
                collate_fn = partial(collate_fn, max_len=params.get("max_len"))
            AccMetric.trainer.load_dataloaders(
                params.get("batch_size", 4), collate_fn=collate_fn
            )
            input_shape = get_shape_from_dataloader(
                AccMetric.trainer.dataloader["train"], params
            )
            old_net = self.reload_net(params, n_classes, input_shape, net)
            AccMetric.old_net = old_net
            acc = AccMetric.train_evaluate(str(self.arm) + "_test_result")
            results.append(acc)
        save(str(self.arm) + "_test_result.npy", asarray(results))
