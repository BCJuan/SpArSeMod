# -*- coding: utf-8 -*-

from .utils_experiment import (
    MyRunner,
    WeightMetric,
    AccuracyMetric,
    FeatureMapMetric,
    LatencyMetric,
    SparseExperiment
)
from ax.core.observation import ObservationFeatures
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.factory import get_botorch
from tqdm import tqdm
from ax import Arm
from os import path, mkdir
from ax.core.data import Data
from functools import partial
from random import choice
from pandas import read_csv
from torch import Size, load, save
from copy import copy
from numpy import argmin, stack
from numpy.linalg import norm
from .utils_data import get_shape_from_dataloader
from .morpher import Morpher
from .heir import clean_models_return_pareto

# TODO: add raytune for distributing machine learning
# TODO: refactorize main
# TODO: change SobolMCSampler to NormalIIDSampler to avoid problems with
# high dimensional spaces q*n > 1111
# TODO: convert full/main process in a class with properties such as name, root, epochs and so
# and methods such as run model

class Sparse(object):

    def __init__(self, **kwargs):
        allowed_keys = {'r1', 'r2', 'r3', 'epochs1', 'epochs2', 'epochs3', 'name', 'root', 'objectives', 'batch_size',
                        'morphisms', 'pruning', 'datasets', 'classes', 'debug', 'search_space', 'net', 'flops', 'quant_scheme',
                        'quant_params', 'collate_fn', 'splitter', 'morpher_ops'}

        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

    def run_sparse(self):
        sparse_exp = SparseExperiment(self.epochs1, **self.__dict__)
        
        self.exp, self.data = sparse_exp.create_load_experiment()

        sobol = get_sobol(self.exp.search_space)
        sobol = self.run_model(self.r1, sobol, self.epochs1, model_type="random")

        self.exp.optimization_config.objective.metrics[0].epochs = self.epochs2
        botorch = get_botorch(experiment=self.exp, data=self.data)
        botorch = self.run_model(self.r2, botorch, self.epochs2, model_type="bo")

        if self.morphisms:
            self.pareto_arms = clean_models_return_pareto(self.data)
            self.develop_morphisms(botorch)

    def run_model(self, r, model, epochs, model_type):
        for _ in tqdm(range(r)):
            self.pareto_arms = clean_models_return_pareto(self.data)
            model, new_data = self.model_loop(model)

            if not new_data.df.empty:
                new_data = self.group_attach_and_save(new_data)
                if model_type == "bo":
                    model.update(new_data, self.exp)
        # TODO: add raise error for empty data
        self.pareto_arms = clean_models_return_pareto(self.data)
        return model

    def model_loop(self, model):
        """
        Makes a loop of random generation and fetching
        """
        if self.batch_size > 1:
            n = self.batch_size
            trial = self.exp.new_batch_trial(generator_run=model.gen(n=n)).run()
        else:
            n = 1
            trial = self.exp.new_trial(generator_run=model.gen(n=n)).run()

        trial, new_data = self.run_trial(trial)
        return model, new_data

    def run_trial(self, trial):
        try:
            new_data = self.exp._fetch_trial_data(trial.index)
            self.exp.trials[trial.index].mark_completed()
        except (RuntimeError, IndexError) as e:
            self.exp.trials[trial.index].mark_failed()
            new_data = Data()
            if self.debug:
                print(e)
        return trial, new_data

    def group_attach_and_save(self, new_data):
        new_data = self.update_data(new_data)
        self.exp.attach_data(new_data)
        self.save_data()
        return new_data


    def update_data(self, new_data):
        self.data = Data.from_multiple_data(data=[self.data, new_data]) if new_data else self.data
        return new_data

    def save_data(self):
        """
        FUnction for saving data, experiment and runner
        """
        self.data.df.to_csv(path.join(self.root, self.name + ".csv"))
        metrics = [AccuracyMetric, WeightMetric, FeatureMapMetric, LatencyMetric]
        for i in range(self.objectives):
            register_metric(metrics[i])
        register_runner(MyRunner)
        save(self.exp, path.join(self.root, self.name + ".json"))


    def develop_morphisms(self, model):
        self.morpher = Morpher(operations=self.morpher_ops)
        self.morpher.retrieve_best_configurations(self.exp, self.pareto_arms)
        self.exp.optimization_config.objective.metrics[0].epochs = self.epochs3
        self.exp.optimization_config.objective.metrics[0].reload = True
        for _ in tqdm(range(self.r3)):
            self.morphism_loop(model)


    def morphism_loop(self, model):
        morpher = Morpher(self.morpher_ops)
        morpher.retrieve_best_configurations(self.exp, self.pareto_arms)
        # TODO: number of morphs should be passed through params in a sparse class
        new_configs = morpher.apply_morphs(n_morphs=20)
        new_arm = ei_new_arm(model, new_configs)

        collate_fn_p = copy(self.collate_fn)
        if self.exp.optimization_config.objective.metrics[0].splitter:
            collate_fn_p = partial(
                collate_fn_p, max_len=self.exp.arms_by_name[new_arm[1]].parameters.get('max_len'))
        self.exp.optimization_config.objective.metrics[0].trainer.load_dataloaders(self.exp.arms_by_name[new_arm[1]].parameters.get("batch_size", 4), 
                                collate_fn=collate_fn_p)
        input_shape = get_shape_from_dataloader(self.exp.optimization_config.objective.metrics[0].trainer.dataloader['train'],
                                    self.exp.arms_by_name[new_arm[1]].parameters)

        old_net = reload_net(self.exp, new_arm[1], self.classes, input_shape, self.net)
        self.exp.optimization_config.objective.metrics[0].old_net = old_net
        trial = (
            self.exp.new_trial()
            .add_arm(
                Arm(
                    name=str(list(self.exp.data_by_trial.keys())[-1] + 1) + "_0",
                    parameters=new_configs[new_arm],
                )
            )
            .run()
        )
        trial, new_data = self.run_trial(trial)
        if not new_data.df.empty:
            new_data = self.group_attach_and_save(new_data)

        self.pareto_arms = clean_models_return_pareto(self.data)
        self.morpher.retrieve_best_configurations(self.exp, self.pareto_arms)
        model.update(new_data, self.exp)


def ei_new_arm(model, new_configs):
    # TODO: use better prediction minimum
    obs_feats = [ObservationFeatures(parameters=i) for i in new_configs.values()]
    f, cov = model.predict(obs_feats)
    predicted_values = stack(list(f.values()))
    min_pred_idx = argmin([norm(predicted_values[:, i]) for i in range(predicted_values.shape[1])])
    new_arm = list(new_configs)[min_pred_idx]
    return new_arm 


# TODO: delete hardcoded model folder name, loook in other parts of python
# files for hardcoded folder model name
def reload_net(exp, arm, classes, input_shape, net):
    net = net(exp.arms_by_name[arm].parameters, classes, input_shape)
    model_filename = "./models/" + arm + ".pth"
    net.load_state_dict(load(model_filename))
    return net
