# -*- coding: utf-8 -*-

from .utils_experiment import (
    get_experiment,
    MyRunner,
    WeightMetric,
    AccuracyMetric,
    FeatureMapMetric,
    create_load_experiment,
    group_attach_and_save,
)
from ax.core.observation import ObservationFeatures
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.factory import get_botorch
from tqdm import tqdm
from ax import Arm
from os import path, mkdir
from ax.core.data import Data
from .morpher import Morpher
from .heir import clean_models_return_pareto
from random import choice
from pandas import read_csv
from torch import Size, load
from copy import copy
from .utils_data import get_shape_from_dataloader
from functools import partial
# TODO: add raytune for distributing machine learning
# TODO: refactorize main
# TODO: change SobolMCSampler to NormalIIDSampler to avoid problems with
# high dimensional spaces q*n > 1111
# TODO: convert full/main process in a class with properties such as name, root, epochs and so
# and methods such as run model


def sparse(
    r1,
    r2,
    r3,
    bits,
    epochs1,
    epochs2,
    epochs3,
    name,
    root,
    objectives,
    batch_size,
    desired_n_param,
    desired_acc,
    desired_ram,
    morphisms,
    begin_sobol,
    scalarizations,
    pruning,
    datasets,
    classes,
    debug,
    search_space,
    net,
    flops,
    desired_latency,
    quant_scheme,
    quant_params=None,
    collate_fn=None,
    splitter=False,
    morpher_ops=None
):
    """
    Main function for running the random evaluation points and
    the points extracted from the gaussian process.

    Args
    ----
    n_rounds (int):
        number of optimization rounds
    n_random_round (int):
        numnber of random points to evaluate at each global round
    n_batches_round (int):
        number of GP rounds to evaluate and generate new points at each
        global round
    batch_size (int):
        batch size for the guassian process evaluation
    bits (int):
        number of bits to compute the neural network size

    """

    # creates or loads experiment
    exp, data = create_load_experiment(
        root,
        name,
        objectives,
        bits,
        epochs1,
        pruning,
        datasets,
        classes,
        search_space,
        net,
        flops,
        quant_scheme,
        quant_params,
        collate_fn,
        splitter
    )


    # 1. sobol process
    if begin_sobol:
        sobol = get_sobol(exp.search_space)
        exp, data, pareto_arms, sobol = run_model(
            r1,
            exp,
            sobol,
            data,
            name,
            root,
            objectives,
            epochs1,
            model_type="random",
            debug=debug,
        )

    # 2. botorch process
    exp.optimization_config.objective.metrics[0].epochs = epochs2
    botorch = get_botorch(experiment=exp, data=data)
    exp, data, pareto_arms, botorch = run_model(
        r2,
        exp,
        botorch,
        data,
        name,
        root,
        objectives,
        epochs2,
        model_type="bo",
        debug=debug,
    )

    # 3. morphisms
    if morphisms:
        pareto_arms = clean_models_return_pareto(data)
        develop_morphisms(
            r3,
            exp,
            pareto_arms,
            data,
            name,
            root,
            objectives,
            epochs3,
            datasets,
            classes,
            debug,
            net,
            morpher_ops,
            botorch,
            collate_fn
        )


def run_model(
    r,
    exp,
    model,
    data,
    name,
    root,
    objectives,
    epochs,
    model_type,
    debug,
):
    for _ in tqdm(range(r)):
        pareto_arms = clean_models_return_pareto(data)
        exp, model, new_data = model_loop(model, 1, exp, debug)

        if not new_data.df.empty:
            data, new_data, exp = group_attach_and_save(
                data, new_data, exp, name, root, objectives
            )
            if model_type == "bo":
                model.update(new_data, exp)

    pareto_arms = clean_models_return_pareto(data)
    return exp, data, pareto_arms, model


def develop_morphisms(
    r3,
    exp,
    pareto_arms,
    data,
    name,
    root,
    objectives,
    epochs3,
    datasets,
    classes,
    debug,
    net,
    morpher_ops,
    model,
    collate_fn
):
    morpher = Morpher(operations=morpher_ops)
    morpher.retrieve_best_configurations(exp, pareto_arms)
    exp.optimization_config.objective.metrics[0].epochs = epochs3
    exp.optimization_config.objective.metrics[0].reload = True
    for _ in tqdm(range(r3)):
        morphism_loop(morpher, exp, collate_fn, classes, net, debug, objectives, root, name, model, data)


def morphism_loop(morpher, exp, collate_fn, classes, net, debug, objectives, root, name, model, data):
    new_configs = morpher.apply_morphs()
    # TODO: new configuration should be passed through acquisiton
    # function not random with chocie -> use model predict
    obs_feats = [ObservationFeatures(parameters=i) for i in new_configs.values()]

    new_arm = choice(list(new_configs))
    print(new_configs[new_arm])
    f, cov = model.predict(obs_feats)
    print(f.shape, f)
    collate_fn_p = copy(collate_fn)
    if exp.optimization_config.objective.metrics[0].splitter:
        collate_fn_p = partial(
            collate_fn, max_len=exp.arms_by_name[new_arm[1]].parameters.get('max_len'))
    exp.optimization_config.objective.metrics[0].trainer.load_dataloaders(exp.arms_by_name[new_arm[1]].parameters.get("batch_size", 4), 
                            collate_fn=collate_fn_p)
    input_shape = get_shape_from_dataloader(exp.optimization_config.objective.metrics[0].trainer.dataloader['train'],
                                exp.arms_by_name[new_arm[1]].parameters)

    old_net = reload_net(exp, new_arm[1], classes, input_shape, net)
    exp.optimization_config.objective.metrics[0].old_net = old_net
    trial = (
        exp.new_trial()
        .add_arm(
            Arm(
                name=str(list(exp.data_by_trial.keys())[-1] + 1) + "_0",
                parameters=new_configs[new_arm],
            )
        )
        .run()
    )
    exp, trial, new_data = run_trial(exp, trial, debug)
    if not new_data.df.empty:
        data, new_data, exp = group_attach_and_save(
            data, new_data, exp, name, root, objectives
        )

    pareto_arms = clean_models_return_pareto(data)
    morpher.retrieve_best_configurations(exp, pareto_arms)

def model_loop(model, batch_size, experiment, debug):
    """
    Makes a loop of random generation and fetching
    """
    if batch_size > 1:
        n = batch_size
        trial = experiment.new_batch_trial(generator_run=model.gen(n=n)).run()
    else:
        n = 1
        trial = experiment.new_trial(generator_run=model.gen(n=n)).run()

    experiment, trial, new_data = run_trial(experiment, trial, debug)

    return experiment, model, new_data


def run_trial(experiment, trial, debug):

    try:
        new_data = experiment._fetch_trial_data(trial.index)
        experiment.trials[trial.index].mark_completed()
    except (RuntimeError, IndexError) as e:
        experiment.trials[trial.index].mark_failed()
        new_data = Data()
        if debug:
            print(e)
    return experiment, trial, new_data


# TODO: delete hardcoded model folder name, loook in other parts of python
# files for hardcoded folder model name
def reload_net(exp, arm, classes, input_shape, net):
    net = net(exp.arms_by_name[arm].parameters, classes, input_shape)
    model_filename = "./models/" + arm + ".pth"
    net.load_state_dict(load(model_filename))
    return net
